"""
Real-time EOG plotter — the gold-standard preflight tool for an EOG session.

Same structure as filtered_plot.py (the EEG version) but tuned for
electrooculography:
  - 4 active electrodes (defaults: left canthus, right canthus, above-eye, below-eye)
  - EOG-friendly filter chain (HPF 0.5 Hz, LPF 30 Hz, 50/60 Hz notch)
  - 6-panel layout: 4 raw + 2 derived (horizontal + vertical EOG diffs)
  - adaptive y-axis (smoothed)

Hardware reminder: SRB1 (reference) on one mastoid + BIAS on the other mastoid
or forehead are still required. Without them, channels are meaningless / saturated.

Usage:
    source .venv/bin/activate
    python filtered_plot_eog.py

Sanity check sequence once it's running:
  1. Look straight ahead, hold still — all channels should be flat-ish
  2. Look hard LEFT for 2 s     — horizontal trace should swing one way
  3. Look hard RIGHT for 2 s    — horizontal trace should swing the other way
  4. Blink 5 times              — vertical trace should show 5 sharp peaks
  5. If any fail → reseat that electrode before trusting any data.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# --- Configuration ---
BOARD_ID = BoardIds.CERELOG_X8_BOARD
SECONDS_TO_DISPLAY    = 10
UPDATE_INTERVAL_MS    = 40
Y_AXIS_PADDING_FACTOR = 1.2

# Which Cerelog EEG-channel slots have your EOG electrodes wired into them.
# Maps: human-readable name → index into BoardShim.get_eeg_channels(BOARD_ID).
# Edit if you wire differently. Order matters: the derived "horizontal" trace
# is right_outer − left_outer, and "vertical" is above − below.
EOG_MONTAGE = [
    ('L outer (left canthus)',  0),
    ('R outer (right canthus)', 1),
    ('V above (above eye)',     2),
    ('V below (below eye)',     3),
]

# EOG-specific filter chain. Eyes move slowly — relevant bandwidth is
# fractional Hz to ~30 Hz, with DC drift dominating below that.
EOG_HPF_HZ   = 0.5    # kill long-term drift while preserving slow saccades
EOG_LPF_HZ   = 30.0   # well above eye-movement bandwidth
NOTCH_BANDS  = ((48.0, 52.0), (58.0, 62.0))  # 50/60 Hz line noise

# --- Globals ---
board         = None
sampling_rate = 0
window_size   = 0
data_buffer   = np.array([])
y_limits      = {}


def main():
    global board, sampling_rate, window_size, data_buffer, y_limits

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-1120"
    params.timeout     = 15
    board = BoardShim(BOARD_ID, params)

    try:
        all_eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        sampling_rate    = BoardShim.get_sampling_rate(BOARD_ID)
        window_size      = SECONDS_TO_DISPLAY * sampling_rate
        if sampling_rate <= 0:
            raise BrainFlowError("Could not get a valid sampling rate from the board.", 0)

        # Resolve montage to actual channel indices
        ch_indices = [all_eeg_channels[idx] for _, idx in EOG_MONTAGE]
        ch_names   = [name for name, _ in EOG_MONTAGE]

        for i in range(6):  # 4 raw + 2 derived
            y_limits[i] = (-200, 200)  # μV — will adapt

        print(f"Connecting to {board.get_board_descr(BOARD_ID)['name']}...")
        print(f"Sampling rate: {sampling_rate} Hz")
        print("EOG montage:")
        for name, slot in EOG_MONTAGE:
            print(f"  {name:30s} → board ch {all_eeg_channels[slot]}")
        print("Reminder: SRB1 (reference) + BIAS still need to be connected.")
        board.prepare_session()
        print("\nStarting stream — close the plot window to stop.")
        board.start_stream(5 * 60 * sampling_rate)
        time.sleep(2)

        num_board_channels = BoardShim.get_num_rows(BOARD_ID)
        data_buffer = np.empty((num_board_channels, 0))

        # 6-panel layout: 4 raw on top (2x2), 2 derived below
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle('Real-Time EOG (4 raw + 2 derived diffs)', fontsize=14)
        axes_flat = axes.flatten()

        panel_titles = ch_names + ['Horizontal EOG (R − L)', 'Vertical EOG (above − below)']
        lines = []
        for i, ax in enumerate(axes_flat):
            line, = ax.plot([], [], lw=1)
            lines.append(line)
            ax.set_title(panel_titles[i])
            ax.set_ylabel('μV')
            ax.grid(True)
            ax.set_xlim(-SECONDS_TO_DISPLAY, 0)

        fig.text(0.5, 0.04, 'Time (Seconds from "Now")', ha='center', va='center')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        def on_close(event):
            print("Plot window closed — stopping stream...")
            if board and board.is_prepared():
                board.stop_stream()
                board.release_session()
            print("Session released. Exiting.")

        fig.canvas.mpl_connect('close_event', on_close)

        ani = FuncAnimation(
            fig, update_plot,
            fargs=(lines, axes_flat, ch_indices),
            interval=UPDATE_INTERVAL_MS, blit=False,
        )
        plt.show()

    except Exception as e:
        print(f"An error occurred in main(): {e}")
    finally:
        if board and board.is_prepared():
            board.release_session()


def update_plot(frame, lines, axes, ch_indices):
    """Pull latest data, filter the 4 EOG channels, derive 2 diff channels,
    update each panel, and adaptively rescale y-axes."""
    global data_buffer, y_limits

    try:
        new_data = board.get_board_data()
        if new_data.shape[1] > 0:
            data_buffer = np.hstack((data_buffer, new_data))
            buf_limit = int(window_size * 1.5)
            if data_buffer.shape[1] > buf_limit:
                data_buffer = data_buffer[:, -buf_limit:]

        plot_data = data_buffer[:, -window_size:]
        n_points  = plot_data.shape[1]
        if n_points < 2:
            return

        # Filter each of the 4 active EOG channels (per data-integrity rules:
        # operate on a copy; never mutate the underlying buffer)
        filtered = []
        for ch_idx in ch_indices:
            x = np.ascontiguousarray((plot_data[ch_idx] * 1e6).astype(np.float64))
            if x.size > 20:
                DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
                DataFilter.perform_lowpass(x,  sampling_rate, EOG_LPF_HZ, 4, FilterTypes.BUTTERWORTH, 0)
                DataFilter.perform_highpass(x, sampling_rate, EOG_HPF_HZ, 4, FilterTypes.BUTTERWORTH, 0)
                for lo, hi in NOTCH_BANDS:
                    DataFilter.perform_bandstop(x, sampling_rate, lo, hi, 3, FilterTypes.BUTTERWORTH, 0)
            filtered.append(x)

        # Derived channels (montage order: 0=L, 1=R, 2=above, 3=below)
        horiz = filtered[1] - filtered[0]   # R outer − L outer  (positive = looking right)
        vert  = filtered[2] - filtered[3]   # above − below       (positive = looking up)

        all_traces = filtered + [horiz, vert]

        time_vec_full = np.linspace(-SECONDS_TO_DISPLAY, 0, window_size)
        time_vec      = time_vec_full[-n_points:]

        for i, (line, ax, trace) in enumerate(zip(lines, axes, all_traces)):
            if np.isnan(trace).any():
                print(f"warn: NaN in panel {i}, skipping update")
                continue
            centered = trace - np.mean(trace)
            line.set_data(time_vec, centered)

            # Adaptive y-axis: scale to recent ~4 s with smoothing
            recent_n = int(4.0 * sampling_rate)
            recent   = centered[-recent_n:] if centered.size > recent_n else centered
            if recent.size > 0:
                target_max = float(np.max(recent)) * Y_AXIS_PADDING_FACTOR
                target_min = float(np.min(recent)) * Y_AXIS_PADDING_FACTOR
            else:
                target_max, target_min = 100.0, -100.0
            if np.isclose(target_max, target_min):
                target_max += 1; target_min -= 1
            cur_min, cur_max = y_limits[i]
            smoothing = 0.1
            new_max = cur_max * (1 - smoothing) + target_max * smoothing
            new_min = cur_min * (1 - smoothing) + target_min * smoothing
            ax.set_ylim(new_min, new_max)
            y_limits[i] = (new_min, new_max)

    except Exception as e:
        print(f"!!! ERROR IN update_plot: {e}")


if __name__ == "__main__":
    main()
