"""
Real-time EOG plotter — the gold-standard preflight tool for an EOG session.

Same structure as filtered_plot.py (the EEG version) but tuned for
electrooculography:
  - 4 active electrodes by default (left canthus, right canthus, above-eye, below-eye)
  - can be reduced to 2 active electrodes by commenting the vertical pair
  - EOG-friendly filter chain (HPF 0.5 Hz, LPF 30 Hz, 50/60 Hz notch)
  - dynamic layout: raw channels + derived horizontal/vertical diffs when available
  - adaptive y-axis (smoothed)

Today's preferred Pong/debug montage is 6 total electrodes:
  - CH1: left outer canthus
  - CH2: right outer canthus
  - CH3: above one eye
  - CH4: below the same eye
  - SRB/reference: mastoid or earlobe
  - BIAS/ground: other mastoid or forehead

Hardware reminder: SRB1/reference + BIAS/ground are still required. Without
them, channels are often meaningless / saturated.

Usage:
    source .venv/bin/activate
    python eog_filtered_plot.py

Sanity check sequence once it's running:
  1. Look straight ahead, hold still — all channels should be flat-ish
  2. Look hard LEFT for 2 s     — horizontal trace should swing one way
  3. Look hard RIGHT for 2 s    — horizontal trace should swing the other way
  4. Blink 5 times              — raw channels may spike; vertical only exists
                                  if the optional vertical pair is configured
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

# Which Cerelog BrainFlow EEG-channel slots have your active EOG electrodes
# wired into.
# Maps: human-readable name -> index into BoardShim.get_eeg_channels(BOARD_ID).
# Edit if you wire differently.
#
# Order matters:
#   slot 0: left outer canthus
#   slot 1: right outer canthus
#   slot 2: optional above-eye electrode
#   slot 3: optional below-eye electrode
#
# Probe note, 2026-04-29: the ESP-EEG stream showed constant values on
# BrainFlow EEG slots 0-3 and live samples on slots 4-7. So the default below
# intentionally uses slots 4-7, even though the physical electrodes may feel
# like "channels 1-4" on the board/cable.
#
# For a 2-active-electrode horizontal-only setup, comment out the vertical pair:
# two active EOG electrodes + SRB/reference + BIAS/ground.
EOG_MONTAGE = [
    ('L outer (left canthus)',  4),
    ('R outer (right canthus)', 5),
    ('V above (above eye)',     6),
    ('V below (below eye)',     7),
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

        n_panels = len(ch_indices)
        if len(ch_indices) >= 2:
            n_panels += 1  # horizontal diff
        if len(ch_indices) >= 4:
            n_panels += 1  # vertical diff

        for i in range(n_panels):
            y_limits[i] = (-200, 200)  # μV — will adapt

        print(f"Connecting to {board.get_board_descr(BOARD_ID)['name']}...")
        print(f"Sampling rate: {sampling_rate} Hz")
        print("EOG montage:")
        for name, slot in EOG_MONTAGE:
            print(f"  {name:30s} → board ch {all_eeg_channels[slot]}")
        print("Reminder: SRB/reference + BIAS/ground still need to be connected.")
        board.prepare_session()
        print("\nStarting stream — close the plot window to stop.")
        board.start_stream(5 * 60 * sampling_rate)
        time.sleep(2)

        num_board_channels = BoardShim.get_num_rows(BOARD_ID)
        data_buffer = np.empty((num_board_channels, 0))

        rows = int(np.ceil(n_panels / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 + 2.4 * rows), sharex=True)
        fig.suptitle('Real-Time EOG', fontsize=14)
        axes_flat = np.atleast_1d(axes).flatten()
        axes_plot = axes_flat[:n_panels]
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        panel_titles = list(ch_names)
        if len(ch_indices) >= 2:
            panel_titles.append('Horizontal EOG (R - L)')
        if len(ch_indices) >= 4:
            panel_titles.append('Vertical EOG (above - below)')
        lines = []
        for i, ax in enumerate(axes_plot):
            line, = ax.plot([], [], lw=1)
            lines.append(line)
            ax.set_title(panel_titles[i])
            ax.set_ylabel('uV')
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
            fargs=(lines, axes_plot, ch_indices),
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

        all_traces = list(filtered)
        if len(filtered) >= 2:
            # Montage order: 0=L outer, 1=R outer. Sign may flip depending on
            # board/electrode polarity; the important thing is opposite
            # deflection for left vs. right gaze.
            all_traces.append(filtered[1] - filtered[0])
        if len(filtered) >= 4:
            all_traces.append(filtered[2] - filtered[3])

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
