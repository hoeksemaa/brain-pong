"""
Minimal 2-channel EOG display — raw signal sanity check.

Just detrends for DC and plots. No other filtering.
If channels are alive you'll see noise / drift / blinks / saccades.
If they're dead you'll see a flat line at 0.

Usage:
    source .venv/bin/activate
    python scripts/eog_display.py
    python scripts/eog_display.py --slots 4 5   # override if needed
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

BOARD_ID           = BoardIds.CERELOG_X8_BOARD
DEFAULT_PORT       = "/dev/cu.usbserial-1120"
DEFAULT_SLOTS      = [0, 1]
SECONDS_TO_DISPLAY = 10
UPDATE_INTERVAL_MS = 40
LPF_HZ             = 100.0
HPF_HZ             = 0.5
NOTCH_BANDS        = ((48.0, 52.0), (58.0, 62.0))

board         = None
sampling_rate = 0
window_size   = 0
data_buffer   = np.array([])


def full_filter(x_uv):
    """Matches filtered_plot.py exactly: detrend → LPF 100 → notch 50/60 → HPF 0.5."""
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size <= 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y,  sampling_rate, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sampling_rate, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sampling_rate, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",  default=DEFAULT_PORT)
    p.add_argument("--slots", nargs=2, type=int, default=DEFAULT_SLOTS)
    return p.parse_args()


def main():
    global board, sampling_rate, window_size, data_buffer

    args   = parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.port
    params.timeout     = 15
    board  = BoardShim(BOARD_ID, params)

    try:
        all_eeg       = BoardShim.get_eeg_channels(BOARD_ID)
        sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
        window_size   = SECONDS_TO_DISPLAY * sampling_rate

        row_L = all_eeg[args.slots[0]]
        row_R = all_eeg[args.slots[1]]

        print(f"SR: {sampling_rate} Hz")
        print(f"Slots {args.slots} → board rows {row_L}, {row_R}")
        print("Connecting…")

        board.prepare_session()
        board.start_stream(int(5 * 60 * sampling_rate))
        time.sleep(2)
        print("Streaming. Close window to stop.\n")

        n_rows      = BoardShim.get_num_rows(BOARD_ID)
        data_buffer = np.empty((n_rows, 0))

        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
        fig.suptitle(f"EOG — slots {args.slots} (board rows {row_L}, {row_R})")

        ax_l_raw, ax_r_raw, ax_l_filt, ax_r_filt = axes
        lines = [ax.plot([], [], lw=1, color=c)[0]
                 for ax, c in zip(axes, ['C0', 'C1', 'C0', 'C1'])]

        panels = [
            (ax_l_raw,  f"slot {args.slots[0]}  row {row_L}  — raw"),
            (ax_r_raw,  f"slot {args.slots[1]}  row {row_R}  — raw"),
            (ax_l_filt, f"slot {args.slots[0]}  row {row_L}  — filtered (detrend→LPF {LPF_HZ:g}→notch 50/60→HPF {HPF_HZ:g})"),
            (ax_r_filt, f"slot {args.slots[1]}  row {row_R}  — filtered (detrend→LPF {LPF_HZ:g}→notch 50/60→HPF {HPF_HZ:g})"),
        ]
        for ax, title in panels:
            ax.set_title(title)
            ax.set_ylabel("µV")
            ax.set_xlim(-SECONDS_TO_DISPLAY, 0)
            ax.set_ylim(-500, 500)
            ax.grid(True)

        axes[-1].set_xlabel('Time (s from now)')
        plt.tight_layout()

        def on_close(event):
            if board and board.is_prepared():
                board.stop_stream()
                board.release_session()

        fig.canvas.mpl_connect("close_event", on_close)

        ani = FuncAnimation(  # noqa: F841
            fig, _update,
            fargs=(lines, axes, row_L, row_R),
            interval=UPDATE_INTERVAL_MS,
            blit=False,
        )
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if board and board.is_prepared():
            board.release_session()


def _update(frame, lines, axes, row_L, row_R):
    global data_buffer

    try:
        new = board.get_board_data()
        if new.shape[1] > 0:
            data_buffer = np.hstack((data_buffer, new))
            if data_buffer.shape[1] > int(window_size * 1.5):
                data_buffer = data_buffer[:, -int(window_size * 1.5):]

        chunk = data_buffer[:, -window_size:]
        n     = chunk.shape[1]
        if n < 2:
            return

        # Fixed-length time axis — never changes shape, no stale-data artifacts
        t = np.linspace(-SECONDS_TO_DISPLAY, 0, window_size)

        def padded(arr_uv, apply_filter=False):
            """Return window_size array, NaN-padded on left, data on right."""
            out = np.full(window_size, np.nan)
            x = np.ascontiguousarray(arr_uv[-n:].astype(np.float64))
            if apply_filter:
                x = full_filter(x)
            else:
                DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
            out[-n:] = x
            return out

        l_uv = chunk[row_L] * 1e6
        r_uv = chunk[row_R] * 1e6
        traces = (padded(l_uv), padded(r_uv),
                  padded(l_uv, apply_filter=True), padded(r_uv, apply_filter=True))

        for line, ax, trace in zip(lines, axes, traces):
            line.set_data(t, trace)
            valid = trace[~np.isnan(trace)]
            peak  = max(float(np.abs(valid).max()) * 1.2, 50.0) if valid.size else 50.0
            ax.set_ylim(-peak, peak)

    except Exception as e:
        print(f"update error: {e}")


if __name__ == "__main__":
    main()
