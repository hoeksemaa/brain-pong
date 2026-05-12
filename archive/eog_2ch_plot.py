"""
Simple 2-channel horizontal EOG plotter.

Hardware setup:
  - active electrode 1: left outer canthus
  - active electrode 2: right outer canthus
  - SRB1/reference: forehead, mastoid, or earlobe
  - BIAS/ground: mastoid, earlobe, or forehead

This script ignores vertical/blink channels and plots:
  - left raw, display-centered
  - right raw, display-centered
  - raw horizontal EOG: right - left
  - filtered horizontal EOG
  - filtered horizontal EOG minus rolling baseline
  - horizontal EOG velocity

Default slots match Cerelog data channels 1 and 2:
BrainFlow EEG slots 0 and 1, i.e. the first and second entries returned by
BoardShim.get_eeg_channels().

Usage:
    source .venv/bin/activate
    python eog_2ch_plot.py

If your active rows differ:
    python eog_2ch_plot.py --slots 4 5
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


BOARD_ID = BoardIds.CERELOG_X8_BOARD
DEFAULT_PORT = "/dev/cu.usbserial-1120"
DEFAULT_SLOTS = [0, 1]
SECONDS_TO_DISPLAY = 10
UPDATE_INTERVAL_MS = 40
LPF_HZ = 20.0
NOTCH_BANDS = ((58.0, 62.0),)
Y_AXIS_PADDING_FACTOR = 1.2
BASELINE_SECONDS = 2.0


board = None
sampling_rate = 0
window_size = 0
data_buffer = np.array([])
y_limits = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple 2-channel horizontal EOG plotter")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--slots", nargs=2, type=int, default=DEFAULT_SLOTS,
                        help="zero-based indices into BoardShim.get_eeg_channels(); default: 0 1")
    parser.add_argument("--seconds", type=int, default=SECONDS_TO_DISPLAY)
    return parser.parse_args()


def to_uv(x):
    return np.ascontiguousarray(x.astype(np.float64) * 1e6)


def center_for_display(x):
    if x.size == 0:
        return x
    return x - np.median(x)


def filter_heog_for_display(x_uv):
    """Return a filtered copy in uV, preserving slow held-gaze deflections."""
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size <= 20:
        return y
    DataFilter.perform_lowpass(
        y, sampling_rate, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0
    )
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(
            y, sampling_rate, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0
        )
    DataFilter.perform_rolling_filter(y, 3, AggOperations.MEDIAN.value)
    return y


def rolling_baseline(x, seconds=2.0):
    """Use the recent median as a live center baseline."""
    n = max(1, int(round(seconds * sampling_rate)))
    recent = x[-n:] if x.size > n else x
    if recent.size == 0:
        return 0.0
    return float(np.median(recent))


def main():
    global board, sampling_rate, window_size, data_buffer, y_limits

    args = parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.port
    params.timeout = 15
    board = BoardShim(BOARD_ID, params)

    try:
        all_eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
        window_size = args.seconds * sampling_rate
        if sampling_rate <= 0:
            raise BrainFlowError("Could not get a valid sampling rate from the board.", 0)

        left_row = all_eeg_channels[args.slots[0]]
        right_row = all_eeg_channels[args.slots[1]]
        for i in range(6):
            y_limits[i] = (-1000, 1000)

        print(f"Connecting to {board.get_board_descr(BOARD_ID)['name']}...")
        print(f"Port: {args.port}")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"BrainFlow EEG rows: {all_eeg_channels}")
        print(f"Using slots {args.slots}:")
        print(f"  left outer canthus  -> row {left_row}")
        print(f"  right outer canthus -> row {right_row}")
        print("SRB1/reference and BIAS/ground should be connected separately.")

        board.prepare_session()
        print("\nStarting stream - close the plot window to stop.")
        board.start_stream(5 * 60 * sampling_rate)
        time.sleep(2)

        num_board_channels = BoardShim.get_num_rows(BOARD_ID)
        data_buffer = np.empty((num_board_channels, 0))

        fig, axes = plt.subplots(6, 1, figsize=(14, 13), sharex=True)
        fig.suptitle("2-Channel Horizontal EOG", fontsize=14)

        titles = [
            "Left outer raw (median-centered for display)",
            "Right outer raw (median-centered for display)",
            "HEOG raw = right - left",
            f"HEOG filtered (LPF {LPF_HZ:g} Hz + 60 Hz notch + 3-sample median)",
            f"HEOG filtered minus rolling median baseline ({BASELINE_SECONDS:g}s)",
            "HEOG velocity from filtered signal",
        ]
        lines = []
        for i, ax in enumerate(axes):
            line, = ax.plot([], [], lw=1)
            lines.append(line)
            ax.set_title(titles[i])
            ax.set_ylabel("uV")
            ax.grid(True)
            ax.set_xlim(-args.seconds, 0)
        axes[-1].set_xlabel('Time (seconds from "now")')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        def on_close(event):
            print("Plot window closed - stopping stream...")
            if board and board.is_prepared():
                board.stop_stream()
                board.release_session()
            print("Session released.")

        fig.canvas.mpl_connect("close_event", on_close)

        FuncAnimation(
            fig, update_plot,
            fargs=(lines, axes, left_row, right_row, args.seconds),
            interval=UPDATE_INTERVAL_MS,
            blit=False,
        )
        plt.show()

    except Exception as e:
        print(f"An error occurred in main(): {e}")
    finally:
        if board and board.is_prepared():
            board.release_session()


def update_plot(frame, lines, axes, left_row, right_row, seconds):
    global data_buffer, y_limits

    try:
        new_data = board.get_board_data()
        if new_data.shape[1] > 0:
            data_buffer = np.hstack((data_buffer, new_data))
            buffer_limit = int(window_size * 1.5)
            if data_buffer.shape[1] > buffer_limit:
                data_buffer = data_buffer[:, -buffer_limit:]

        plot_data = data_buffer[:, -window_size:]
        n_points = plot_data.shape[1]
        if n_points < 2:
            return

        left_raw = to_uv(plot_data[left_row])
        right_raw = to_uv(plot_data[right_row])
        heog_raw = right_raw - left_raw
        heog_filtered = filter_heog_for_display(heog_raw)
        heog_baseline = heog_filtered - rolling_baseline(heog_filtered, BASELINE_SECONDS)
        heog_velocity = np.gradient(heog_filtered) * sampling_rate if heog_filtered.size > 1 else heog_filtered

        traces = [
            center_for_display(left_raw),
            center_for_display(right_raw),
            center_for_display(heog_raw),
            center_for_display(heog_filtered),
            heog_baseline,
            center_for_display(heog_velocity),
        ]

        time_vec_full = np.linspace(-seconds, 0, window_size)
        time_vec = time_vec_full[-n_points:]

        for i, (line, ax, trace) in enumerate(zip(lines, axes, traces)):
            if np.isnan(trace).any():
                print(f"warn: NaN in panel {i}, skipping update")
                continue
            line.set_data(time_vec, trace)

            recent_n = int(4.0 * sampling_rate)
            recent = trace[-recent_n:] if trace.size > recent_n else trace
            if recent.size:
                target_min = float(np.min(recent)) * Y_AXIS_PADDING_FACTOR
                target_max = float(np.max(recent)) * Y_AXIS_PADDING_FACTOR
            else:
                target_min, target_max = -1000.0, 1000.0
            if np.isclose(target_min, target_max):
                target_min -= 1.0
                target_max += 1.0
            cur_min, cur_max = y_limits[i]
            smoothing = 0.1
            new_min = cur_min * (1 - smoothing) + target_min * smoothing
            new_max = cur_max * (1 - smoothing) + target_max * smoothing
            ax.set_ylim(new_min, new_max)
            y_limits[i] = (new_min, new_max)

    except Exception as e:
        print(f"!!! ERROR IN update_plot: {e}")


if __name__ == "__main__":
    main()
