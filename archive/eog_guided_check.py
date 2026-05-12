"""
Guided EOG sanity check for BrainPong.

This records short CENTER / LEFT / RIGHT / BLINK segments and prints objective
stats for the active EOG rows. Use it when the scrolling plot is hard to judge.

Usage:
    source .venv/bin/activate
    python eog_guided_check.py

Default slots match the live ESP-EEG rows observed on 2026-04-29:
BrainFlow EEG slots 4-7, i.e. the fifth through eighth entries returned by
BoardShim.get_eeg_channels().
"""

import argparse
import time

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


BOARD_ID = BoardIds.CERELOG_X8_BOARD
DEFAULT_PORT = "/dev/cu.usbserial-1120"
DEFAULT_SLOTS = [4, 5, 6, 7]
RAIL_VOLTS = 0.1875


def collect_segment(board, seconds, sampling_rate):
    n_samples = int(round(seconds * sampling_rate))
    time.sleep(seconds)
    return board.get_current_board_data(n_samples)


def summarize_trace(name, x):
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return f"{name}: no finite samples"
    span = float(np.max(finite) - np.min(finite))
    std = float(np.std(finite))
    mean = float(np.mean(finite))
    rail_frac = float(np.mean(np.abs(finite) > RAIL_VOLTS * 0.98))
    return (
        f"{name}: mean={mean * 1e6:9.0f}uV  std={std * 1e6:8.0f}uV  "
        f"span={span * 1e6:9.0f}uV  rail={rail_frac * 100:5.1f}%"
    )


def segment_features(data, rows):
    traces = [data[row].astype(float) for row in rows]
    heog = traces[1] - traces[0]
    veog = traces[2] - traces[3] if len(traces) >= 4 else None
    return traces, heog, veog


def wait_for_enter(prompt):
    input(f"\n{prompt}\nPress Enter to record...")


def main():
    parser = argparse.ArgumentParser(description="Guided EOG signal-quality check")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--slots", nargs="+", type=int, default=DEFAULT_SLOTS,
                        help="zero-based indices into BoardShim.get_eeg_channels(); default: 4 5 6 7")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_port = args.port
    params.timeout = 15

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
    rows = [eeg_channels[slot] for slot in args.slots]

    print(f"port: {args.port}")
    print(f"sampling_rate: {sampling_rate} Hz")
    print(f"BrainFlow EEG rows: {eeg_channels}")
    print(f"Using zero-based slots {args.slots} -> rows {rows}")
    print("Expected order: L outer, R outer, above eye, below eye")

    board = BoardShim(BOARD_ID, params)
    segments = {}
    try:
        board.prepare_session()
        board.start_stream(45000)
        print("\nStream started. Let the board settle for 2 seconds...")
        time.sleep(2.0)

        for label, prompt in [
            ("center", "CENTER: relax face, look straight ahead"),
            ("left", "LEFT: hold a strong left gaze"),
            ("center2", "CENTER AGAIN: return to center"),
            ("right", "RIGHT: hold a strong right gaze"),
            ("blink", "BLINK: blink deliberately several times"),
        ]:
            wait_for_enter(prompt)
            segments[label] = collect_segment(board, args.seconds, sampling_rate)
            print(f"recorded {label}")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()

    print("\n=== Raw Channel Stats ===")
    for label, data in segments.items():
        print(f"\n[{label}]")
        traces, heog, veog = segment_features(data, rows)
        for i, trace in enumerate(traces):
            print(summarize_trace(f"raw{i + 1}", trace))
        print(summarize_trace("HEOG R-L", heog))
        if veog is not None:
            print(summarize_trace("VEOG A-B", veog))

    center_heog = segment_features(segments["center"], rows)[1]
    left_heog = segment_features(segments["left"], rows)[1]
    right_heog = segment_features(segments["right"], rows)[1]
    center_std = float(np.std(center_heog))
    left_mean = float(np.mean(left_heog))
    right_mean = float(np.mean(right_heog))
    separation = abs(right_mean - left_mean)
    snrish = separation / center_std if center_std > 0 else float("inf")

    print("\n=== Horizontal EOG Summary ===")
    print(f"center std:       {center_std * 1e6:.0f} uV")
    print(f"left mean:        {left_mean * 1e6:.0f} uV")
    print(f"right mean:       {right_mean * 1e6:.0f} uV")
    print(f"L/R separation:   {separation * 1e6:.0f} uV")
    print(f"separation/std:   {snrish:.2f}x")

    if snrish >= 5.0:
        print("verdict: GOOD horizontal separation for a threshold detector")
    elif snrish >= 2.0:
        print("verdict: WEAK but probably salvageable; improve contact/strain relief")
    else:
        print("verdict: NOT READY; debug channel mapping/contact/reference before filters")


if __name__ == "__main__":
    main()
