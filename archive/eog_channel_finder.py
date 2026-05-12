"""
Live Cerelog channel finder.

Use this when a plot appears zeroed or the physical Cerelog channel labels do
not seem to match BrainFlow rows. It prints rolling raw stats for all 8 EEG
channels, before filtering or display-centering.

Suggested use:
  1. Connect SRB1/reference and BIAS/ground.
  2. Put one active electrode or jumper on a physical Cerelog input.
  3. Run this script.
  4. Touch/tap/wiggle only that one active electrode.
  5. The responsive row is the BrainFlow row/slot for that physical input.

Usage:
    source .venv/bin/activate
    python eog_channel_finder.py
"""

import argparse
import time

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


BOARD_ID = BoardIds.CERELOG_X8_BOARD
DEFAULT_PORT = "/dev/cu.usbserial-1120"
RAIL_VOLTS = 0.1875


def summarize(x):
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return None
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "span": float(np.max(finite) - np.min(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "rail": float(np.mean(np.abs(finite) > RAIL_VOLTS * 0.98)),
        "unique": int(np.unique(finite[: min(500, finite.size)]).size),
    }


def status_for(stats):
    if stats is None:
        return "NO FINITE"
    if stats["std"] <= 1e-12 and stats["span"] <= 1e-12:
        return "FLAT"
    if stats["rail"] > 0.2:
        return "RAIL"
    if stats["std"] > 1e-5 or stats["span"] > 5e-5:
        return "ACTIVE?"
    return "quiet"


def print_header(eeg_channels):
    print("\nslot row  status   mean(uV)    std(uV)   span(uV)   rail%  unique")
    print("---- ---  -------  ---------  ---------  ---------  -----  ------")


def main():
    parser = argparse.ArgumentParser(description="Live Cerelog EEG row/channel finder")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--interval-s", type=float, default=1.0)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_port = args.port
    params.timeout = 15

    board = BoardShim(BOARD_ID, params)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
    n_samples = max(1, int(round(args.window_s * sampling_rate)))

    print(f"port: {args.port}")
    print(f"sampling_rate: {sampling_rate} Hz")
    print(f"BrainFlow EEG rows: {eeg_channels}")
    print("Ctrl-C to stop.")

    try:
        board.prepare_session()
        board.start_stream(45000)
        time.sleep(2.0)

        while True:
            data = board.get_current_board_data(n_samples)
            print_header(eeg_channels)
            active_slots = []
            for slot, row in enumerate(eeg_channels):
                stats = summarize(data[row]) if data.size else None
                status = status_for(stats)
                if status in ("ACTIVE?", "RAIL"):
                    active_slots.append(slot)
                if stats is None:
                    print(f"{slot:>4} {row:>3}  {status:7}")
                    continue
                print(
                    f"{slot:>4} {row:>3}  {status:7}  "
                    f"{stats['mean'] * 1e6:9.0f}  "
                    f"{stats['std'] * 1e6:9.0f}  "
                    f"{stats['span'] * 1e6:9.0f}  "
                    f"{stats['rail'] * 100:5.1f}  "
                    f"{stats['unique']:>6}"
                )
            if active_slots:
                print(f"responsive/railed slots this window: {active_slots}")
            print("\nTip: touch one physical input at a time; the responsive slot is its software row.")
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()


if __name__ == "__main__":
    main()
