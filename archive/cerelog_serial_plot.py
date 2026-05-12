"""
Live Cerelog USB serial plotter with no BrainFlow dependency.

This reuses cerelog_serial_monitor.py's direct serial handshake and packet
parser, then plots decoded ADS1299 channels from the raw firmware packets.

Usage:
    source .venv/bin/activate
    python cerelog_serial_plot.py
    python cerelog_serial_plot.py --channels 1 2 3 4 --derive-eog
    python cerelog_serial_plot.py --channels 5 6 --derive-eog
    python cerelog_serial_plot.py --no-handshake --baud 115200
"""

import argparse
import os
import time
from pathlib import Path

# Keep matplotlib's cache inside the repo/sandbox when the home cache is not
# writable. Must be set before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".cache" / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / ".cache").resolve()))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cerelog_serial_monitor import (
    ADS_CHANNELS,
    DEFAULT_PORT,
    FINAL_BAUD,
    INITIAL_BAUD,
    extract_packets,
    handshake_and_reopen,
    open_serial,
    read_available,
)


SAMPLE_RATE_HZ = 250
DEFAULT_WINDOW_S = 10.0
UPDATE_INTERVAL_MS = 40
Y_AXIS_PADDING_FACTOR = 1.2


def parse_args():
    parser = argparse.ArgumentParser(description="Live direct-serial Cerelog plotter")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=FINAL_BAUD)
    parser.add_argument("--initial-baud", type=int, default=INITIAL_BAUD)
    parser.add_argument("--handshake", dest="handshake", action="store_true", default=True)
    parser.add_argument("--no-handshake", dest="handshake", action="store_false")
    parser.add_argument("--boot-wait-s", type=float, default=5.0)
    parser.add_argument("--switch-wait-s", type=float, default=2.0)
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument("--update-ms", type=int, default=UPDATE_INTERVAL_MS)
    parser.add_argument("--channels", nargs="+", type=int, default=list(range(1, ADS_CHANNELS + 1)),
                        help="1-based packet channel numbers to plot; default: 1 2 3 4 5 6 7 8")
    parser.add_argument("--derive-eog", action="store_true",
                        help="add selected-channel diffs: second-first, and third-fourth when 4+ channels are selected")
    parser.add_argument("--center", choices=["median", "mean", "none"], default="median",
                        help="display centering only; does not alter saved/decoded data")
    parser.add_argument("--stats-every-s", type=float, default=1.0)
    return parser.parse_args()


def validate_channels(channels):
    bad = [ch for ch in channels if ch < 1 or ch > ADS_CHANNELS]
    if bad:
        raise ValueError(f"channels must be in 1..{ADS_CHANNELS}; got {bad}")
    seen = set()
    unique = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    return unique


def center_trace(x, mode):
    if x.size == 0 or mode == "none":
        return x
    if mode == "mean":
        return x - np.mean(x)
    return x - np.median(x)


def smooth_limits(current, trace):
    if trace.size == 0:
        target_min, target_max = -100.0, 100.0
    else:
        target_min = float(np.min(trace)) * Y_AXIS_PADDING_FACTOR
        target_max = float(np.max(trace)) * Y_AXIS_PADDING_FACTOR
    if np.isclose(target_min, target_max):
        target_min -= 1.0
        target_max += 1.0
    cur_min, cur_max = current
    smoothing = 0.12
    return (
        cur_min * (1.0 - smoothing) + target_min * smoothing,
        cur_max * (1.0 - smoothing) + target_max * smoothing,
    )


def open_board(args):
    if args.handshake:
        return handshake_and_reopen(
            args.port,
            args.initial_baud,
            args.baud,
            args.boot_wait_s,
            args.switch_wait_s,
        )
    print(f"opening {args.port} at {args.baud} baud without handshake")
    return open_serial(args.port, args.baud)


def main():
    args = parse_args()
    channels = validate_channels(args.channels)
    window_samples = max(2, int(round(args.window_s * SAMPLE_RATE_HZ)))

    print("Direct serial Cerelog plotter")
    print(f"Port: {args.port}")
    print(f"Channels: {channels}")
    print(f"Window: {args.window_s:g}s at assumed {SAMPLE_RATE_HZ} Hz")
    print("Close the plot window or press Ctrl-C in the terminal to stop.")

    fd = open_board(args)
    serial_buffer = bytearray()
    samples = np.empty((ADS_CHANNELS, 0), dtype=np.float64)
    sample_times = np.empty((0,), dtype=np.float64)
    y_limits = {}

    panel_specs = [("raw", ch - 1, f"CH{ch} raw") for ch in channels]
    if args.derive_eog and len(channels) >= 2:
        panel_specs.append((
            "diff",
            (channels[1] - 1, channels[0] - 1),
            f"Horizontal EOG = CH{channels[1]} - CH{channels[0]}",
        ))
    if args.derive_eog and len(channels) >= 4:
        panel_specs.append((
            "diff",
            (channels[2] - 1, channels[3] - 1),
            f"Vertical EOG = CH{channels[2]} - CH{channels[3]}",
        ))

    for i in range(len(panel_specs)):
        y_limits[i] = (-200.0, 200.0)

    rows = int(np.ceil(len(panel_specs) / 2.0))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.0 + 2.3 * rows), sharex=True)
    fig.suptitle("Cerelog Direct USB Serial", fontsize=14)
    axes_flat = np.atleast_1d(axes).flatten()
    axes_plot = axes_flat[:len(panel_specs)]
    for ax in axes_flat[len(panel_specs):]:
        ax.set_visible(False)

    lines = []
    for i, (_, _, title) in enumerate(panel_specs):
        line, = axes_plot[i].plot([], [], lw=1)
        lines.append(line)
        axes_plot[i].set_title(title)
        axes_plot[i].set_ylabel("uV")
        axes_plot[i].grid(True)
        axes_plot[i].set_xlim(-args.window_s, 0)
    fig.text(0.5, 0.04, 'Time (seconds from "now")', ha="center", va="center")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    stats = {
        "bytes": 0,
        "packets": 0,
        "dropped": 0,
        "bad_checksum": 0,
        "all_zero": 0,
        "last_t": time.monotonic(),
        "last_bytes": 0,
        "last_packets": 0,
    }

    def close_serial(event=None):
        nonlocal fd
        if fd is not None:
            os.close(fd)
            fd = None
            print("Serial port closed.")

    fig.canvas.mpl_connect("close_event", close_serial)

    def update(_frame):
        nonlocal serial_buffer, samples, sample_times
        if fd is None:
            return lines

        chunk = read_available(fd, timeout_s=0.001, max_bytes=8192)
        while chunk:
            stats["bytes"] += len(chunk)
            serial_buffer.extend(chunk)
            packets, dropped = extract_packets(serial_buffer)
            stats["dropped"] += dropped
            if packets:
                new_cols = []
                new_times = []
                for _packet, decoded in packets:
                    stats["packets"] += 1
                    if not decoded["checksum_ok"]:
                        stats["bad_checksum"] += 1
                    if all(value == 0 for value in decoded["counts"]):
                        stats["all_zero"] += 1
                    new_cols.append(decoded["microvolts"])
                    new_times.append(decoded["board_ms"] / 1000.0)
                samples = np.hstack((samples, np.asarray(new_cols, dtype=np.float64).T))
                sample_times = np.concatenate((sample_times, np.asarray(new_times, dtype=np.float64)))
                if samples.shape[1] > window_samples:
                    samples = samples[:, -window_samples:]
                    sample_times = sample_times[-window_samples:]
            chunk = read_available(fd, timeout_s=0.0, max_bytes=8192)

        if samples.shape[1] < 2:
            return lines

        t = sample_times - sample_times[-1]
        if not np.all(np.isfinite(t)) or np.ptp(t) <= 0:
            t = np.linspace(-samples.shape[1] / SAMPLE_RATE_HZ, 0, samples.shape[1])

        for i, (kind, ch_idx, _title) in enumerate(panel_specs):
            if kind == "raw":
                trace = samples[ch_idx]
            elif kind == "diff":
                pos_idx, neg_idx = ch_idx
                trace = samples[pos_idx] - samples[neg_idx]
            else:
                raise ValueError(f"unknown panel kind: {kind}")

            display_trace = center_trace(trace, args.center)
            lines[i].set_data(t, display_trace)
            recent_n = min(display_trace.size, int(round(4.0 * SAMPLE_RATE_HZ)))
            recent = display_trace[-recent_n:]
            y_limits[i] = smooth_limits(y_limits[i], recent)
            axes_plot[i].set_ylim(*y_limits[i])

        now = time.monotonic()
        if now - stats["last_t"] >= args.stats_every_s:
            dt = now - stats["last_t"]
            bps = (stats["bytes"] - stats["last_bytes"]) / dt
            pps = (stats["packets"] - stats["last_packets"]) / dt
            print(
                f"[stats] bytes/s={bps:8.1f} packets/s={pps:6.1f} "
                f"packets={stats['packets']} dropped={stats['dropped']} "
                f"bad_checksum={stats['bad_checksum']} all_zero_packets={stats['all_zero']}"
            )
            stats["last_t"] = now
            stats["last_bytes"] = stats["bytes"]
            stats["last_packets"] = stats["packets"]

        return lines

    try:
        ani = FuncAnimation(fig, update, interval=args.update_ms, blit=False)
        plt.show()
    finally:
        close_serial()


if __name__ == "__main__":
    main()
