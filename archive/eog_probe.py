"""
Raw Cerelog EOG/EEG stream probe.

Use this when the live plot looks flat or zeroed. It prints raw channel stats
before any filtering, centering, CCA, or plotting, so we can tell whether the
board is streaming real samples and whether the expected rows are changing.

Useful commands:
    source .venv/bin/activate
    python eog_probe.py
    python eog_probe.py --attempts 10 --seconds 3
    python eog_probe.py --attempts 10 --seconds 3 --save-dir diagnostics
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


BOARD_ID = BoardIds.CERELOG_X8_BOARD
DEFAULT_PORT = "/dev/cu.usbserial-1120"
DEFAULT_RAIL_VOLTS = 0.1875


def finite_view(x):
    return x[np.isfinite(x)]


def summarize_channel(x, near_zero_eps, rail_volts):
    finite = finite_view(x.astype(float))
    if finite.size == 0:
        return None
    return {
        "n": int(finite.size),
        "finite_frac": float(finite.size / max(1, x.size)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "span": float(np.max(finite) - np.min(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "near_zero_frac": float(np.mean(np.abs(finite) <= near_zero_eps)),
        "rail_frac": float(np.mean(np.abs(finite) >= rail_volts * 0.98)),
        "unique_first_1k": int(np.unique(finite[: min(1000, finite.size)]).size),
    }


def channel_status(stats, near_zero_eps):
    if stats is None:
        return "NO FINITE"
    if stats["near_zero_frac"] >= 0.99:
        return "ZERO"
    if stats["std"] <= near_zero_eps and stats["span"] <= near_zero_eps:
        return "FLAT"
    if stats["rail_frac"] > 0.2:
        return "RAIL"
    if stats["unique_first_1k"] <= 1:
        return "FLAT"
    return "OK"


def summarize_timestamps(data, sampling_rate):
    try:
        ts_row = BoardShim.get_timestamp_channel(BOARD_ID)
    except Exception:
        return {"row": None, "ok": False, "reason": "no timestamp row"}
    if ts_row >= data.shape[0]:
        return {"row": ts_row, "ok": False, "reason": "timestamp row outside matrix"}
    ts = finite_view(data[ts_row].astype(float))
    if ts.size < 2:
        return {"row": ts_row, "ok": False, "reason": "fewer than 2 finite timestamps"}
    dt = np.diff(ts)
    positive_dt = dt[dt > 0]
    if positive_dt.size == 0:
        return {"row": ts_row, "ok": False, "reason": "timestamps did not advance"}
    span_s = float(ts[-1] - ts[0])
    observed_hz = float((ts.size - 1) / span_s) if span_s > 0 else 0.0
    expected_dt = 1.0 / float(sampling_rate) if sampling_rate else 0.0
    return {
        "row": int(ts_row),
        "ok": True,
        "n": int(ts.size),
        "span_s": span_s,
        "observed_hz": observed_hz,
        "dt_median_ms": float(np.median(positive_dt) * 1000.0),
        "dt_min_ms": float(np.min(positive_dt) * 1000.0),
        "dt_max_ms": float(np.max(positive_dt) * 1000.0),
        "expected_dt_ms": float(expected_dt * 1000.0),
        "nonmonotonic_count": int(np.sum(dt <= 0)),
        "large_gap_count": int(np.sum(positive_dt > expected_dt * 2.5)) if expected_dt else 0,
    }


def print_channel_table(eeg_channels, data, args):
    print("\nslot row  status  mean(uV*)   std(uV*)  span(uV*)  zero%  rail%  unique")
    print("---- ---  ------  ---------  ---------  ---------  -----  -----  ------")
    summaries = []
    for slot, row in enumerate(eeg_channels):
        stats = summarize_channel(data[row], args.near_zero_eps, args.rail_volts)
        status = channel_status(stats, args.near_zero_eps)
        summaries.append((slot, row, status, stats))
        if stats is None:
            print(f"{slot:>4} {row:>3}  {status:6}")
            continue
        print(
            f"{slot:>4} {row:>3}  {status:6}  "
            f"{stats['mean'] * 1e6:9.1f}  "
            f"{stats['std'] * 1e6:9.1f}  "
            f"{stats['span'] * 1e6:9.1f}  "
            f"{stats['near_zero_frac'] * 100:5.1f}  "
            f"{stats['rail_frac'] * 100:5.1f}  "
            f"{stats['unique_first_1k']:>6}"
        )
    print("* uV columns assume Cerelog/BrainFlow rows are volts. Use raw values in saved npz if unit mapping is in doubt.")
    return summaries


def verdict_from(summaries, ts_summary, n_samples):
    if n_samples <= 0:
        return "NO_DATA"
    statuses = [status for _, _, status, _ in summaries]
    if statuses and all(status == "ZERO" for status in statuses):
        return "ALL_EEG_ZERO_WITH_TIMESTAMPS" if ts_summary.get("ok") else "ALL_EEG_ZERO"
    if statuses and all(status in ("ZERO", "FLAT") for status in statuses):
        return "ALL_EEG_FLAT_WITH_TIMESTAMPS" if ts_summary.get("ok") else "ALL_EEG_FLAT"
    if any(status == "RAIL" for status in statuses):
        return "SOME_CHANNELS_RAILING"
    if any(status == "OK" for status in statuses):
        return "RAW_EEG_CHANGING"
    return "UNKNOWN"


def save_attempt(save_dir, attempt, data, metadata):
    if not save_dir:
        return None
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"cerelog-probe-{stamp}-attempt{attempt:02d}.npz"
    np.savez(out_path, data=data, metadata=np.array(metadata, dtype=object))
    return out_path


def run_attempt(attempt, args, eeg_channels, sampling_rate):
    params = BrainFlowInputParams()
    params.serial_port = args.port
    params.timeout = args.timeout

    expected_samples = int(round(args.seconds * sampling_rate))
    board = BoardShim(BOARD_ID, params)
    data = np.empty((0, 0))

    print(f"\n=== Attempt {attempt}/{args.attempts} ===")
    print(f"prepare/start on {args.port}; settle {args.settle_seconds:.1f}s; collect {args.seconds:.1f}s")

    try:
        board.prepare_session()
        board.start_stream(max(45000, expected_samples * 4))
        time.sleep(args.settle_seconds)
        try:
            board.get_board_data()
        except Exception as e:
            print(f"settle-buffer flush failed: {e}")
        time.sleep(args.seconds)
        data = board.get_board_data()
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()

    n_samples = int(data.shape[1]) if data.size else 0
    print(f"matrix shape: {data.shape}; expected about {expected_samples} samples")

    ts_summary = summarize_timestamps(data, sampling_rate) if data.size else {"ok": False, "reason": "no data"}
    if ts_summary.get("ok"):
        print(
            "timestamps: "
            f"row={ts_summary['row']} observed={ts_summary['observed_hz']:.2f}Hz "
            f"dt_med={ts_summary['dt_median_ms']:.3f}ms "
            f"dt_max={ts_summary['dt_max_ms']:.3f}ms "
            f"large_gaps={ts_summary['large_gap_count']} "
            f"nonmono={ts_summary['nonmonotonic_count']}"
        )
    else:
        print(f"timestamps: BAD ({ts_summary.get('reason', 'unknown')})")

    summaries = print_channel_table(eeg_channels, data, args) if data.size else []
    verdict = verdict_from(summaries, ts_summary, n_samples)
    print(f"verdict: {verdict}")
    if verdict.endswith("_WITH_TIMESTAMPS"):
        print("interpretation: samples/timestamps arrived, but the EEG payload is flat. Look upstream of Dash/DSP first.")

    metadata = {
        "attempt": attempt,
        "port": args.port,
        "seconds": args.seconds,
        "settle_seconds": args.settle_seconds,
        "sampling_rate": sampling_rate,
        "eeg_channels": list(eeg_channels),
        "timestamp": ts_summary,
        "verdict": verdict,
        "channel_statuses": [
            {"slot": slot, "row": row, "status": status, "stats": stats}
            for slot, row, status, stats in summaries
        ],
    }
    out_path = save_attempt(args.save_dir, attempt, data, metadata)
    if out_path:
        print(f"saved raw matrix: {out_path}")

    return {
        "attempt": attempt,
        "samples": n_samples,
        "verdict": verdict,
        "timestamp_ok": bool(ts_summary.get("ok")),
        "observed_hz": ts_summary.get("observed_hz"),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe raw Cerelog EEG/EOG rows")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--near-zero-eps", type=float, default=1e-12)
    parser.add_argument("--rail-volts", type=float, default=DEFAULT_RAIL_VOLTS)
    parser.add_argument("--save-dir", default=None, help="optional directory for raw .npz captures")
    parser.add_argument("--brainflow-log", action="store_true", help="enable verbose BrainFlow board logging")
    args = parser.parse_args()

    if args.brainflow_log:
        BoardShim.enable_dev_board_logger()

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)

    print(f"port: {args.port}")
    print(f"sampling_rate: {sampling_rate} Hz")
    print(f"BrainFlow EEG rows: {eeg_channels}")
    print(f"near-zero threshold: abs(x) <= {args.near_zero_eps:g}")

    results = []
    for attempt in range(1, args.attempts + 1):
        try:
            results.append(run_attempt(attempt, args, eeg_channels, sampling_rate))
        except Exception as e:
            print(f"\n=== Attempt {attempt}/{args.attempts} failed before summary ===")
            print(f"error: {e}")
            results.append({
                "attempt": attempt,
                "samples": 0,
                "verdict": "EXCEPTION",
                "timestamp_ok": False,
                "observed_hz": None,
            })
        if attempt < args.attempts:
            time.sleep(args.gap_seconds)

    if len(results) > 1:
        print("\n=== Attempt Summary ===")
        for result in results:
            hz = result["observed_hz"]
            hz_text = f"{hz:.2f}Hz" if isinstance(hz, (int, float)) else "n/a"
            print(
                f"{result['attempt']:>2}: {result['verdict']:<30} "
                f"samples={result['samples']:<5} ts={str(result['timestamp_ok']):<5} rate={hz_text}"
            )
        counts = {}
        for result in results:
            counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
        print("counts: " + json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
