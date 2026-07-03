"""
replay_eog.py — stream recorded npz through the GAME's exact detector, offline.

The eval suite (bench.py / eval_simple.py) scores *per-trial window* classification
with the detect.py functions. This is different: it replays the LIVE glance-pair
state machine the pong game actually runs (`eog_core._run_eog_sm`) — same 0.5 s
window, 0.1 s poll cadence, 5 s calibration, refractory — so we can measure, with
NO board, how the real game would behave on a recording:

  • commands actually fired (with timestamps + direction)
  • direction-match vs the LEFT/RIGHT cues + firing latency   (cued recordings)
  • FALSE-FIRE rate                                            (flat / noisy / free runs)

That last metric is the tournament-critical one: a paddle that twitches on noise
loses. The known oscillation blind-spot (eog_core docstring) shows up here as
false fires on the railing/noisy recordings — and any fix can be verified offline.

Usage:
    source .venv/bin/activate
    python scripts/replay_eog.py                      # all data/eog/*.npz
    python scripts/replay_eog.py data/eog/X.npz -v    # one file, print every fire
"""
import argparse
import io
import glob
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from brainpong.eog_core import (
    eog_diff, _eog_filter, _make_eog_state, _run_eog_sm, EOG_BASELINE_S,
)

# Game-loop cadence — mirrors pong_game_brainflow.py (_poll_eog). These are
# poll/window timing, NOT detector constants (those live in eog_core).
EOG_POLL_S   = 0.1     # state machine advances every 0.1 s
EOG_SETTLE_S = 0.4     # IIR warm-up prepended to each filtered window
MATCH_WINDOW_S = 3.0   # a fire is "explained" by a cue this recent before it

REC_DIR = Path(__file__).resolve().parent.parent / "data" / "eog"


def replay(path):
    """Stream one recording through the game detector exactly as the game would."""
    d = np.load(path, allow_pickle=True)
    eeg = d["eeg"]
    sr  = int(d["sample_rate"][0])
    chL = int(d["eog_ch_L"][0]); chR = int(d["eog_ch_R"][0])
    diff = eog_diff(eeg, chR, chL)                      # full recording, µV, R−L

    n_settle = int(EOG_SETTLE_S * sr)
    n_new    = max(1, int(EOG_POLL_S * sr))
    win      = n_settle + n_new

    st = _make_eog_state()
    st["sr"], st["ch_L"], st["ch_R"] = sr, chL, chR

    fires = []                                          # (t_sec, 'LEFT'|'RIGHT')
    with redirect_stdout(io.StringIO()):                # silence per-tick prints
        end = win
        while end <= diff.size:
            window   = diff[end - win:end]
            filtered = _eog_filter(window.copy(), sr)
            new_sig  = filtered[-n_new:]
            now      = end / sr                         # uniform n/fs clock
            res = _run_eog_sm(st, new_sig, now)
            if res is not None:
                fires.append((now, res["command"]))
            end += n_new

    evs = d["event_samples"] if "event_samples" in d else np.array([])
    evl = d["event_labels"]  if "event_labels"  in d else np.array([])
    cues = [(int(s) / sr, str(l)) for s, l in zip(evs, evl) if str(l) in ("LEFT", "RIGHT")]

    return {
        "id": Path(path).stem, "sr": sr, "dur": diff.size / sr,
        "sigma": st["baseline_sigma"], "calibrated": st["sm"] != "CALIBRATING",
        "fires": fires, "cues": cues,
    }


def score(r):
    """Match each fire to the most recent cue within MATCH_WINDOW_S before it."""
    matched, lat, unexplained = 0, [], 0
    for ft, fc in r["fires"]:
        recent = [(ct, cl) for ct, cl in r["cues"] if 0 <= ft - ct <= MATCH_WINDOW_S]
        if recent:
            ct, cl = recent[-1]
            if cl == fc:
                matched += 1; lat.append(ft - ct)
            else:
                unexplained += 1                       # fired, wrong direction
        else:
            unexplained += 1                           # fired with no cue → false
    return matched, lat, unexplained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="npz files (default: all data/eog/*.npz)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every fire")
    args = ap.parse_args()

    paths = args.files or sorted(glob.glob(str(REC_DIR / "*.npz")))
    print(f"replaying {len(paths)} recording(s) through the game detector "
          f"(glance-pair, {EOG_POLL_S*1000:.0f}ms poll, {EOG_BASELINE_S:.0f}s calib)\n")
    print(f"{'recording':<26} {'σ µV':>7} {'cues':>5} {'fires':>6} {'match':>6} "
          f"{'lat':>6}  {'false':>5}")
    print("-" * 72)

    cued_match, cued_total, all_lat, false_total, false_secs = 0, 0, [], 0, 0.0
    for p in paths:
        r = replay(p)
        m, lat, unexp = score(r)
        ncue = len(r["cues"]); nfire = len(r["fires"])
        sig = f"{r['sigma']:.1f}" if r["sigma"] is not None else "  —"
        if not r["calibrated"]:
            print(f"{r['id']:<26} {sig:>7} {'—':>5} {'calib-only (recording < ~6s)':>30}")
            continue
        if ncue:                                       # cued recording → accuracy
            mr = f"{100*m/ncue:.0f}%"
            ml = f"{np.median(lat):.2f}s" if lat else "  —"
            cued_match += m; cued_total += ncue; all_lat += lat
            print(f"{r['id']:<26} {sig:>7} {ncue:>5} {nfire:>6} {mr:>6} {ml:>6}  {unexp:>5}")
        else:                                          # free/noisy → all fires are false
            false_total += nfire; false_secs += r["dur"]
            print(f"{r['id']:<26} {sig:>7} {'0':>5} {nfire:>6} {'—':>6} {'—':>6}  {nfire:>5}")
        if args.verbose and r["fires"]:
            for ft, fc in r["fires"]:
                print(f"      {ft:7.2f}s  {fc}")

    print("-" * 72)
    if cued_total:
        ml = f"{np.median(all_lat):.2f}s" if all_lat else "—"
        print(f"CUED:  {cued_match}/{cued_total} cues hit "
              f"({100*cued_match/cued_total:.0f}%), median latency {ml}")
    if false_secs:
        print(f"FALSE: {false_total} phantom fires over {false_secs/60:.1f} min of "
              f"non-cued signal ({false_total/(false_secs/60):.1f}/min)")
    print("\nNote: cued runs are single-gaze holds; the game wants glance-PAIRs "
          "(look one way, then back). A look-and-return reads as a pair, so fires\n"
          "cluster on cue→rest transitions. 'false' = fired with no recent matching cue.")


if __name__ == "__main__":
    main()
