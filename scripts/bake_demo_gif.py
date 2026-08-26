#!/usr/bin/env python3
"""Bake the README demo clip by replaying a recorded 2-player match through the REAL game.

No hardware and no re-implementation of the game: each player's frozen `.npz` is
streamed into `pong_game_brainflow.py` through a mock board that serves the recording
in wall-clock time. The recorded eye signals therefore drive the real detector
(`eog_core`), the real physics callback and the real canvas renderer
(`assets/render.js`) — a headless browser plays the match back and records it, and
ffmpeg cuts the capture into the GIF.

What is real and what is not: the EOG, the glance detection, the paddle commands and
every pixel of the UI are the live game's own. The ball is not — no ball trajectory is
stored in a recording — so physics is re-simulated from the replayed commands. Scores
in the clip are the scores of THIS replay, not of the night itself.

The README clip — the last match of the 2026-08-17 tournament, Player Q vs Player U —
was baked in two steps: replay the whole match once, then cut the liveliest stretch out
of that one capture (re-cutting never re-plays, and the physics is random, so a re-run
is a different match).

    python scripts/bake_demo_gif.py --pair 20260817-201403 \
        --p1-name "Player Q" --p2-name "Player U" --capture /tmp/match.webm
    python scripts/bake_demo_gif.py --pair 20260817-201403 \
        --from-capture /tmp/match.webm --clip 24.05:20 --out assets/brainpong-demo.gif

Player names are the portal's pseudonyms (`scripts/bake_portal.py` assigns them; the
published mapping lives in `web/portal-data/`), never the real subject_id in the
filename — this clip is a public asset.

`--clip START:LEN` is in recording seconds (0 = the moment New Game was pressed, which
is also sample 0 of the npz), so a window picked from the signal in the viewer or from
`pipeline.replay` maps straight onto the clip. The first pass prints every score change
on the same axis, which is how a window with rallies in it gets found.
"""

import argparse
import contextlib
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "eog"
GAME_PATH = ROOT / "scripts" / "pong_game_brainflow.py"

# Capture geometry. The stage is responsive (render.js sizeLayout), so the viewport
# aspect alone fixes the composition: a 16:9 window gives the square field in the
# middle with the two score cards left and the two EOG feeds right.
VIEWPORT = {"width": 1280, "height": 720}


# ==============================================================================
# === Mock board ===============================================================
# ==============================================================================
class MockBoard:
    """A `BoardShim` stand-in that serves a frozen recording in wall-clock time.

    Only `get_current_board_data(n)` is ever called on a live board by the game's
    detector poll and waveform poll; both want "the newest n samples, channel-major,
    as a fresh copy". Sample 0 is served at `t0`, so a poll at wallclock T returns the
    window ending at recording second (T - t0) — the same alignment the board gave the
    game on the night, which is what makes the recorded events (calib_start /
    play_start) line up with the replay's own app-flow.
    """

    def __init__(self, eeg, sample_rate):
        self._eeg = np.ascontiguousarray(eeg, dtype=np.float64)
        self.sample_rate = int(sample_rate)
        self.t0 = None          # set by start(); until then the board reads as empty

    def start(self, t0):
        self.t0 = float(t0)

    def cursor(self):
        """Index of the newest sample available right now (0 before start)."""
        if self.t0 is None:
            return 0
        i = int((time.time() - self.t0) * self.sample_rate)
        return max(0, min(i, self._eeg.shape[1]))

    def exhausted(self):
        return self.cursor() >= self._eeg.shape[1]

    # ── the BoardShim surface the game actually touches ──
    def get_current_board_data(self, n):
        i = self.cursor()
        return self._eeg[:, max(0, i - int(n)):i].copy()

    def is_prepared(self):
        return False

    def stop_stream(self):
        pass

    def release_session(self):
        pass


# ==============================================================================
# === Recording pair ===========================================================
# ==============================================================================
def load_pair(stamp):
    """The two npz of one 2-player game, keyed by player slot."""
    paths = sorted(DATA_DIR.glob(f"{stamp}-*.npz"))
    if len(paths) != 2:
        sys.exit(f"expected exactly 2 recordings for {stamp}, found {len(paths)}: "
                 + ", ".join(p.name for p in paths))
    out = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        slot = str(d["player_slot"][0]) if "player_slot" in d.files else None
        if slot not in ("P1", "P2"):
            sys.exit(f"{p.name}: no player_slot — not a 2-player in-game recording")
        out[slot] = {
            "path": p,
            "subject": str(d["subject_id"][0]),
            "eeg": d["eeg"],
            "sr": int(d["sample_rate"][0]),
            "ch_L": int(d["eog_ch_L"][0]),
            "ch_R": int(d["eog_ch_R"][0]),
            "unix_start": float(d["unix_start"][0]),
            "events": {str(l): int(s) for l, s in zip(d["event_labels"], d["event_samples"])},
            "tuning": {k: float(d[k][0]) for k in
                       ("sigma_thr", "glance_window_s", "hpf_hz", "lpf_hz") if k in d.files},
            "detector": str(d["detector"][0]) if "detector" in d.files else "velocity",
        }
    if set(out) != {"P1", "P2"}:
        sys.exit(f"{stamp}: both recordings claim slot {list(out)[0]}")
    return out


def check_tuning(pair, game):
    """Warn if the live defaults differ from the settings the match was played with.

    The replay runs the game as it stands today; if a detector constant has moved since
    the recording, the paddles in the clip are not the paddles of that night.
    """
    live = {"sigma_thr": game.EOG_SIGMA_THR, "glance_window_s": game.GLANCE_WINDOW_S,
            "hpf_hz": game.EOG_HPF_HZ, "lpf_hz": game.EOG_LPF_HZ}
    for slot, rec in pair.items():
        for k, was in rec["tuning"].items():
            if abs(was - live[k]) > 1e-9:
                print(f"  ! {slot} {rec['subject']}: played at {k}={was:g}, "
                      f"replaying at {live[k]:g}")
        if rec["detector"] != "velocity":
            print(f"  ! {slot} {rec['subject']}: played with detector={rec['detector']}")


# ==============================================================================
# === The game, wired to the mock boards =======================================
# ==============================================================================
def import_game():
    """Import pong_game_brainflow as a module, in 2-player mode, without running main()."""
    argv = sys.argv
    sys.argv = ["pong_game_brainflow.py", "--2player"]
    try:
        spec = importlib.util.spec_from_file_location("pong_game_brainflow", GAME_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = argv


def wire_boards(game, pair):
    """Put the mock boards where the real ones would be, and mute in-game recording."""
    boards = {}
    for slot, st in (("P1", game.eog_state), ("P2", game.eog_state_p2)):
        rec = pair[slot]
        boards[slot] = MockBoard(rec["eeg"], rec["sr"])
        st["sr"] = rec["sr"]
        st["ch_L"] = rec["ch_L"]
        st["ch_R"] = rec["ch_R"]
    game.board_p1, game.board_p2 = boards["P1"], boards["P2"]
    game.sampling_rate = pair["P1"]["sr"]

    # A replay must never write into data/eog: the corpus is read-only ground truth.
    # These are looked up as module globals by the recording callback at call time,
    # so replacing them here disarms it without touching the game file.
    game._start_recording = lambda *a, **k: None
    game._stop_and_save_recording = lambda *a, **k: None
    game._log_event = lambda *a, **k: None
    return boards


def free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(game, port):
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    t = threading.Thread(
        target=lambda: game.app.run(host="127.0.0.1", port=port, debug=False,
                                    use_reloader=False, threaded=True),
        daemon=True)
    t.start()
    for _ in range(200):                       # wait for the server to answer
        try:
            with contextlib.closing(socket.create_connection(("127.0.0.1", port), 0.2)):
                return
        except OSError:
            time.sleep(0.05)
    sys.exit("Dash server did not come up")


# ==============================================================================
# === Capture ==================================================================
# ==============================================================================
def play_and_record(url, boards, pair, names, seconds, video_dir, headed=False):
    """Drive one full replay in a browser.

    Returns (video_path, lead_s, timeline). `lead_s` is how far into the capture the
    New Game click landed — the capture starts a beat earlier, at page load — so a clip
    window expressed in recording seconds can be shifted onto the video. `timeline` is
    the scoreboard read out of the DOM as the replay runs: [{t, p1, p2}] in recording
    seconds, one entry per score change, which is how a rally is found afterwards.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed, args=[
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--autoplay-policy=no-user-gesture-required",
            "--mute-audio",
        ])
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1,
                                  record_video_dir=str(video_dir),
                                  record_video_size=VIEWPORT)
        page = ctx.new_page()
        t_page = time.time()                   # ≈ first frame of the capture
        # NOT networkidle: the 16 ms physics interval keeps the page permanently busy.
        page.goto(url, wait_until="load")
        page.wait_for_selector("#pong-game-canvas")
        for sel, value in (("#p1-name", names["P1"]), ("#p2-name", names["P2"])):
            page.fill(sel, value)
            page.press(sel, "Tab")
        page.wait_for_timeout(600)

        # The recordings begin at the New Game click: start both streams at that instant,
        # each offset by its own board's unix_start so the two players stay in sync with
        # each other exactly as they were on the night.
        base = min(rec["unix_start"] for rec in pair.values())
        skew = max(rec["unix_start"] for rec in pair.values()) - base
        # Pre-fix recordings can carry a board-clock skew of minutes (see the save-path
        # note in pong_game_brainflow). A skew that large is a broken clock, not two
        # players who started apart, so fall back to aligning both on sample 0.
        use_unix = skew <= 1.0
        if not use_unix:
            print(f"  ! {skew:.1f}s unix_start skew — aligning on sample 0 instead")
        t0 = time.time()
        for slot, brd in boards.items():
            brd.start(t0 + (pair[slot]["unix_start"] - base if use_unix else 0.0))
        page.click("#start-button")

        timeline, last = [], None
        deadline = t0 + seconds
        while time.time() < deadline:
            page.wait_for_timeout(200)
            score = page.evaluate(
                "() => [document.getElementById('p1-score').textContent,"
                "       document.getElementById('p2-score').textContent]")
            if score != last:
                last = score
                timeline.append({"t": round(time.time() - t0, 2),
                                 "p1": score[0], "p2": score[1]})
            if all(b.exhausted() for b in boards.values()):
                print("  recording exhausted — stopping")
                break
        t_end = time.time()
        video = page.video
        ctx.close()
        path = Path(video.path())
        browser.close()
        return path, {"lead": t0 - t_page, "run": t_end - t0, "timeline": timeline}


# ==============================================================================
# === Encode ===================================================================
# ==============================================================================
def encode_gif(video, out, start, length, fps, width, crop=None, colors=128):
    """webm capture → looping GIF, via a per-clip palette.

    The UI is a handful of neon inks on near-black, so a clip-specific palette of 128
    colours reproduces it almost exactly and dithering only adds noise that costs
    megabytes — `dither=none` is both smaller and cleaner here than the usual bayer.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = []
    if crop:
        vf.append(f"crop={crop}")
    vf.append(f"fps={fps},scale={width}:-2:flags=lanczos")
    chain = ",".join(vf)
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length),
                        "-i", str(video), "-vf",
                        f"{chain},palettegen=stats_mode=diff:max_colors={colors}",
                        str(palette)], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length),
                        "-i", str(video), "-i", str(palette), "-lavfi",
                        f"{chain}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle",
                        "-loop", "0", str(out)], check=True)
    return out


def encode_mp4(video, out, start, length, width, crop=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = ([f"crop={crop}"] if crop else []) + [f"scale={width}:-2:flags=lanczos"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length),
                    "-i", str(video), "-vf", ",".join(vf), "-an",
                    "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-crf", "20", str(out)], check=True)
    return out


def video_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def replay_match(args, pair):
    """Run the match once and return (capture path, meta) — meta locates recording
    seconds inside the capture and lists every score change."""
    game = import_game()
    check_tuning(pair, game)
    boards = wire_boards(game, pair)

    n_s = pair["P1"]["eeg"].shape[1] / pair["P1"]["sr"]
    run_s = min(n_s, args.seconds) if args.seconds else n_s
    print(f"replaying {run_s:.0f}s of a {n_s:.0f}s recording")

    port = free_port()
    serve(game, port)
    out_dir = Path(args.capture).parent if args.capture else Path(
        tempfile.mkdtemp(prefix="brainpong-replay-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    video, meta = play_and_record(f"http://127.0.0.1:{port}/", boards, pair,
                                  {"P1": args.p1_name, "P2": args.p2_name},
                                  run_s, out_dir, headed=args.headed)

    # Playwright's video covers [page open, context close]; map recording seconds onto it
    # proportionally rather than trusting either clock alone.
    dur = video_duration(video)
    span = meta["lead"] + meta["run"]
    meta["scale"] = dur / span if span > 0 else 1.0
    meta["video_lead"] = meta["lead"] * meta["scale"]
    meta["duration"] = dur
    if args.capture:
        dest = Path(args.capture)
        shutil.move(str(video), dest)
        video = dest
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"capture: {video} ({video.stat().st_size / 1e6:.1f} MB), "
          f"{dur:.1f}s, lead-in {meta['video_lead']:.1f}s")
    print("score timeline (recording seconds):")
    for e in meta["timeline"]:
        print(f"    {e['t']:7.2f}s  {e['p1']} : {e['p2']}")
    return video, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True,
                    help="recording stamp shared by the two npz, e.g. 20260817-201403")
    ap.add_argument("--p1-name", default="Player 1", help="name shown on the TOP (purple) card")
    ap.add_argument("--p2-name", default="Player 2", help="name shown on the BOTTOM (yellow) card")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop the replay early (default: play the whole recording)")
    ap.add_argument("--clip", default=None, metavar="START:LEN",
                    help="clip window in recording seconds (default: 20 s from play_start)")
    ap.add_argument("--fps", type=int, default=13, help="GIF frame rate (default 13)")
    ap.add_argument("--width", type=int, default=640, help="output width (default 640)")
    ap.add_argument("--colors", type=int, default=128, help="GIF palette size (default 128)")
    ap.add_argument("--crop", default=None, help="ffmpeg crop=W:H:X:Y applied before scaling")
    ap.add_argument("--out", default=None, help="write the GIF here (repo-relative)")
    ap.add_argument("--also-mp4", default=None, help="write an mp4 of the same clip here")
    ap.add_argument("--capture", default=None,
                    help="keep the full-match capture here (webm; .json holds its timeline)")
    ap.add_argument("--from-capture", default=None, metavar="WEBM",
                    help="skip the replay and cut from an earlier --capture instead")
    ap.add_argument("--headed", action="store_true", help="watch the replay in a real window")
    args = ap.parse_args()

    pair = load_pair(args.pair)
    print(f"pair {args.pair}: "
          + "  ".join(f"{s}={pair[s]['subject']}({pair[s]['path'].name})" for s in ("P1", "P2")))

    if args.from_capture:
        video = Path(args.from_capture)
        meta = json.loads(video.with_suffix(".json").read_text())
        print(f"cutting from {video} ({meta['duration']:.1f}s, "
              f"lead-in {meta['video_lead']:.1f}s)")
    else:
        video, meta = replay_match(args, pair)

    if not args.out:
        if not (args.capture or args.from_capture):
            shutil.rmtree(video.parent, ignore_errors=True)
        return

    # Clip window: recording seconds, 0 = the New Game click = npz sample 0.
    play_start = pair["P1"]["events"].get("play_start", 0) / pair["P1"]["sr"]
    if args.clip:
        start_s, length_s = (float(v) for v in args.clip.split(":"))
    else:
        start_s, length_s = play_start + 1.0, 20.0
    v_start = meta["video_lead"] + start_s * meta["scale"]
    v_len = length_s * meta["scale"]
    print(f"clip {start_s:.1f}–{start_s + length_s:.1f}s of the recording "
          f"→ {v_start:.1f}s +{v_len:.1f}s of the capture")

    gif = encode_gif(video, ROOT / args.out, v_start, v_len, args.fps, args.width,
                     crop=args.crop, colors=args.colors)
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    if args.also_mp4:
        mp4 = encode_mp4(video, ROOT / args.also_mp4, v_start, v_len, args.width * 2,
                         crop=args.crop)
        print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    if not (args.capture or args.from_capture):
        shutil.rmtree(video.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
