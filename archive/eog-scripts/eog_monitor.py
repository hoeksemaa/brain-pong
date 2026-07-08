"""
EOG live diagnostic monitor — DC-coupled raw viewer for electrode/contact debugging.

The record_eog.py display HIGH-PASS filters at 0.5 Hz, which flattens the near-DC
EOG gaze plateau and hides contact problems. THIS viewer does NOT high-pass — it
shows the raw signal in real units so you can actually see DC level, rail proximity,
floating inputs, and the slow gaze steps. It is VIEW-ONLY (records nothing).

What to look for while it runs:
  • TAP an electrode  → a big spike on THAT channel only  (proves it's live + wired)
  • BLINK hard        → both canthi jump together (common-mode)  (electrodes alive)
  • LOOK left/right, HOLD ~3s → channels should move ANTI-PHASE; the diff steps
                        to a new level and HOLDS (that's EOG; a HPF would hide it)
  • A channel pinned near ±rail with the two channels IDENTICAL → that input is
                        FLOATING (electrode off / open lead)  → the verdict turns red

Failure-mode verdicts (per channel, live):
  OK            mid-range DC, real noise
  HIGH/float?   |DC| > 70% of rail — likely floating / poor contact
  RAILING       |DC| > 92% of rail — saturated
  DEAD/flat     ~zero noise — frozen / no live data

Keys:  r = toggle centered ↔ absolute(±rail)   n = toggle 60/50 Hz notch   q = quit

Usage:
    source .venv/bin/activate
    python scripts/eog_monitor.py
    python scripts/eog_monitor.py --port /dev/cu.usbserial-1120 --seconds 10
    python scripts/eog_monitor.py --notch            # start with mains notch on
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

# ── Board / channel config (mirrors record_eog.py) ──────────────────────────
SERIAL_PORT = "/dev/cu.usbserial-1120"
BOARD_ID    = BoardIds.CERELOG_X8_BOARD
EOG_SLOT_L  = 0     # left  canthus → get_eeg_channels()[0]  (CH1)
EOG_SLOT_R  = 1     # right canthus → get_eeg_channels()[1]  (CH2)
V_REF       = 4.5   # ADS1299 reference; full-scale ≈ ±V_REF/gain referred to input

# colours (match record_eog aesthetic)
C_L, C_R, C_D = "#4499ff", "#ff8844", "#aaffaa"
C_OK, C_WARN, C_BAD = "#88cc44", "#ffcc44", "#ff5555"
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))


def verdict(dc_uv, std_uv, ceil_uv):
    """Per-channel contact verdict from DC level + noise."""
    pct = abs(dc_uv) / ceil_uv
    if std_uv < 1.0:    return "DEAD/flat", C_BAD       # real ADC always has noise
    if pct > 0.92:      return "RAILING",   C_BAD
    if pct > 0.70:      return "HIGH/float?", C_WARN
    return "OK", C_OK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port',    default=SERIAL_PORT)
    ap.add_argument('--seconds', type=float, default=8.0, help='display window (s)')
    ap.add_argument('--gain',    type=int, default=24, help='PGA gain → sets rail ceiling')
    ap.add_argument('--chL',     type=int, default=EOG_SLOT_L, help='L eeg-channel slot')
    ap.add_argument('--chR',     type=int, default=EOG_SLOT_R, help='R eeg-channel slot')
    ap.add_argument('--notch',   action='store_true', help='start with 50/60 Hz notch on')
    args = ap.parse_args()

    ceil_uv = V_REF / args.gain * 1e6   # ±full-scale referred to input, in µV

    params = BrainFlowInputParams()
    params.serial_port = args.port
    params.timeout = 15
    BoardShim.disable_board_logger()
    sr  = BoardShim.get_sampling_rate(BOARD_ID)
    eeg = BoardShim.get_eeg_channels(BOARD_ID)
    chL, chR = eeg[args.chL], eeg[args.chR]
    win = int(args.seconds * sr)

    state = {'centered': True, 'notch': args.notch}

    print(f"port={args.port}  fs={sr}Hz  L=row{chL}  R=row{chR}  rail=±{ceil_uv:.0f}µV (x{args.gain})")
    print("connecting (board boots ~8s)…")
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream(int(5 * 60 * sr))
    time.sleep(2)
    print("streaming. tap/blink/look-and-hold to test.  keys: r=centered/abs  n=notch  q=quit\n")

    disp = np.full((2, win), np.nan)   # rolling [L, R] window

    # ── figure ──────────────────────────────────────────────────────────────
    fig, (ax_ch, ax_d) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [3, 2]})
    fig.patch.set_facecolor('#111111')
    for ax in (ax_ch, ax_d):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#666666')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333333')
        ax.grid(True, color='#222222')

    t = np.linspace(-args.seconds, 0, win)
    (ln_L,) = ax_ch.plot(t, np.full(win, np.nan), lw=1.0, color=C_L, label='L canthus')
    (ln_R,) = ax_ch.plot(t, np.full(win, np.nan), lw=1.0, color=C_R, label='R canthus')
    (ln_D,) = ax_d.plot(t,  np.full(win, np.nan), lw=1.2, color=C_D)
    ax_ch.legend(loc='upper left', facecolor='#222222', edgecolor='#333333',
                 labelcolor='white', fontsize=9)
    ax_ch.set_ylabel('µV', color='#888888')
    ax_d.set_ylabel('R − L  (µV)', color='#888888')
    ax_d.set_xlabel('time (s)', color='#888888')
    ax_ch.set_xlim(-args.seconds, 0); ax_d.set_xlim(-args.seconds, 0)

    txt_L = ax_ch.text(0.99, 0.95, '', transform=ax_ch.transAxes, ha='right', va='top',
                       fontsize=10, family='monospace', color=C_OK)
    txt_R = ax_ch.text(0.99, 0.86, '', transform=ax_ch.transAxes, ha='right', va='top',
                       fontsize=10, family='monospace', color=C_OK)
    txt_D = ax_d.text(0.99, 0.95, '', transform=ax_d.transAxes, ha='right', va='top',
                      fontsize=10, family='monospace', color='white')
    sup = fig.suptitle('', color='white', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    def _notch(x):
        x = np.ascontiguousarray(x.astype(np.float64))
        for lo, hi in NOTCH_BANDS:
            DataFilter.perform_bandstop(x, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
        return x

    def on_key(ev):
        if ev.key == 'r': state['centered'] = not state['centered']
        elif ev.key == 'n': state['notch'] = not state['notch']
        elif ev.key == 'q': plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_):
        new = board.get_board_data()
        if new.shape[1] > 0:
            chunk = np.vstack((new[chL], new[chR])) * 1e6  # µV, RAW (no filter)
            n = chunk.shape[1]
            if n >= win:
                disp[:] = chunk[:, -win:]
            else:
                disp[:, :-n] = disp[:, n:]
                disp[:, -n:] = chunk

        L, R = disp[0], disp[1]
        valid = ~np.isnan(L)
        if valid.sum() < 5:
            return
        Lv, Rv = L[valid], R[valid]

        # stats on RAW (DC included) — this is the diagnostic truth
        for tx, v, name in ((txt_L, Lv, 'L'), (txt_R, Rv, 'R')):
            dc, sd = float(v.mean()), float(v.std())
            vd, col = verdict(dc, sd, ceil_uv)
            tx.set_text(f"{name} DC={dc/1000:+.1f}mV ({abs(dc)/ceil_uv*100:.0f}%rail) σ={sd:.0f}µV  {vd}")
            tx.set_color(col)
        Dv = Rv - Lv
        corr = float(np.corrcoef(Lv, Rv)[0, 1]) if Lv.std() > 0 and Rv.std() > 0 else 0.0
        txt_D.set_text(f"diff σ={Dv.std():.0f}µV   corr(L,R)={corr:+.2f}")
        hint = ""
        if corr > 0.9 and Dv.std() < 30:
            hint = "  L≡R → both floating? check contact"
        sup.set_text(f"EOG live monitor — {'CENTERED' if state['centered'] else 'ABSOLUTE ±rail'}"
                     f"{' +notch' if state['notch'] else ' (raw)'}{hint}")

        # plotting series (display transform only; stats above are untouched RAW)
        pL, pR = L.copy(), R.copy()
        if state['notch']:
            pL[valid], pR[valid] = _notch(Lv), _notch(Rv)
        pD = pR - pL
        if state['centered']:
            pL = pL - np.nanmean(pL); pR = pR - np.nanmean(pR); pD = pD - np.nanmean(pD)
        ln_L.set_ydata(pL); ln_R.set_ydata(pR); ln_D.set_ydata(pD)

        if state['centered']:
            pk = np.nanmax(np.abs(np.concatenate([pL[valid], pR[valid]])))
            pk = max(pk * 1.2, 15.0)
            ax_ch.set_ylim(-pk, pk)
            dpk = max(np.nanmax(np.abs(pD[valid])) * 1.2, 15.0)
            ax_d.set_ylim(-dpk, dpk)
        else:
            ax_ch.set_ylim(-ceil_uv, ceil_uv)
            ax_d.set_ylim(-ceil_uv, ceil_uv)

    ani = FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            board.stop_stream(); board.release_session()
        except Exception:
            pass
        print("monitor closed.")


if __name__ == '__main__':
    main()
