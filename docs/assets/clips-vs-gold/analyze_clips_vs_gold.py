"""
Deep numerical comparison: carbon-rubber EAR CLIPS vs GOLD ear-cup electrodes
as the bias-drive + SRB1 reference pair, on Player G's 2-electrode HEOG rig.

Both recordings: same gold-cup canthi (eye) signal electrodes, same board
(CERELOG_X8 unit:original), gain x24, fs=250, same record_eog.py paradigm
(25 LEFT + 25 RIGHT cued gaze trials, 2 s hold, 1.5 s rest).

  CLIPS : 20260630-173432-playerG.npz   (carbon/rubber clip ears + Signa gel)
  GOLD  : 20260630-175142-playerG.npz   (gold cups + Ten20, taped)

Channel model: row1 = CH1 = LEFT canthus referenced to SRB1 (ear),
               row2 = CH2 = RIGHT canthus referenced to SRB1 (ear).
HEOG = CH2 - CH1 (R - L); common-mode (reference noise/mains) cancels here,
so per-channel metrics expose reference/bias contact quality most directly.

Outputs: JSON of all metrics + diagnostic PNGs to the scratchpad dir.
Reads .npz read-only; all DataFilter-style ops on copies (never mutate source).
"""
import json
import numpy as np
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

FS = 250
GAIN = 24
VREF = 4.5
RAIL_V = VREF / GAIN            # +/- referred-to-input ceiling in volts
RAIL_MV = RAIL_V * 1e3         # +/- 187.5 mV
OUT = "/private/tmp/claude-501/-Users-john-Dev-brain-pong/a296d6bb-7bc0-4ef9-9395-839225665241/scratchpad"

SESS = {
    "CLIPS": "data/eog/20260630-173432-playerG.npz",
    "GOLD":  "data/eog/20260630-175142-playerG.npz",
}

def load(path):
    d = np.load(path, allow_pickle=True)
    eeg = d["eeg"]
    return {
        "ch1": np.ascontiguousarray(eeg[1].astype(np.float64)),  # volts, LEFT canthus / SRB1
        "ch2": np.ascontiguousarray(eeg[2].astype(np.float64)),  # volts, RIGHT canthus / SRB1
        "ev_s": d["event_samples"].astype(int),
        "ev_l": [str(x) for x in d["event_labels"]],
        "n": eeg.shape[1],
    }

def butter_filt(x, lo, hi, fs=FS, order=4):
    nyq = fs / 2
    if lo and hi:
        b, a = sps.butter(order, [lo/nyq, hi/nyq], btype="band")
    elif hi:
        b, a = sps.butter(order, hi/nyq, btype="low")
    else:
        b, a = sps.butter(order, lo/nyq, btype="high")
    return sps.filtfilt(b, a, x)

def band_rms(x, lo, hi):
    return float(np.std(butter_filt(x, lo, hi)))

def welch_psd(x, fs=FS, nper=2048):
    f, p = sps.welch(x, fs=fs, nperseg=nper, noverlap=nper//2, detrend="constant")
    return f, p

def band_power_rms(f, p, lo, hi):
    """RMS amplitude (same units as x) integrated over [lo,hi]."""
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return 0.0
    df = f[1] - f[0]
    return float(np.sqrt(_trapz(p[m], dx=df)))

def peak_at(f, p, f0, halfwidth=1.0):
    m = (f >= f0-halfwidth) & (f <= f0+halfwidth)
    return float(np.sqrt(_trapz(p[m], dx=f[1]-f[0]))) if m.any() else 0.0

def rest_windows(ev_s, ev_l, n):
    """Return list of (start,stop) sample idx for REST holds (use middle 1.0s)."""
    out = []
    for s, l in zip(ev_s, ev_l):
        if l == "REST":
            a = s + int(0.25*FS); b = s + int(1.25*FS)
            if b <= n:
                out.append((a, b))
    return out

def trial_epochs(ev_s, ev_l, n):
    """For each LEFT/RIGHT cue: preceding-rest baseline window + response window."""
    rows = []
    for i, (s, l) in enumerate(zip(ev_s, ev_l)):
        if l in ("LEFT", "RIGHT"):
            base_a, base_b = s - int(1.25*FS), s - int(0.25*FS)   # the rest before
            resp_a, resp_b = s + int(0.30*FS), s + int(1.90*FS)   # skip saccade onset/settle
            if base_a >= 0 and resp_b <= n:
                rows.append((l, base_a, base_b, resp_a, resp_b))
    return rows

def analyze(name, S):
    ch1, ch2, n = S["ch1"], S["ch2"], S["n"]
    heog = ch2 - ch1
    res = {"name": name, "n_samples": n, "duration_s": n/FS}

    # ---- A. DC offset & rail safety (per channel; HEOG offset too) ----
    res["dc"] = {}
    for nm, x in [("CH1", ch1), ("CH2", ch2), ("HEOG", heog)]:
        xmv = x * 1e3
        absmax = float(np.abs(xmv).max())
        res["dc"][nm] = {
            "mean_mV": float(xmv.mean()),
            "min_mV": float(xmv.min()),
            "max_mV": float(xmv.max()),
            "abs_max_mV": absmax,
            "headroom_min_mV": RAIL_MV - absmax,          # worst-case instantaneous headroom
            "headroom_dc_mV": RAIL_MV - abs(float(xmv.mean())),
            "rail_use_pct": 100*absmax/RAIL_MV,            # how close to the ceiling, %
            "frac_within_5pct_rail": float(np.mean(np.abs(xmv) > 0.95*RAIL_MV)),
        }

    # ---- B. Drift over time (low-frequency wander) ----
    t = np.arange(n)/FS
    res["drift"] = {}
    for nm, x in [("CH1", ch1), ("CH2", ch2), ("HEOG", heog)]:
        xmv = x*1e3
        # linear slope
        sl, intc = np.polyfit(t, xmv, 1)
        # very-low-freq wander (<0.1 Hz) peak-to-peak
        slow = butter_filt(xmv, None, 0.1)
        # within-rest baseline wander: std of per-rest mean levels
        res["drift"][nm] = {
            "slope_mV_per_min": float(sl*60),
            "total_linear_change_mV": float(sl*t[-1]),
            "slow_p2p_mV_<0.1Hz": float(slow.max()-slow.min()),
            "slow_std_mV_<0.1Hz": float(np.std(slow)),
        }

    # ---- C. Broadband noise floor (drift-immune + banded), in microvolts ----
    res["noise"] = {}
    rw = rest_windows(S["ev_s"], S["ev_l"], n)
    res["n_rest_windows"] = len(rw)
    for nm, x in [("CH1", ch1), ("CH2", ch2), ("HEOG", heog)]:
        xuv = x*1e6
        # sample-to-sample (high-freq) noise, drift-immune: std(diff)/sqrt(2)
        s2s = float(np.std(np.diff(xuv))/np.sqrt(2))
        # rest-only band RMS (concatenate detrended rest segments)
        segs = []
        for a, b in rw:
            seg = xuv[a:b].copy()
            seg = seg - seg.mean()
            segs.append(seg)
        restcat = np.concatenate(segs) if segs else xuv*0
        res["noise"][nm] = {
            "sample_to_sample_uV_rms": s2s,
            "rest_total_std_uV": float(np.std(restcat)),
            "rest_band_1_40Hz_uV_rms": band_rms(restcat, 1, 40) if restcat.size>200 else None,
            "rest_band_15_45Hz_uV_rms": band_rms(restcat, 15, 45) if restcat.size>200 else None,
            "rest_band_above70Hz_uV_rms": band_rms(restcat, 70, 124) if restcat.size>200 else None,
        }

    # ---- D. Mains / spectral (PSD on uniform n/fs axis), microvolts ----
    res["mains"] = {}
    res["_psd"] = {}
    for nm, x in [("CH1", ch1), ("CH2", ch2), ("HEOG", heog)]:
        xuv = x*1e6
        f, p = welch_psd(xuv)
        res["_psd"][nm] = (f, p)   # for plotting; stripped before JSON
        res["mains"][nm] = {
            "rms_60Hz_pm1_uV": peak_at(f, p, 60, 1.0),
            "rms_120Hz_pm1_uV": peak_at(f, p, 120, 1.0),
            "rms_50Hz_pm1_uV": peak_at(f, p, 50, 1.0),
            "rms_total_1_124Hz_uV": band_power_rms(f, p, 1, 124),
        }

    # ---- E. Task performance: LEFT vs RIGHT separability on HEOG ----
    heog_uv = heog*1e6
    ep = trial_epochs(S["ev_s"], S["ev_l"], n)
    defl = {"LEFT": [], "RIGHT": []}
    base_noise = []
    for l, ba, bb, ra, rb in ep:
        base = heog_uv[ba:bb].mean()
        resp = heog_uv[ra:rb].mean()
        defl[l].append(resp - base)
        base_noise.append(np.std(heog_uv[ba:bb] - heog_uv[ba:bb].mean()))
    L = np.array(defl["LEFT"]); R = np.array(defl["RIGHT"])
    # pooled within-class std
    pooled = np.sqrt(((L.var(ddof=1)*(len(L)-1)) + (R.var(ddof=1)*(len(R)-1))) /
                     (len(L)+len(R)-2))
    cohen_d = float(abs(R.mean()-L.mean())/pooled)
    # 1-D classification: optimal threshold accuracy + AUC
    allv = np.concatenate([L, R]); lab = np.array([0]*len(L)+[1]*len(R))
    # AUC (rank-based)
    order = np.argsort(allv)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(allv)+1)
    n1 = lab.sum(); n0 = len(lab)-n1
    auc = float((ranks[lab==1].sum() - n1*(n1+1)/2)/(n0*n1))
    auc = max(auc, 1-auc)
    # best threshold accuracy
    thr_cand = np.sort(allv)
    best_acc = 0.0
    for thr in thr_cand:
        for sign in (+1, -1):
            pred = ((allv > thr).astype(int) if sign>0 else (allv <= thr).astype(int))
            acc = max((pred==lab).mean(), (pred!=lab).mean())
            best_acc = max(best_acc, acc)
    res["task"] = {
        "n_LEFT": len(L), "n_RIGHT": len(R),
        "LEFT_mean_uV": float(L.mean()), "LEFT_std_uV": float(L.std(ddof=1)),
        "RIGHT_mean_uV": float(R.mean()), "RIGHT_std_uV": float(R.std(ddof=1)),
        "separation_uV": float(abs(R.mean()-L.mean())),
        "pooled_within_std_uV": float(pooled),
        "cohen_d": cohen_d,
        "auc": auc,
        "best_threshold_acc": float(best_acc),
        "mean_trial_baseline_noise_uV": float(np.mean(base_noise)),
        "snr_sep_over_noise": float(abs(R.mean()-L.mean())/np.mean(base_noise)),
    }
    res["_defl"] = (L, R)

    # ---- G. Stationarity across thirds (rest s2s noise per third) ----
    res["thirds_s2s_uV"] = {}
    for nm, x in [("CH1", ch1), ("CH2", ch2), ("HEOG", heog)]:
        xuv = x*1e6
        th = []
        for k in range(3):
            seg = xuv[k*n//3:(k+1)*n//3]
            th.append(float(np.std(np.diff(seg))/np.sqrt(2)))
        res["thirds_s2s_uV"][nm] = th
    return res

results = {}
for name, path in SESS.items():
    results[name] = analyze(name, load(path))

# ---------- PLOTS ----------
# 1) PSD overlay (CH2 and HEOG)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
colors = {"CLIPS": "#d9534f", "GOLD": "#d4af37"}
for ax, chan in zip(axes, ["CH2", "HEOG"]):
    for name in SESS:
        f, p = results[name]["_psd"][chan]
        ax.semilogy(f, p, label=name, color=colors[name], lw=1.1)
    ax.axvline(60, color="#888", ls=":", lw=0.8)
    ax.set_title(f"PSD — {chan}"); ax.set_xlabel("Hz"); ax.set_ylabel("uV^2/Hz")
    ax.set_xlim(0, 125); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/psd_compare.png", dpi=110); plt.close()

# 2) Raw HEOG full timeseries (mV) with rail lines + DC level
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
for ax, name in zip(axes, ["CLIPS", "GOLD"]):
    S = load(SESS[name]); heog = (S["ch2"]-S["ch1"])
    ch2mv = S["ch2"]*1e3
    t = np.arange(S["n"])/FS
    ax.plot(t, ch2mv, color=colors[name], lw=0.5, label=f"{name} CH2 (R/ref)")
    ax.axhline(-RAIL_MV, color="k", ls="--", lw=0.8); ax.axhline(RAIL_MV, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_ylabel("mV (ref to input)"); ax.legend(loc="upper right")
    ax.set_title(f"{name}: CH2 raw — DC offset {ch2mv.mean():.1f} mV  (rail +/-{RAIL_MV:.0f} mV)")
    ax.set_ylim(-RAIL_MV*1.05, RAIL_MV*1.05)
axes[-1].set_xlabel("s")
plt.tight_layout(); plt.savefig(f"{OUT}/raw_rail.png", dpi=110); plt.close()

# 3) LEFT/RIGHT deflection distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, name in zip(axes, ["CLIPS", "GOLD"]):
    L, R = results[name]["_defl"]
    ax.hist(L, bins=12, alpha=0.6, color="#4499ff", label="LEFT")
    ax.hist(R, bins=12, alpha=0.6, color="#ff8844", label="RIGHT")
    t = results[name]["task"]
    ax.set_title(f"{name}: HEOG deflection  d={t['cohen_d']:.2f}  acc={t['best_threshold_acc']*100:.0f}%")
    ax.set_xlabel("HEOG deflection (uV)"); ax.legend()
axes[0].set_ylabel("trials")
plt.tight_layout(); plt.savefig(f"{OUT}/deflection_hist.png", dpi=110); plt.close()

# strip non-serializable, dump JSON
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    return o
with open(f"{OUT}/metrics.json", "w") as fh:
    json.dump(strip(results), fh, indent=2)
print(json.dumps(strip(results), indent=2))
print("\nPlots:", "psd_compare.png raw_rail.png deflection_hist.png")
