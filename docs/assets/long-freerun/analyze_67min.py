"""
Deep analysis of the 67-min all-gold free-run torture test vs the short gold session.
  LONG : 20260701-133403-playerG.npz  (all-gold + Ten20, 67.6 min, free-run, soap-washed)
  SHORT: 20260630-175142-playerG.npz  (all-gold, 3.25 min, cued)
Reads npz read-only; all ops on copies. Outputs metrics JSON + PNGs to scratchpad.
"""
import json, numpy as np
from scipy import signal as sps
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_trapz=np.trapezoid if hasattr(np,"trapezoid") else np.trapz
FS=250; RAIL_MV=187.5
OUT="/private/tmp/claude-501/-Users-john-Dev-brain-pong/a296d6bb-7bc0-4ef9-9395-839225665241/scratchpad"
LONG="data/eog/20260701-133403-playerG.npz"; SHORT="data/eog/20260630-175142-playerG.npz"

def load(f):
    d=np.load(f,allow_pickle=True); e=d['eeg']
    return (np.ascontiguousarray(e[1].astype(float)), np.ascontiguousarray(e[2].astype(float)),
            e[10].astype(float), e.shape[1])
c1,c2,ts,N = load(LONG)              # volts
heog=(c2-c1)                         # volts
t=np.arange(N)/FS
M={}

# ---- 0. timing gaps ----
dt=np.diff(ts); gap_idx=np.where(dt>2/FS)[0]
M['gaps']=[{"at_s":float(ts[i]-ts[0]),"at_min":float((ts[i]-ts[0])/60),
            "gap_s":float(dt[i]),"missing_samples":int(round(dt[i]*FS))-1} for i in gap_idx]
M['duration_min']=N/FS/60; M['n_samples']=N

# ---- helpers ----
def bp(x,lo,hi,o=4):
    b,a=sps.butter(o,[lo/(FS/2),hi/(FS/2)],btype='band'); return sps.filtfilt(b,a,x)
def lp(x,hi,o=4):
    b,a=sps.butter(o,hi/(FS/2),btype='low'); return sps.filtfilt(b,a,x)
def sliding(x,winsec,stepsec,fn):
    w=int(winsec*FS); s=int(stepsec*FS); out=[]; tc=[]
    for a in range(0,len(x)-w,s):
        out.append(fn(x[a:a+w])); tc.append((a+w/2)/FS)
    return np.array(tc),np.array(out)
def amp60(x):  # 60Hz RMS uV in a window (x in volts)
    xu=(x-x.mean())*1e6
    f,p=sps.welch(xu,FS,nperseg=min(len(xu),512))
    m=(f>=59)&(f<=61); return float(np.sqrt(_trapz(p[m],f[m]))) if m.any() else 0.0

# ---- 1. DC / rail / headroom ----
M['dc']={}
for nm,x in [("CH1",c1),("CH2",c2),("HEOG",heog)]:
    xmv=x*1e3; am=float(np.abs(xmv).max())
    M['dc'][nm]=dict(mean_mV=float(xmv.mean()),min_mV=float(xmv.min()),max_mV=float(xmv.max()),
        abs_max_mV=am, rail_use_pct=100*am/RAIL_MV, min_headroom_mV=RAIL_MV-am,
        frac_gt80pct_rail=float(np.mean(np.abs(xmv)>0.8*RAIL_MV)))

# ---- 2. drift: settle fit + windowed slope ----
def settle(x):  # x volts -> mV; returns dict
    xmv=x*1e3; sl,ic=np.polyfit(t,xmv,1)
    tc,ws=sliding(x,60,30,lambda s:np.polyfit(np.arange(len(s))/FS,s*1e3,1)[0]*60)  # mV/min per 60s win
    try:
        p0=[xmv[0]-xmv[-1],400,xmv[-1]]
        po,_=curve_fit(lambda tt,A,tau,C:A*np.exp(-tt/tau)+C,t,xmv,p0=p0,maxfev=40000)
        tau=float(po[1]); C=float(po[2]); r2=1-np.var(xmv-(po[0]*np.exp(-t/po[1])+po[2]))/np.var(xmv)
    except Exception: tau=C=r2=float('nan')
    slow=lp(xmv,0.02)
    return dict(slope_mV_min=float(sl*60), total_change_mV=float(xmv[-1*FS:].mean()-xmv[:FS].mean()),
        tau_s=tau, asymptote_mV=C, exp_r2=float(r2),
        early_slope=float(ws[:4].mean()), late_slope=float(ws[-4:].mean()),
        slow_p2p_mV=float(slow.max()-slow.min()), win_slopes=ws.tolist(), win_t=tc.tolist())
M['drift']={nm:settle(x) for nm,x in [("CH1",c1),("CH2",c2),("HEOG",heog)]}

# ---- 3. 60Hz + s2s over time (startup settle, gel-dry check) ----
t60,a60_c1=sliding(c1,10,10,amp60)
_,a60_c2=sliding(c2,10,10,amp60)
_,a60_h =sliding(heog,10,10,amp60)
ts2,s2s_h=sliding(heog,10,10,lambda s:np.std(np.diff(s*1e6))/np.sqrt(2))
_,s2s_c1=sliding(c1,10,10,lambda s:np.std(np.diff(s*1e6))/np.sqrt(2))
M['t60_min']=(t60/60).tolist(); M['a60_c1']=a60_c1.tolist(); M['a60_c2']=a60_c2.tolist(); M['a60_h']=a60_h.tolist()
M['s2s_h']=s2s_h.tolist(); M['s2s_c1']=s2s_c1.tolist()
def seg(a,lo,hi):
    m=(t60/60>=lo)&(t60/60<hi); return float(np.median(a[m])) if m.any() else float('nan')
M['noise_epochs']={
  'first2min_60Hz_c1':seg(a60_c1,0,2),'mid_60Hz_c1':seg(a60_c1,20,40),'last5min_60Hz_c1':seg(a60_c1,62,67),
  'first2min_s2s_h':seg(s2s_h,0,2),'mid_s2s_h':seg(s2s_h,20,40),'last5min_s2s_h':seg(s2s_h,62,67)}

# ---- 4. blinks/artifacts over time ----
hf=bp(heog,1,10)*1e6            # uV, blink band
absh=np.abs(hf)
thr=6*np.median(absh)/0.6745*0.0+40   # fixed 40uV prominence (gold scale)
pk,props=sps.find_peaks(absh,prominence=40,distance=int(0.15*FS))
blink_t=t[pk]; blink_amp=absh[pk]
tb,brate=sliding(np.zeros(N),60,60,lambda s:0)  # placeholder times
# blink rate per minute
edges=np.arange(0,N/FS+60,60); cnts,_=np.histogram(blink_t,bins=edges)
M['blinks']=dict(n=int(pk.size), rate_per_min=float(pk.size/(N/FS)*60),
    med_amp_uV=float(np.median(blink_amp)), rate_by_min=cnts.tolist())
# big transients (swallow/move): low-freq large excursions in R-L beyond blink band
lfe=bp(heog,0.3,4)*1e6
big,_=sps.find_peaks(np.abs(lfe),prominence=150,distance=int(0.5*FS))
M['big_transients']=dict(n=int(big.size), times_min=[float(t[i]/60) for i in big[:40]])

# ---- 5. compare to SHORT gold ----
s1,s2,sts,SN=load(SHORT); sh=(s2-s1); st=np.arange(SN)/FS
def core(c1_,c2_,h_,tt):
    return dict(
      dc_ch1=float(c1_.mean()*1e3),dc_ch2=float(c2_.mean()*1e3),
      s2s_ch1=float(np.std(np.diff(c1_*1e6))/np.sqrt(2)),s2s_ch2=float(np.std(np.diff(c2_*1e6))/np.sqrt(2)),
      s2s_heog=float(np.std(np.diff(h_*1e6))/np.sqrt(2)),
      a60_ch1=amp60(c1_ if len(c1_)<300000 else c1_[:150000]),
      a60_heog=amp60(h_ if len(h_)<300000 else h_[60000:210000]),
      slope_ch2_mV_min=float(np.polyfit(tt,c2_*1e3,1)[0]*60),
      headroom_ch2_mV=RAIL_MV-float(np.abs(c2_*1e3).max()))
M['compare']={'LONG_67min':core(c1,c2,heog,t),'SHORT_3min':core(s1,s2,sh,st)}

# ================= PLOTS =================
dec=25  # decimate for plotting
td=t[::dec]/60
# Fig1: per-channel + R-L full hour, rail context
fig,ax=plt.subplots(3,1,figsize=(14,8),sharex=True)
ax[0].plot(td,c1[::dec]*1e3,lw=.4,color="#4499ff"); ax[0].set_ylabel("CH1 L (mV)"); ax[0].axhline(-RAIL_MV,color='r',ls='--',lw=.7)
ax[1].plot(td,c2[::dec]*1e3,lw=.4,color="#ff8844"); ax[1].set_ylabel("CH2 R (mV)"); ax[1].axhline(-RAIL_MV,color='r',ls='--',lw=.7)
ax[2].plot(td,heog[::dec]*1e6,lw=.3,color="#33aa55"); ax[2].set_ylabel("R-L (uV)"); ax[2].set_xlabel("min")
for a in ax[:2]: a.set_ylim(-RAIL_MV*1.05,10)
ax[0].set_title(f"67-min all-gold free run — per-channel sit at ~-35/-38mV, rail at -187.5mV (dashed)")
for g in M['gaps']:
    for a in ax: a.axvline(g['at_min'],color='m',ls=':',lw=.8)
plt.tight_layout(); plt.savefig(f"{OUT}/long_overview.png",dpi=110); plt.close()

# Fig2: drift settle — full trend + windowed slope + first-5min zoom
fig,ax=plt.subplots(2,2,figsize=(14,7))
for nm,x,col in [("CH1",c1,"#4499ff"),("CH2",c2,"#ff8844")]:
    ax[0,0].plot(t[::dec]/60, lp(x*1e3,0.02)[::dec], color=col,label=nm)
ax[0,0].set_title("Per-channel slow trend (<0.02Hz) over the hour"); ax[0,0].set_xlabel("min"); ax[0,0].set_ylabel("mV"); ax[0,0].legend()
d=M['drift']['CH2']; ax[0,1].plot(np.array(d['win_t'])/60,d['win_slopes'],color="#ff8844")
ax[0,1].axhline(0,color='k',lw=.5); ax[0,1].set_title("CH2 drift RATE (60s windows) — decel = settling"); ax[0,1].set_xlabel("min"); ax[0,1].set_ylabel("mV/min")
z=int(300*FS); ax[1,0].plot(t[:z:5]/60,c1[:z:5]*1e3,color="#4499ff",lw=.5,label="CH1"); ax[1,0].plot(t[:z:5]/60,c2[:z:5]*1e3,color="#ff8844",lw=.5,label="CH2")
ax[1,0].set_title("First 5 min (startup settle)"); ax[1,0].set_xlabel("min"); ax[1,0].set_ylabel("mV"); ax[1,0].legend()
ax[1,1].plot(np.array(M['t60_min']),M['a60_c1'],color="#4499ff",label="60Hz CH1"); ax[1,1].plot(np.array(M['t60_min']),M['s2s_h'],color="#33aa55",label="s2s R-L")
ax[1,1].set_title("60Hz & s2s noise over time (startup + gel-dry check)"); ax[1,1].set_xlabel("min"); ax[1,1].set_ylabel("uV"); ax[1,1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/long_drift.png",dpi=110); plt.close()

# Fig3: blink rate/amp over time + spectrogram
fig,ax=plt.subplots(3,1,figsize=(14,8))
ax[0].bar(np.arange(len(M['blinks']['rate_by_min'])),M['blinks']['rate_by_min'],color="#8888ff",width=1.0)
ax[0].set_title(f"Blink rate per minute (n={M['blinks']['n']}, {M['blinks']['rate_per_min']:.1f}/min avg)"); ax[0].set_ylabel("blinks/min")
ax[1].scatter(blink_t/60,blink_amp,s=3,color="#5555aa",alpha=.4); ax[1].set_title("Blink amplitude over time (uV)"); ax[1].set_ylabel("uV"); ax[1].set_ylim(0,np.percentile(blink_amp,99)*1.2)
f,tt,Sxx=sps.spectrogram((heog-heog.mean())*1e6,FS,nperseg=2048,noverlap=1024)
ax[2].pcolormesh(tt/60,f,10*np.log10(Sxx+1e-6),shading='gouraud',cmap='magma',vmin=-20,vmax=30)
ax[2].set_ylim(0,70); ax[2].set_title("R-L spectrogram (dB) — look for drift/sweat bands, 60Hz stability"); ax[2].set_xlabel("min"); ax[2].set_ylabel("Hz")
plt.tight_layout(); plt.savefig(f"{OUT}/long_blinks_spec.png",dpi=110); plt.close()

def strip(o):
    if isinstance(o,dict): return {k:strip(v) for k,v in o.items() if k not in('win_slopes','win_t')}
    return o
json.dump(strip(M),open(f"{OUT}/long_metrics.json","w"),indent=2,default=float)
print("GAPS:",M['gaps'])
print("\nDC/RAIL:"); [print(" ",k,{kk:round(vv,2) for kk,vv in v.items()}) for k,v in M['dc'].items()]
print("\nDRIFT:");
for k,v in M['drift'].items(): print(f"  {k}: slope={v['slope_mV_min']:+.3f}mV/min net={v['total_change_mV']:+.2f}mV tau={v['tau_s']:.0f}s asym={v['asymptote_mV']:.1f}mV early_slp={v['early_slope']:+.2f} late_slp={v['late_slope']:+.2f} slow_p2p={v['slow_p2p_mV']:.2f}mV")
print("\nNOISE epochs:",{k:round(v,2) for k,v in M['noise_epochs'].items()})
print("BLINKS:",{k:(round(v,2) if isinstance(v,float) else v) for k,v in M['blinks'].items() if k!='rate_by_min'})
print("BIG transients:",M['big_transients']['n'])
print("\nCOMPARE:")
for k,v in M['compare'].items(): print(f"  {k}: "+", ".join(f"{kk}={vv:.2f}" for kk,vv in v.items()))
print("\nplots: long_overview.png long_drift.png long_blinks_spec.png")
