/* EOG diagnostic viewer — talks to the Flask API (scripts/serve_viewer.py).
 * Raw signal is read from the frozen npz server-side and min/max-decimated; this
 * client renders the decimated arrays and manages the trim keep-window, which is
 * persisted as a DB annotation (the npz are never modified). */
(function () {
  const THEME = {
    panel:"#10141d", grid:"#212838", zero:"#313a4a",
    text:"#e6e9f0", muted:"#8b93a7", mono:"ui-monospace,Menlo,monospace", ui:"Inter,system-ui,sans-serif",
    palette:["#4ea8ff","#ff9d5c","#5ad19a","#b388ff","#ff6f91","#38bdf8","#fbbf24","#2dd4bf"],
    rib:"#7ef0b0", ribFill:"rgba(126,240,176,0.13)",
    rail:"#ff6b81", railFill:"rgba(255,107,129,0.08)", evL:"#5b8cff", evR:"#ff9d5c",
    trimDim:"rgba(8,10,16,0.70)", trimEdge:"#7ef0b0",
    status:{ok:"#34d399", railing:"#fb7185", flat:"#fbbf24"}, chFillA:0.13,
    side:"#b388ff",   // uniform, status-neutral sidebar accent (dot + spark)
  };
  const FILTERS = [["raw","Raw","neutral"],["bp_0530","0.5–30 Hz","accent"],
                   ["bp_0130","0.1–30 Hz","accent2"],["velocity","Velocity","accent3"]];
  const FILTER_LABEL = Object.fromEntries(FILTERS.map(([id,l])=>[id,l]));
  const WIDTH = 1600;
  const PADL = 48, PADR = 8;

  const $ = (s, r=document) => r.querySelector(s);
  function el(tag, cls, html){ const e=document.createElement(tag); if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e; }
  function fmtYMD(id){ const m=id.match(/^(\d{4})(\d{2})(\d{2})/); return m?`${m[1]}-${m[2]}-${m[3]}`:id; }
  const lerp=(a,b,t)=>a+(b-a)*t;
  function median(a){const b=a.slice().sort((x,y)=>x-y);const n=b.length;if(!n)return 0;const m=n>>1;return n%2?b[m]:(b[m-1]+b[m])/2;}
  function pctl(a,p){const b=a.slice().sort((x,y)=>x-y);if(!b.length)return 0;return b[Math.min(b.length-1,Math.max(0,Math.floor(p/100*b.length)))];}
  function hexA(hex,a){const n=parseInt(hex.slice(1),16);return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;}

  // ── structured logger ───────────────────────────────────────────────────────
  // Auditable trace of how the app runs + where it fails. Every event lands in an
  // in-memory ring buffer (window.__eoglog.buffer()) regardless of console level,
  // so the harness/tests can assert on it. Console output: lifecycle (info) +
  // failures (warn/error) by default; add ?debug=1 (or localStorage eog:debug=1)
  // for the granular API/render/trim trace.
  const LOG = (function(){
    const t0=performance.now(), buf=[], CAP=800;
    const LV={debug:10,info:20,warn:30,error:40};
    const p=new URLSearchParams(location.search);
    let thr=(p.has("debug")||localStorage.getItem("eog:debug")==="1")?LV.debug:LV.info;
    function emit(level,msg,data){
      const rec={t:+(performance.now()-t0).toFixed(1),level,msg,data};
      buf.push(rec); if(buf.length>CAP) buf.shift();
      if(LV[level]>=thr||level==="warn"||level==="error"){
        const css=level==="error"?"color:#ff6b81":level==="warn"?"color:#fbbf24":level==="debug"?"color:#8b93a7":"color:#7ef0b0";
        (console[level]||console.log)(`%c[eog +${rec.t}ms] ${level.toUpperCase()} ${msg}`,css,data===undefined?"":data);
      }
      return rec;
    }
    return {
      debug:(m,d)=>emit("debug",m,d), info:(m,d)=>emit("info",m,d),
      warn:(m,d)=>emit("warn",m,d), error:(m,d)=>emit("error",m,d),
      time:(label)=>{const s=performance.now();return (extra)=>emit("debug",`${label} ${(performance.now()-s).toFixed(0)}ms`,extra);},
      buffer:()=>buf.slice(), since:(t)=>buf.filter(r=>r.t>=t), setLevel:(l)=>{thr=LV[l]||thr;},
    };
  })();
  window.__eoglog = LOG;

  // ── API ─────────────────────────────────────────────────────────────────────
  // Single fetch chokepoint: every request is timed + logged, every non-OK or
  // network failure is logged at error level (this is "where it's failing").
  function j(u,o){
    const method=(o&&o.method)||"GET", done=LOG.time(`${method} ${u}`);
    return fetch(u,o).then(r=>{
      if(!r.ok){ LOG.error(`HTTP ${r.status} ${method} ${u}`); return Promise.reject(r.status); }
      done({status:r.status});
      return r.status===204?null:r.json();
    }).catch(e=>{ if(typeof e!=="number") LOG.error(`fetch failed ${method} ${u}`,String(e)); throw e; });
  }
  const API = {
    list:    () => j("/api/recordings"),
    detail:  (id) => j(`/api/recordings/${id}`),
    ribbon:  (id,f,w,t0,t1) => j(`/api/recordings/${id}/ribbon?filter=${f}&width=${w}`+(t0!=null?`&t0=${t0}&t1=${t1}`:"")),
    channels:(id,rows,w) => j(`/api/recordings/${id}/channels?width=${w}`+(rows?`&rows=${rows.join(",")}`:"")),
    eegRows: (id) => j(`/api/recordings/${id}/eeg_rows`),
    health:  (id,t0,t1) => j(`/api/recordings/${id}/health`+(t0!=null?`?t0=${t0}&t1=${t1}`:"")),
    putTrim: (id,t0,t1) => j(`/api/recordings/${id}/trim`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({t0,t1})}),
    delTrim: (id) => j(`/api/recordings/${id}/trim`,{method:"DELETE"}),
  };

  // Live rail % over the current keep-window. Single-flight throttle: keep only
  // the latest requested window in flight so dragging stays smooth and the badge
  // tracks the kept signal (computed server-side from raw samples, not decimated).
  const RAIL = { full:0, fetching:false, want:null };
  function setRailBadge(pct){
    const e=document.querySelector(".railpct"); if(!e)return;
    e.textContent=`RAIL ${pct.toFixed(1)}%`; e.style.color = pct>0 ? THEME.rail : THEME.status.ok;
    LOG.debug(`rail badge ${pct.toFixed(1)}%`);
  }
  function refreshRail(id,t0,t1){
    RAIL.want=[id,t0,t1];
    if(RAIL.fetching) return;
    RAIL.fetching=true;
    (function go(){
      const [cid,a,b]=RAIL.want;
      API.health(cid,a,b).then(h=>{
        setRailBadge(h.rail_pct);
        const [nid,na,nb]=RAIL.want;
        if(nid!==cid||na!==a||nb!==b) go(); else RAIL.fetching=false;
      }).catch(()=>{ RAIL.fetching=false; });
    })();
  }

  const state = { recs:[], rec:null, detail:null, filt:"raw", chans:2, trim:{}, ribbon:null, channels:null, eegRows:null };
  const RB = { cv:null, t:null, rid:null, draw:null, drag:null };

  // ── canvas drawing (shared) ─────────────────────────────────────────────────
  function setup(cv,h){ const dpr=window.devicePixelRatio||1, w=cv.clientWidth||900; cv.width=w*dpr; cv.height=h*dpr; cv.style.height=h+"px"; const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h); return {ctx,w,h}; }
  function envelope(ctx,t,mn,mx,xm,ym,fill,stroke){
    ctx.beginPath(); ctx.moveTo(xm(t[0]),ym(mx[0]));
    for(let i=1;i<t.length;i++) ctx.lineTo(xm(t[i]),ym(mx[i]));
    for(let i=t.length-1;i>=0;i--) ctx.lineTo(xm(t[i]),ym(mn[i]));
    ctx.closePath(); ctx.fillStyle=fill; ctx.fill();
    if(stroke){ ctx.beginPath(); for(let i=0;i<t.length;i++){const y=ym((mn[i]+mx[i])/2); i?ctx.lineTo(xm(t[i]),y):ctx.moveTo(xm(t[i]),y);} ctx.strokeStyle=stroke; ctx.lineWidth=1.1; ctx.stroke(); }
  }
  function drawTrace(cv,t,series,opts){
    opts=opts||{}; const {ctx,w,h}=setup(cv,opts.height||70);
    const padT=7, padB=opts.ruler?18:7, x0=t[0], x1=t[t.length-1];
    const xm=(tt)=>lerp(PADL,w-PADR,(tt-x0)/(x1-x0||1));
    const wLo=opts.scaleWindow?opts.scaleWindow.t0:x0, wHi=opts.scaleWindow?opts.scaleWindow.t1:x1;
    const inWin=(i)=>t[i]>=wLo&&t[i]<=wHi;
    const prim=series.filter(s=>s.primary), scaleSeries=prim.length?prim:series;
    let off=0;
    if(opts.center){const m=[];for(const s of scaleSeries)for(let i=0;i<t.length;i++)if(inWin(i))m.push((s.mn[i]+s.mx[i])/2);off=median(m);}
    let A;
    if(opts.robust){const av=[];for(const s of scaleSeries)for(let i=0;i<t.length;i++)if(inWin(i))av.push(Math.abs(s.mn[i]-off),Math.abs(s.mx[i]-off));A=pctl(av,99)*1.15;}
    else{A=1e-6;for(const s of scaleSeries)for(let i=0;i<t.length;i++)if(inWin(i))A=Math.max(A,Math.abs(s.mn[i]-off),Math.abs(s.mx[i]-off));A*=1.12;}
    A=Math.max(A,1e-6);
    const ym=(v)=>lerp(padT,h-padB,(A-(v-off))/(2*A));

    ctx.fillStyle=THEME.panel; ctx.fillRect(0,0,w,h);
    ctx.strokeStyle=THEME.grid; ctx.lineWidth=1;
    for(const f of [0.25,0.5,0.75]){const y=lerp(padT,h-padB,f);ctx.beginPath();ctx.moveTo(PADL,y);ctx.lineTo(w-PADR,y);ctx.stroke();}
    ctx.strokeStyle=THEME.zero; ctx.beginPath(); ctx.moveTo(PADL,ym(0)); ctx.lineTo(w-PADR,ym(0)); ctx.stroke();

    if(opts.ceil && opts.ceil<A*0.99){
      ctx.fillStyle=THEME.railFill;
      ctx.fillRect(PADL,padT,w-PADR-PADL,ym(opts.ceil)-padT);
      ctx.fillRect(PADL,ym(-opts.ceil),w-PADR-PADL,(h-padB)-ym(-opts.ceil));
      ctx.strokeStyle=THEME.rail; ctx.setLineDash([4,3]);
      for(const c of [opts.ceil,-opts.ceil]){ctx.beginPath();ctx.moveTo(PADL,ym(c));ctx.lineTo(w-PADR,ym(c));ctx.stroke();}
      ctx.setLineDash([]); ctx.fillStyle=THEME.rail; ctx.font="10px "+THEME.mono; ctx.textAlign="right"; ctx.fillText("RAIL",w-PADR-3,ym(opts.ceil)+11);
    }
    if(opts.ev){
      const lr=opts.ev.filter(e=>e.label==="LEFT"||e.label==="RIGHT"), letters=opts.evLabels&&lr.length<=24;
      for(const e of lr){ if(e.t<x0||e.t>x1)continue; const x=xm(e.t),col=e.label==="LEFT"?THEME.evL:THEME.evR;
        ctx.strokeStyle=col;ctx.globalAlpha=0.22;ctx.beginPath();ctx.moveTo(x,padT);ctx.lineTo(x,h-padB);ctx.stroke();ctx.globalAlpha=1;
        if(letters){ctx.fillStyle=col;ctx.font="9px "+THEME.mono;ctx.textAlign="center";ctx.fillText(e.label[0],x,padT+9);} }
    }
    for(const s of series) envelope(ctx,t,s.mn,s.mx,xm,ym,s.fill,s.stroke);

    if(opts.trim){
      const {t0,t1}=opts.trim;
      ctx.fillStyle=THEME.trimDim;
      if(t0>x0) ctx.fillRect(PADL,padT,xm(t0)-PADL,(h-padB)-padT);
      if(t1<x1) ctx.fillRect(xm(t1),padT,(w-PADR)-xm(t1),(h-padB)-padT);
      if(opts.trimHandles!==false){
        for(const tt of [t0,t1]){ if(tt<=x0||tt>=x1)continue; const x=xm(tt);
          ctx.strokeStyle=THEME.trimEdge;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,padT);ctx.lineTo(x,h-padB);ctx.stroke();
          ctx.fillStyle=THEME.trimEdge;ctx.fillRect(x-3,padT,6,13); }
      }
    }
    ctx.fillStyle=THEME.muted; ctx.font="10px "+THEME.mono; ctx.textAlign="right";
    const lab=(v)=>v>=1000?(v/1000).toFixed(0)+"k":v.toFixed(0);
    ctx.fillText(`+${lab(A)}`,PADL-5,padT+9); ctx.fillText(`-${lab(A)}`,PADL-5,h-padB-2);
    if(opts.name){ctx.fillStyle=opts.nameColor||THEME.text;ctx.textAlign="left";ctx.font="600 11px "+THEME.ui;ctx.fillText(opts.name,PADL+4,padT+11);}
    if(opts.ruler){ctx.fillStyle=THEME.muted;ctx.font="10px "+THEME.mono;ctx.textAlign="center";for(let i=0;i<=8;i++){const tt=lerp(x0,x1,i/8);ctx.fillText(tt.toFixed(0)+"s",xm(tt),h-4);}}
  }
  function drawSpark(cv,spark,color){ const {ctx,w,h}=setup(cv,22); const xm=(i)=>lerp(2,w-2,i/(spark.length-1)),ym=(v)=>lerp(3,h-3,(1-v)/2); ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath(); spark.forEach((v,i)=>i?ctx.lineTo(xm(i),ym(v)):ctx.moveTo(xm(i),ym(v))); ctx.stroke(); }

  // ── data loading ────────────────────────────────────────────────────────────
  async function loadSignal(){
    const id=state.rec.id;
    const rows = state.chans===2 ? null : (state.eegRows || (state.eegRows = await API.eegRows(id)));
    const done=LOG.time(`loadSignal ${id} filter=${state.filt} chans=${state.chans}`);
    const [ribbon,channels] = await Promise.all([ API.ribbon(id,state.filt,WIDTH), API.channels(id,rows,WIDTH) ]);
    state.ribbon=ribbon; state.channels=channels;
    done({ribbonPts:ribbon.t.length, chans:channels.channels.length});
  }
  async function selectRec(rec){
    LOG.info(`select recording: ${rec.id}`,{status:rec.status,rail_pct:rec.rail_pct});
    const done=LOG.time("select->rendered");
    state.rec=rec; state.eegRows=null;
    $(".main").innerHTML='<div class="loading">loading…</div>';
    state.detail=await API.detail(rec.id);
    renderSidebar();
    await loadSignal();
    renderMain();
    done();
  }

  // ── main render ─────────────────────────────────────────────────────────────
  function renderMain(){
    const r=state.detail, main=$(".main"); main.innerHTML="";
    LOG.debug("render",{id:r.id,filter:state.filt,chans:state.chans,trimmed:!!state.trim[r.id]});

    const top=el("header","topbar"); top.appendChild(el("div","title",r.id));
    const chips=el("div","chips");
    chips.appendChild(el("span","chip",`${r.fs} Hz`));
    chips.appendChild(el("span","chip", r.gain_assumed?`gain ×${r.gain}?`:`gain ×${r.gain}`));
    chips.appendChild(el("span","chip",`${r.duration}s`));
    chips.appendChild(el("span","chip",r.subject));
    top.appendChild(chips); main.appendChild(top);

    const ev=r.events||[];

    // DERIVED zone
    const dz=el("section","zone zone-derived"), dh=el("div","zonehead");
    dh.appendChild(el("span","zlabel zlabel-d","DERIVED")); dh.appendChild(el("span","zsub","L−R ribbon"));
    RAIL.full=r.rail_pct||0;
    const railEl=el("span","railpct",`RAIL ${RAIL.full.toFixed(1)}%`);
    railEl.style.color=RAIL.full>0?THEME.rail:THEME.status.ok;
    railEl.title="% of raw samples near the ADC clip ceiling, over the kept (trim) window";
    dh.appendChild(railEl);
    const tr0=state.trim[r.id];
    if(tr0) refreshRail(r.id,tr0.t0,tr0.t1);   // active trim → show windowed value
    const fbtns=el("div","fbtns");
    for(const [id,label,kind] of FILTERS){
      const b=el("button","fbtn k-"+kind+(state.filt===id?" active":""),label);
      b.onclick=async()=>{ if(state.filt===id)return; LOG.info(`filter change: ${state.filt} -> ${id}`); state.filt=id; state.ribbon=await API.ribbon(r.id,id,WIDTH); renderMain(); };
      fbtns.appendChild(b);
    }
    dh.appendChild(fbtns); dz.appendChild(dh);
    const rpane=el("div","ribbonpane"), rcv=el("canvas"); rpane.appendChild(rcv); dz.appendChild(rpane);
    const trimbar=el("div","trimbar"); trimbar.appendChild(el("span","trimlabel","TRIM"));
    const trimInfo=el("span","triminfo",""); trimbar.appendChild(trimInfo);
    const resetBtn=el("button","trimreset","Reset");
    resetBtn.onclick=()=>{ delete state.trim[r.id]; API.delTrim(r.id); renderSidebar(); renderMain(); };
    trimbar.appendChild(resetBtn); dz.appendChild(trimbar); main.appendChild(dz);

    // RAW zone
    const rz=el("section","zone zone-raw"), rh=el("div","zonehead");
    rh.appendChild(el("span","zlabel zlabel-r","RAW"));
    rh.appendChild(el("span","zsub", state.chans===2?"L / R electrodes":"all 8 board channels"));
    const chTog=el("button","chtoggle", state.chans===2?"2 ch ▸ 8":"8 ch ▸ 2");
    chTog.onclick=async()=>{ state.chans=state.chans===2?8:2; LOG.info(`channels toggle: ${state.chans}ch`); await loadSignal(); renderMain(); };
    rh.appendChild(chTog); rz.appendChild(rh);
    const stack=el("div","stack");
    const chCanvas=state.channels.channels.map((ch,i)=>{ const row=el("div","chrow"),cv=el("canvas"); row.appendChild(cv); stack.appendChild(row); return {cv,ch,last:i===state.channels.channels.length-1}; });
    rz.appendChild(stack); main.appendChild(rz);

    // draw
    function drawRibbon(){
      const rb=state.ribbon, tr=state.trim[r.id]||null;
      drawTrace(rcv, rb.t, [{mn:rb.mn,mx:rb.mx,fill:THEME.ribFill,stroke:THEME.rib,primary:true}],
        { height:225, ceil:rb.ceil_uv, ev, evLabels:true, center:true, ruler:true, trim:tr, scaleWindow:tr,
          name:`R−L · ${FILTER_LABEL[rb.filter]}`+(rb.unit==="µV/s"?" (µV/s)":""), nameColor:THEME.rib });
      if(tr){ const cut=(r.duration-(tr.t1-tr.t0)); trimInfo.innerHTML=`analysis window <b>${tr.t0.toFixed(0)}–${tr.t1.toFixed(0)}s</b> · ${cut.toFixed(0)}s excluded · <span class="gate">FFT &amp; stats use this window</span>`; resetBtn.style.display=""; }
      else{ trimInfo.innerHTML=`full <b>${r.duration.toFixed(0)}s</b> · drag the handles to trim — <span class="safe">raw stays untouched</span>`; resetBtn.style.display="none"; }
    }
    function drawChannels(){
      const ct=state.channels.t, tr=state.trim[r.id]||null;
      chCanvas.forEach(({cv,ch,last})=>{ const col=THEME.palette[(ch.row-1)%8];
        drawTrace(cv, ct, [{mn:ch.mn,mx:ch.mx,fill:hexA(col,THEME.chFillA),stroke:col}],
          { height: state.chans===2?138:84, ceil:state.channels.ceil_uv, ev, center:true, robust:true,
            scaleWindow:tr, trim:tr, trimHandles:false, name:ch.label, nameColor:col, ruler:last }); });
    }
    function drawAll(){ drawRibbon(); drawChannels(); }
    RB.cv=rcv; RB.t=state.ribbon.t; RB.rid=r.id; RB.draw=drawAll; RB.drag=null;
    rcv.addEventListener("mousedown",(e)=>{ const t=RB.t, tt=evtToT(e), span=t[t.length-1]-t[0], cur=state.trim[r.id], near=(a)=>Math.abs(a-tt)<span*0.02;
      if(cur&&near(cur.t0)) RB.drag="t0"; else if(cur&&near(cur.t1)) RB.drag="t1";
      else { state.trim[r.id]={t0:tt,t1:tt}; RB.drag="t1"; }   // press-drag to select keep-window
      LOG.debug(`trim grab ${RB.drag} @ ${tt.toFixed(1)}s`);
      drawAll(); e.preventDefault(); });
    requestAnimationFrame(drawAll);
  }

  function evtToT(e){ const rect=RB.cv.getBoundingClientRect(), x=e.clientX-rect.left, t=RB.t, x0=t[0], x1=t[t.length-1];
    return Math.min(x1,Math.max(x0, x0+(x1-x0)*(x-PADL)/(rect.width-PADL-PADR))); }
  window.addEventListener("mousemove",(e)=>{ if(!RB.drag||!RB.rid)return; const cur=state.trim[RB.rid]; if(!cur)return;
    const tt=evtToT(e); if(RB.drag==="t0")cur.t0=Math.min(tt,cur.t1-1); else cur.t1=Math.max(tt,cur.t0+1);
    RB.draw(); refreshRail(RB.rid,cur.t0,cur.t1); });               // rail % tracks the drag
  window.addEventListener("mouseup",()=>{
    if(RB.drag&&RB.rid){ const tr=state.trim[RB.rid];
      if(tr){ if(Math.abs(tr.t1-tr.t0)<2){ LOG.info(`trim clear: ${RB.rid}`); delete state.trim[RB.rid]; API.delTrim(RB.rid); if(RB.draw)RB.draw(); setRailBadge(RAIL.full); }
              else { LOG.info(`trim commit: ${RB.rid} [${tr.t0.toFixed(1)}, ${tr.t1.toFixed(1)}]s`); API.putTrim(RB.rid,tr.t0,tr.t1); refreshRail(RB.rid,tr.t0,tr.t1); } renderSidebar(); } }
    RB.drag=null; });

  // ── sidebar ─────────────────────────────────────────────────────────────────
  function renderSidebar(){
    const list=$(".reclist"); list.innerHTML="";
    for(const r of state.recs){
      const li=el("li","recitem"+(state.rec&&state.rec.id===r.id?" sel":""));
      const head=el("div","rihead");
      head.appendChild(el("span","rsub",r.subject));            // name
      head.appendChild(el("span","rdate",fmtYMD(r.id)));        // date  Y-M-D
      head.appendChild(el("span","rlen",`${Math.round(r.duration)}s`));  // length, rounded
      li.appendChild(head);
      const spark=el("canvas","spark"); li.appendChild(spark);
      li.onclick=()=>selectRec(r); list.appendChild(li);
      requestAnimationFrame(()=>drawSpark(spark,r.spark,THEME.side));
    }
  }

  async function init(){
    LOG.info("init: loading recordings");
    try{
      state.recs=await API.list();
    }catch(err){ LOG.error("init failed: API unreachable",String(err)); $(".main").innerHTML=`<div class="empty">Could not reach the API (${err}). Is scripts/serve_viewer.py running?</div>`; return; }
    const nTrim=state.recs.filter(r=>r.trim).length;
    state.recs.forEach(r=>{ if(r.trim) state.trim[r.id]=r.trim; });
    LOG.info(`recordings loaded: ${state.recs.length} (${nTrim} trimmed)`);
    renderSidebar();
    const q=new URLSearchParams(location.search).get("rec");
    const def=state.recs.find(r=>r.id===q)||state.recs[0];
    if(def) await selectRec(def);
    window.addEventListener("resize",()=>{ renderSidebar(); if(state.rec)renderMain(); });
    setTimeout(()=>{window.__ready=true; LOG.info("ready");},300);
  }
  document.addEventListener("DOMContentLoaded",init);
})();
