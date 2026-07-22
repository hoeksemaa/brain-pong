/* BrainPong — EOG Data Portal (static, public).
 * Reads the baked ./portal-data/ tree (no server): meta.json + manifest.json +
 * rec/<id>.json. Every recording is anonymized at bake time — no real names here.
 * Filter the whole corpus by name/date/tag/pipeline/quality/numeric range, then
 * view any recording through each historical filtering paradigm ("lens"). */
(function () {
  "use strict";
  const THEME = {
    panel: "#10141d", grid: "#212838", zero: "#33405c", text: "#e6e9f0", muted: "#8b93a7",
    mono: "ui-monospace,Menlo,monospace", ui: "Inter,system-ui,sans-serif",
    rib: "#7ef0b0", ribFill: "rgba(126,240,176,0.12)",
    rail: "#ff6b81", railFill: "rgba(255,107,129,0.08)",
    thr: "#fbbf24", evL: "#5b8cff", evR: "#ff9d5c", evO: "#5b6b86",
    L: "#4ea8ff", R: "#ff9d5c",
    status: { ok: "#34d399", railing: "#fb7185", flat: "#fbbf24" },
  };
  const PADL = 46, PADR = 10;
  const $ = (s, r = document) => r.querySelector(s);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  function median(a){const b=a.slice().sort((x,y)=>x-y),n=b.length;if(!n)return 0;const m=n>>1;return n%2?b[m]:(b[m-1]+b[m])/2;}
  function pctl(a,p){const b=a.slice().sort((x,y)=>x-y);if(!b.length)return 0;return b[clamp(Math.floor(p/100*b.length),0,b.length-1)];}
  const hexA = (h, a) => { const n = parseInt(h.slice(1), 16); return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`; };
  const humanDur = s => s >= 60 ? `${Math.floor(s/60)}m${String(Math.round(s%60)).padStart(2,"0")}s` : `${Math.round(s)}s`;
  const MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const fmtDate = (d, yr) => { if (!d) return "—"; const [y,m,dd] = d.split("-"); return `${MON[+m-1]} ${+dd}` + (yr ? ` ${y}` : ""); };

  // ── data layer (pure static fetch) ──────────────────────────────────────────
  const base = "./portal-data";
  const DATA = {
    meta: () => fetch(`${base}/meta.json`).then(r => r.json()),
    manifest: () => fetch(`${base}/manifest.json`).then(r => r.json()),
    rec: id => fetch(`${base}/rec/${encodeURIComponent(id)}.json`).then(r => { if (!r.ok) throw r.status; return r.json(); }),
  };

  const state = {
    meta: null, recs: [], filtered: [], sel: null, detail: null, lens: "raw",
    sortDesc: true,
    f: { q: "", tour: null, st: new Set(), rig: new Set(), det: new Set(), pipe: new Set(),
         prep: new Set(), elec: new Set(), qual: new Set(), df: "", dt: "",
         dmin: 0, dmax: Infinity, rmax: 100 },
  };

  // ── canvas kit ──────────────────────────────────────────────────────────────
  function setup(cv, h) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 900;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }
  function envelope(ctx, t, mn, mx, xm, ym, fill, stroke) {
    ctx.beginPath(); ctx.moveTo(xm(t[0]), ym(mx[0]));
    for (let i = 1; i < t.length; i++) ctx.lineTo(xm(t[i]), ym(mx[i]));
    for (let i = t.length - 1; i >= 0; i--) ctx.lineTo(xm(t[i]), ym(mn[i]));
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.beginPath(); for (let i = 0; i < t.length; i++){const y=ym((mn[i]+mx[i])/2); i?ctx.lineTo(xm(t[i]),y):ctx.moveTo(xm(t[i]),y);} ctx.strokeStyle=stroke; ctx.lineWidth=1.1; ctx.stroke(); }
  }
  function drawTrace(cv, t, series, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(cv, opts.height || 70);
    if (!t || !t.length) { ctx.fillStyle = THEME.panel; ctx.fillRect(0,0,w,h); return; }
    const padT = 8, padB = opts.ruler ? 18 : 8, x0 = t[0], x1 = t[t.length-1];
    const xm = tt => lerp(PADL, w - PADR, (tt - x0) / (x1 - x0 || 1));
    let off = 0;
    if (opts.center) { const m=[]; for (const s of series) for (let i=0;i<t.length;i++) m.push((s.mn[i]+s.mx[i])/2); off = median(m); }
    let A;
    if (opts.robust) { const av=[]; for (const s of series) for (let i=0;i<t.length;i++) av.push(Math.abs(s.mn[i]-off),Math.abs(s.mx[i]-off)); A=pctl(av,99)*1.15; }
    else { A=1e-6; for (const s of series) for (let i=0;i<t.length;i++) A=Math.max(A,Math.abs(s.mn[i]-off),Math.abs(s.mx[i]-off)); A*=1.12; }
    if (opts.minA) A = Math.max(A, opts.minA);
    A = Math.max(A, 1e-6);
    const ym = v => lerp(padT, h - padB, (A - (v - off)) / (2 * A));

    ctx.fillStyle = THEME.panel; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;
    for (const f of [0.25,0.5,0.75]){const y=lerp(padT,h-padB,f);ctx.beginPath();ctx.moveTo(PADL,y);ctx.lineTo(w-PADR,y);ctx.stroke();}
    ctx.strokeStyle = THEME.zero; ctx.beginPath(); ctx.moveTo(PADL, ym(0)); ctx.lineTo(w-PADR, ym(0)); ctx.stroke();

    if (opts.ceil && opts.ceil < A * 0.99) {
      ctx.fillStyle = THEME.railFill;
      ctx.fillRect(PADL, padT, w-PADR-PADL, ym(opts.ceil)-padT);
      ctx.fillRect(PADL, ym(-opts.ceil), w-PADR-PADL, (h-padB)-ym(-opts.ceil));
      ctx.strokeStyle = THEME.rail; ctx.setLineDash([4,3]);
      for (const c of [opts.ceil,-opts.ceil]){ctx.beginPath();ctx.moveTo(PADL,ym(c));ctx.lineTo(w-PADR,ym(c));ctx.stroke();}
      ctx.setLineDash([]); ctx.fillStyle=THEME.rail; ctx.font="10px "+THEME.mono; ctx.textAlign="right"; ctx.fillText("RAIL",w-PADR-3,ym(opts.ceil)+11);
    }
    if (opts.thr) {   // znorm σ-threshold lines
      ctx.strokeStyle = THEME.thr; ctx.setLineDash([5,4]); ctx.globalAlpha=.8;
      for (const c of [opts.thr,-opts.thr]){ if(c<A){ctx.beginPath();ctx.moveTo(PADL,ym(c));ctx.lineTo(w-PADR,ym(c));ctx.stroke();} }
      ctx.setLineDash([]); ctx.globalAlpha=1; ctx.fillStyle=THEME.thr; ctx.font="9px "+THEME.mono; ctx.textAlign="right";
      if(opts.thr<A) ctx.fillText(`${opts.thr}σ`, w-PADR-3, ym(opts.thr)-3);
    }
    if (opts.events && opts.events.length) {
      let evs = opts.events;
      if (evs.length > 50) evs = evs.filter(e => /^(LEFT|RIGHT|REST)$/.test(e.label));
      const letters = evs.length <= 30;
      for (const e of evs) { if (e.t<x0||e.t>x1) continue; const x=xm(e.t);
        const col = e.label==="LEFT"?THEME.evL : e.label==="RIGHT"?THEME.evR : THEME.evO;
        ctx.strokeStyle=col; ctx.globalAlpha=.28; ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,h-padB); ctx.stroke(); ctx.globalAlpha=1;
        if (letters){ ctx.fillStyle=col; ctx.font="9px "+THEME.mono; ctx.textAlign="center"; ctx.fillText(e.label[0], x, padT+9); }
      }
    }
    for (const s of series) envelope(ctx, t, s.mn, s.mx, xm, ym, s.fill, s.stroke);

    ctx.fillStyle = THEME.muted; ctx.font = "10px "+THEME.mono; ctx.textAlign = "right";
    const lab = v => v>=1000?(v/1000).toFixed(v>=10000?0:1)+"k":v.toFixed(A<10?1:0);
    ctx.fillText(`+${lab(A)}`, PADL-5, padT+9); ctx.fillText(`-${lab(A)}`, PADL-5, h-padB-2);
    if (opts.unit){ ctx.save(); ctx.translate(11,(padT+h-padB)/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center"; ctx.fillStyle=THEME.muted; ctx.font="9px "+THEME.mono; ctx.fillText(opts.unit,0,0); ctx.restore(); }
    if (opts.name){ ctx.fillStyle=opts.nameColor||THEME.text; ctx.textAlign="left"; ctx.font="600 11px "+THEME.ui; ctx.fillText(opts.name, PADL+4, padT+11); }
    if (opts.ruler){ ctx.fillStyle=THEME.muted; ctx.font="10px "+THEME.mono; ctx.textAlign="center"; for(let i=0;i<=8;i++){const tt=lerp(x0,x1,i/8);ctx.fillText(tt.toFixed(0)+"s",xm(tt),h-4);} }
  }
  function drawSpark(cv, spark, color) {
    const { ctx, w, h } = setup(cv, 20);
    if (!spark || !spark.length) return;
    const xm=i=>lerp(2,w-2,i/(spark.length-1)), ym=v=>lerp(3,h-3,(1-v)/2);
    ctx.strokeStyle=color; ctx.lineWidth=1; ctx.beginPath();
    spark.forEach((v,i)=>i?ctx.lineTo(xm(i),ym(v)):ctx.moveTo(xm(i),ym(v))); ctx.stroke();
  }

  // ── filtering ───────────────────────────────────────────────────────────────
  function passes(r) {
    const f = state.f, t = r.tags;
    if (f.q && !r.subject.toLowerCase().includes(f.q.toLowerCase())) return false;
    if (f.tour !== null && t.tournament !== f.tour) return false;
    if (f.st.size && !f.st.has(t.session_type)) return false;
    if (f.rig.size && !f.rig.has(t.rig_board)) return false;
    if (f.prep.size && !f.prep.has(t.cleaning_regimen)) return false;
    if (f.elec.size && !f.elec.has(t.electrode_type)) return false;
    if (f.det.size && !f.det.has(r.detector)) return false;
    if (f.pipe.size && !f.pipe.has(r.pipeline)) return false;
    if (f.qual.size && !f.qual.has(r.status)) return false;
    if (f.df && r.date < f.df) return false;
    if (f.dt && r.date > f.dt) return false;
    if (r.duration < f.dmin || r.duration > f.dmax) return false;
    if (r.rail_pct > f.rmax) return false;
    return true;
  }
  function recompute() {
    state.filtered = state.recs.filter(passes)
      .sort((a, b) => state.sortDesc ? b.id.localeCompare(a.id) : a.id.localeCompare(b.id));
    syncURL();
    renderList();
  }

  // ── URL state (shareable filtered views) ────────────────────────────────────
  function syncURL() {
    const f = state.f, p = new URLSearchParams();
    if (f.q) p.set("q", f.q);
    if (f.tour !== null) p.set("tour", f.tour ? "1" : "0");
    const setCSV = (k, s) => { if (s.size) p.set(k, [...s].join(",")); };
    setCSV("st", f.st); setCSV("rig", f.rig); setCSV("det", f.det); setCSV("pipe", f.pipe);
    setCSV("prep", f.prep); setCSV("qual", f.qual);
    if (f.df) p.set("df", f.df); if (f.dt) p.set("dt", f.dt);
    if (f.dmin > 0) p.set("dmin", f.dmin); if (f.dmax !== Infinity) p.set("dmax", f.dmax);
    if (f.rmax < 100) p.set("rmax", f.rmax);
    if (state.sel) p.set("rec", state.sel);
    if (state.lens !== "raw") p.set("lens", state.lens);
    history.replaceState(null, "", "?" + p.toString());
  }
  function loadURL() {
    const p = new URLSearchParams(location.search), f = state.f;
    f.q = p.get("q") || "";
    f.tour = p.has("tour") ? p.get("tour") === "1" : null;
    const getCSV = (k, s, cast) => { const v = p.get(k); if (v) v.split(",").forEach(x => s.add(cast ? cast(x) : x)); };
    getCSV("st", f.st); getCSV("rig", f.rig); getCSV("det", f.det); getCSV("pipe", f.pipe);
    getCSV("prep", f.prep); getCSV("qual", f.qual);
    f.df = p.get("df") || ""; f.dt = p.get("dt") || "";
    if (p.has("dmin")) f.dmin = +p.get("dmin"); if (p.has("dmax")) f.dmax = +p.get("dmax");
    if (p.has("rmax")) f.rmax = +p.get("rmax");
    if (p.has("lens")) state.lens = p.get("lens");
    return p.get("rec");
  }

  // ── corpus stats (top bar) ──────────────────────────────────────────────────
  function renderStats() {
    const c = state.meta.corpus, q = c.quality || {};
    const items = [
      [c.n_recordings, "recordings"], [c.n_sessions, "sessions"],
      [c.total_hours + "h", "signal"], [c.n_subjects, "subjects"],
      [`${c.date_start.slice(5)} – ${c.date_end.slice(5)}`, "date span"],
      [`<span class="qok">${q.ok||0}</span> / <span class="qbad">${q.railing||0}</span> / <span class="qwarn">${q.flat||0}</span>`, "ok / rail / flat"],
    ];
    const box = $("#stats"); box.innerHTML = "";
    for (const [v, k] of items) { const c2 = el("div", "cstat"); c2.appendChild(el("div", "v", v)); c2.appendChild(el("div", "k", k)); box.appendChild(c2); }
  }

  // ── filter UI ───────────────────────────────────────────────────────────────
  function chipGroup(title, values, sel, onToggle, statusClass) {
    const g = el("div", "fgroup");
    const h = el("h4", null, title);
    if (sel.size) { const c = el("span", "clr", "clear"); c.onclick = () => { sel.clear(); recompute(); renderFilters(); }; h.appendChild(c); }
    g.appendChild(h);
    const chips = el("div", "chips");
    for (const { v, count } of values) {
      const label = v === true ? "Yes" : v === false ? "No" : v == null ? "—" : String(v);
      const active = sel.has(v);
      const cls = "chip" + (active ? " on" : "") + (statusClass ? " st-" + v : "");
      const b = el("button", cls, `${label}<span class="n">${count}</span>`);
      b.onclick = () => { active ? sel.delete(v) : sel.add(v); recompute(); renderFilters(); };
      chips.appendChild(b);
    }
    g.appendChild(chips); return g;
  }
  function renderFilters() {
    const wrap = $("#filters"); wrap.innerHTML = "";
    const meta = state.meta, f = state.f;

    // tournament (boolean)
    const tv = meta.tags.tournament.values;
    const tg = el("div", "fgroup"); tg.appendChild(el("h4", null, "Tournament"));
    const tchips = el("div", "chips");
    for (const { v, count } of tv) {
      const b = el("button", "chip" + (f.tour === v ? " on tour" : ""), `${v ? "Tournament" : "Other"}<span class="n">${count}</span>`);
      b.onclick = () => { f.tour = f.tour === v ? null : v; recompute(); renderFilters(); };
      tchips.appendChild(b);
    }
    tg.appendChild(tchips); wrap.appendChild(tg);

    wrap.appendChild(chipGroup("Session type", meta.tags.session_type.values, f.st, recompute));
    wrap.appendChild(chipGroup("Rig / board", meta.tags.rig_board.values, f.rig, recompute));

    // pipeline / detector — the paradigm axis, as a list filter
    const dv = Object.entries(meta.pipeline.detector).map(([k, n]) => ({ v: k === "null" ? null : k, count: n })).filter(x => x.v);
    wrap.appendChild(chipGroup("Detector", dv, f.det, recompute));
    const pv = Object.entries(meta.pipeline.version).map(([k, n]) => ({ v: k === "null" ? null : k, count: n })).filter(x => x.v);
    wrap.appendChild(chipGroup("Pipeline", pv, f.pipe, recompute));

    wrap.appendChild(chipGroup("Skin prep", meta.tags.cleaning_regimen.values, f.prep, recompute));

    // quality (status colors)
    const qv = ["ok", "railing", "flat"].map(s => ({ v: s, count: (state.meta.corpus.quality[s] || 0) })).filter(x => x.count);
    wrap.appendChild(chipGroup("Signal quality", qv, f.qual, recompute, true));

    // electrode — single value in this corpus, shown as static info
    const ev = meta.tags.electrode_type.values.filter(x => x.count);
    if (ev.length === 1) {
      const g = el("div", "fgroup"); g.appendChild(el("h4", null, "Electrode"));
      const chips = el("div", "chips");
      chips.appendChild(el("span", "chip static", `${ev[0].v}<span class="n">${ev[0].count}</span>`));
      g.appendChild(chips); wrap.appendChild(g);
    } else {
      wrap.appendChild(chipGroup("Electrode", ev, f.elec, recompute));
    }

    // date range
    const dg = el("div", "fgroup"); dg.appendChild(el("h4", null, "Date range"));
    const dr = el("div", "daterow");
    const di = (val, on) => { const i = el("input"); i.type = "date"; i.value = val; i.min = meta.corpus.date_start; i.max = meta.corpus.date_end; i.onchange = () => { on(i.value); recompute(); }; return i; };
    dr.appendChild(di(f.df, v => f.df = v)); dr.appendChild(el("span", null, "→")); dr.appendChild(di(f.dt, v => f.dt = v));
    dg.appendChild(dr); wrap.appendChild(dg);

    // numeric: max rail %
    const rg = el("div", "fgroup"); rg.appendChild(el("h4", null, "Max rail %"));
    const rr = el("div", "rangerow"); const ri = el("input"); ri.type = "range"; ri.min = 0; ri.max = 100; ri.step = 1; ri.value = f.rmax;
    const rvl = el("span", "rv", `≤ ${f.rmax}%`); ri.oninput = () => { f.rmax = +ri.value; rvl.textContent = `≤ ${f.rmax}%`; recompute(); };
    rr.appendChild(ri); rr.appendChild(rvl); rg.appendChild(rr); wrap.appendChild(rg);

    // numeric: min duration
    const gg = el("div", "fgroup"); gg.appendChild(el("h4", null, "Min duration"));
    const gr = el("div", "rangerow"); const gi = el("input"); gi.type = "range"; gi.min = 0; gi.max = 120; gi.step = 5; gi.value = Math.min(f.dmin, 120);
    const gvl = el("span", "rv", f.dmin ? `≥ ${f.dmin}s` : "any"); gi.oninput = () => { f.dmin = +gi.value; gvl.textContent = f.dmin ? `≥ ${f.dmin}s` : "any"; recompute(); };
    gr.appendChild(gi); gr.appendChild(gvl); gg.appendChild(gr); wrap.appendChild(gg);
  }

  // ── sidebar list ────────────────────────────────────────────────────────────
  function renderList() {
    const meta = $("#listmeta"); meta.innerHTML = "";
    meta.appendChild(el("span", null, `<b style="color:var(--ink)">${state.filtered.length}</b> of ${state.recs.length}`));
    const sb = el("button", "sortbtn", state.sortDesc ? "↓ Newest" : "↑ Oldest");
    sb.onclick = () => { state.sortDesc = !state.sortDesc; recompute(); };
    meta.appendChild(sb);

    const list = $("#reclist"); list.innerHTML = "";
    if (!state.filtered.length) { list.appendChild(el("li", "empty", "No recordings match these filters.")); return; }
    for (const r of state.filtered) {
      const li = el("li", "recitem" + (state.sel === r.id ? " sel" : ""));
      const head = el("div", "rihead");
      head.appendChild(el("span", "rdot " + r.status));
      head.appendChild(el("span", "rsub", r.subject + (r.opponent ? ` <span style="color:var(--faint);font-weight:400">vs ${r.opponent}</span>` : "")));
      head.appendChild(el("span", "rdate", r.date.slice(5)));
      head.appendChild(el("span", "rlen", humanDur(r.duration)));
      li.appendChild(head);
      const tags = el("div", "ritags");
      if (r.tags.tournament) tags.appendChild(el("span", "ttag tour", "tournament"));
      tags.appendChild(el("span", "ttag", r.tags.session_type));
      tags.appendChild(el("span", "ttag", r.tags.rig_board));
      if (r.detector) tags.appendChild(el("span", "ttag", r.detector));
      li.appendChild(tags);
      const spark = el("canvas", "spark"); li.appendChild(spark);
      li.onclick = () => selectRec(r.id);
      list.appendChild(li);
      requestAnimationFrame(() => drawSpark(spark, r.spark, THEME.status[r.status] || "#b388ff"));
    }
  }

  // ── detail view ─────────────────────────────────────────────────────────────
  async function selectRec(id) {
    state.sel = id;
    $("#main").innerHTML = '<div class="empty">loading…</div>';
    renderList();
    try { state.detail = await DATA.rec(id); }
    catch (e) { $("#main").innerHTML = `<div class="empty">Could not load recording (${e}).</div>`; return; }
    syncURL();
    renderDetail();
  }
  function lensMeta(id) { return state.meta.lenses.find(l => l.id === id) || state.meta.lenses[0]; }
  function renderDetail() {
    const r = state.detail, main = $("#main"); main.innerHTML = "";
    const wrap = el("div", "detail");

    const head = el("div", "dhead");
    const ttl = el("div"); ttl.appendChild(el("div", "dtitle", r.subject + (r.opponent ? ` <span class="opp">vs ${r.opponent}</span>` : "")));
    ttl.appendChild(el("div", "dsub", `${r.id} · ${r.date} ${r.time} · ${humanDur(r.duration)} @ ${r.fs} Hz`));
    head.appendChild(ttl); wrap.appendChild(head);

    const badges = el("div", "badges");
    if (r.tags.tournament) badges.appendChild(el("span", "badge tour", "🏆 Tournament"));
    badges.appendChild(el("span", "badge", `Session <b>${r.tags.session_type}</b>`));
    badges.appendChild(el("span", "badge", `Rig <b>${r.tags.rig_board}</b>`));
    badges.appendChild(el("span", "badge", `Electrode <b>${r.tags.electrode_type}</b>`));
    badges.appendChild(el("span", "badge", `Skin prep <b>${r.tags.cleaning_regimen}</b>`));
    if (r.detector) badges.appendChild(el("span", "badge pipe", `Detector <b>${r.detector}</b>${r.pipeline ? " · " + r.pipeline : ""}`));
    if (r.n_players) badges.appendChild(el("span", "badge", `${r.n_players}-player`));
    wrap.appendChild(badges);

    // ── DSP paradigm lens card ──
    const lc = el("div", "card");
    const lh = el("div", "cardhead");
    lh.appendChild(el("span", "clabel", "PARADIGM LENS"));
    lh.appendChild(el("span", "csub", "view the raw recording through each era of the filtering pipeline"));
    const rail = el("span", "railpill", `RAIL ${r.rail_pct.toFixed(1)}%`);
    rail.style.color = r.rail_pct > 0 ? THEME.rail : THEME.status.ok;
    lh.appendChild(rail); lc.appendChild(lh);

    const lw = el("div", "lenswrap");
    const lrail = el("div", "lensrail");
    for (const L of state.meta.lenses) {
      const node = el("div", "lensnode" + (state.lens === L.id ? " on" : ""));
      node.appendChild(el("div", "lensdot"));
      node.appendChild(el("div", "lenslbl", L.label));
      node.appendChild(el("div", "lensdate", fmtDate(L.date)));
      node.onclick = () => { state.lens = L.id; syncURL(); renderDetail(); };
      lrail.appendChild(node);
    }
    lw.appendChild(lrail);
    const lm = lensMeta(state.lens);
    const blurb = el("div", "lensblurb", `<span class="ld">${lm.label}${lm.date ? " · " + fmtDate(lm.date, true) : ""} · ${lm.unit}</span><br>${lm.blurb}`);
    lw.appendChild(blurb); lc.appendChild(lw);

    const pp = el("div", "plot"); const rcv = el("canvas"); pp.appendChild(rcv); lc.appendChild(pp);
    wrap.appendChild(lc);

    // event key
    if (r.events && r.events.length) {
      const key = el("div", "eventkey");
      const kinds = [...new Set(r.events.map(e => e.label))];
      const shown = kinds.slice(0, 6);
      key.appendChild(el("span", null, `<b>${r.events.length}</b> event markers:`));
      for (const k of shown) key.appendChild(el("span", null, k));
      if (kinds.length > 6) key.appendChild(el("span", null, `+${kinds.length - 6} more`));
      lc.appendChild(key);
    }

    // ── raw electrodes card ──
    const rc = el("div", "card");
    const rhh = el("div", "cardhead");
    rhh.appendChild(el("span", "clabel", "RAW ELECTRODES"));
    rhh.appendChild(el("span", "csub", "left / right, unfiltered (µV)"));
    rc.appendChild(rhh);
    const stack = el("div", "stack");
    const lrow = el("div", "chrow"), lcv = el("canvas"); lrow.appendChild(lcv);
    const rrow = el("div", "chrow"), rcv2 = el("canvas"); rrow.appendChild(rcv2);
    stack.appendChild(lrow); stack.appendChild(rrow); rc.appendChild(stack); wrap.appendChild(rc);

    main.appendChild(wrap);

    requestAnimationFrame(() => {
      const lens = r.lenses[state.lens], lmeta = lensMeta(state.lens);
      drawTrace(rcv, r.t, [{ mn: lens.mn, mx: lens.mx, fill: THEME.ribFill, stroke: THEME.rib }], {
        height: 240, center: true, ruler: true, ceil: lens.ceil, thr: lens.thr,
        minA: lens.thr ? lens.thr * 1.15 : 0, events: r.events, unit: lmeta.unit,
        name: `R−L · ${lmeta.label}`, nameColor: THEME.rib,
      });
      drawTrace(lcv, r.t, [{ mn: r.channels.l.mn, mx: r.channels.l.mx, fill: hexA(THEME.L, .12), stroke: THEME.L }],
        { height: 116, center: true, robust: true, ceil: r.ceil_uv, name: "L", nameColor: THEME.L });
      drawTrace(rcv2, r.t, [{ mn: r.channels.r.mn, mx: r.channels.r.mx, fill: hexA(THEME.R, .12), stroke: THEME.R }],
        { height: 116, center: true, robust: true, ceil: r.ceil_uv, ruler: true, name: "R", nameColor: THEME.R });
    });
  }

  // ── dashboard (nothing selected) ────────────────────────────────────────────
  function renderDashboard() {
    const c = state.meta.corpus, q = c.quality || {}, tot = c.n_recordings || 1;
    const main = $("#main"); main.innerHTML = "";
    const d = el("div", "dash");
    d.appendChild(el("h2", null, "Play Pong with your eyes."));
    d.appendChild(el("p", null,
      `Every recording here is horizontal <b>EOG</b> — the tiny voltages that appear near the eyes when you glance left or right, read by a two-electrode montage on a Cerelog X8 and used to drive a Pong paddle. This portal is a public, read-only browser over the whole labeled corpus: <b>${c.n_recordings}</b> recordings across <b>${c.n_sessions}</b> sessions and <b>${c.n_subjects}</b> subjects, ${c.date_start} → ${c.date_end}. Names are anonymized.`));
    d.appendChild(el("p", null,
      `Use the filters on the left to slice by tag, pipeline, quality, date, or duration. Open any recording to view it through each <b>historical filtering paradigm</b> — from the first EOG band in April to the velocity/matched detector the game runs today.`));
    // quality bar
    const bar = el("div", "qbar");
    for (const [k, col] of [["ok", THEME.status.ok], ["railing", THEME.status.railing], ["flat", THEME.status.flat]]) {
      if (!q[k]) continue; const seg = el("div", "qseg"); seg.style.width = (100 * q[k] / tot) + "%"; seg.style.background = col; bar.appendChild(seg);
    }
    d.appendChild(bar);
    const leg = el("div", "qlegend");
    leg.appendChild(el("span", null, `<b style="color:${THEME.status.ok}">${q.ok||0}</b> ok`));
    leg.appendChild(el("span", null, `<b style="color:${THEME.status.railing}">${q.railing||0}</b> railing`));
    leg.appendChild(el("span", null, `<b style="color:${THEME.status.flat}">${q.flat||0}</b> flat`));
    d.appendChild(leg);
    d.appendChild(el("p", null, `<span style="color:var(--muted)">← pick a recording to begin.</span>`));
    main.appendChild(d);
  }

  // ── about modal ─────────────────────────────────────────────────────────────
  function showAbout() {
    const m = $("#aboutModal");
    m.innerHTML = "";
    const box = el("div", "box");
    box.innerHTML = `<h2>About this portal</h2>
      <p><b>BrainPong</b> is a hobby project: play Pong with your eyes. A two-electrode horizontal <b>electrooculography</b> (EOG) montage on a Cerelog X8 reads left/right glances and moves the paddle in real time. The sensor is receive-only — it measures the naturally-occurring skin voltages near the eyes; nothing is applied back to anyone.</p>
      <p>This page browses the recording corpus that backs the detector work. Each recording can be viewed through the successive <b>filtering paradigms</b> the pipeline passed through, so you can see the signal-processing progression over time. Subject names are anonymized; the underlying arrays are the project owner's and consenting players' EOG.</p>
      <p>The <a href="cmrr/">CMRR explainer ↗</a> covers why electrode-impedance mismatch degrades common-mode rejection.</p>
      <button class="tbtn close">Close</button>`;
    m.appendChild(box); m.hidden = false;
    const close = () => { m.hidden = true; };
    box.querySelector(".close").onclick = close;
    m.onclick = e => { if (e.target === m) close(); };
  }

  // ── init ────────────────────────────────────────────────────────────────────
  async function init() {
    let meta, manifest;
    try { [meta, manifest] = await Promise.all([DATA.meta(), DATA.manifest()]); }
    catch (e) { $("#main").innerHTML = `<div class="empty">Could not load portal data (${e}).<br>Run <code>python scripts/bake_portal.py</code> then reload.</div>`; return; }
    state.meta = meta; state.recs = manifest.recordings;
    const selId = loadURL();
    renderStats();
    renderFilters();
    $("#search").value = state.f.q;
    $("#search").oninput = e => { state.f.q = e.target.value; recompute(); };
    $("#aboutBtn").onclick = showAbout;
    recompute();
    if (selId && state.recs.some(r => r.id === selId)) await selectRec(selId);
    else renderDashboard();
    window.addEventListener("resize", () => { if (state.detail && state.sel) renderDetail(); });
    window.__ready = true;
  }
  document.addEventListener("DOMContentLoaded", init);
})();
