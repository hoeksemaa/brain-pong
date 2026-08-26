/* BrainPong — EOG Data Portal (static, public).
 * Reads the baked ./portal-data/ tree (no server): meta.json + manifest.json +
 * rec/<id>.json. Every recording is anonymized at bake time — no real names here.
 * Filter the corpus by subject/date/tag/quality, then view any recording as three
 * raw traces: the R−L difference the game reads, plus each electrode alone. */
(function () {
  "use strict";
  const THEME = {
    panel: "#10141d", grid: "#212838", zero: "#33405c", text: "#e6e9f0", muted: "#8b93a7",
    mono: "ui-monospace,Menlo,monospace", ui: "Inter,system-ui,sans-serif",
    rib: "#7ef0b0", ribFill: "rgba(126,240,176,0.12)",
    rail: "#ff6b81", railFill: "rgba(255,107,129,0.08)",
    evL: "#5b8cff", evR: "#ff9d5c", evO: "#5b6b86", evRest: "#93a0b8",
    L: "#4ea8ff", R: "#ff9d5c",
    status: { ok: "#34d399", railing: "#fb7185", flat: "#fbbf24" },
    dim: "rgba(8,10,16,0.70)", edge: "#7ef0b0",   // studio viewer's trim colours
  };
  const PADL = 46, PADR = 10;
  const NARROW = 430;                       // below this a canvas needs a tighter gutter
  const padL = w => w < NARROW ? 38 : PADL;  // must match on both the draw and hit-test side
  const $ = (s, r = document) => r.querySelector(s);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  function median(a){const b=a.slice().sort((x,y)=>x-y),n=b.length;if(!n)return 0;const m=n>>1;return n%2?b[m]:(b[m-1]+b[m])/2;}
  function pctl(a,p){const b=a.slice().sort((x,y)=>x-y);if(!b.length)return 0;return b[clamp(Math.floor(p/100*b.length),0,b.length-1)];}
  const hexA = (h, a) => { const n = parseInt(h.slice(1), 16); return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`; };
  const humanDur = s => s >= 60 ? `${Math.floor(s/60)}m${String(Math.round(s%60)).padStart(2,"0")}s` : `${Math.round(s)}s`;
  const QUALITY = {   // plain-language names for the bake's status field
    ok: "good", railing: "clipped", flat: "flat",
  };
  // Display text for an event tag ("tournament-1" -> "Tournament 1"), read from the
  // bake's own labels so adding a third tournament is a bake-only change. "none" is
  // the sentinel a Set (and a URL) can carry for "tagged with no event".
  const tourLabel = v => {
    if (v === "none" || v === null) return "Other days";
    const hit = (state.meta?.tags.tournament.values || []).find(x => x.v === v);
    return hit ? hit.label : v;
  };
  const ISODATE = /^\d{4}-\d{2}-\d{2}$/;
  const debounce = (fn, ms) => { let h; return () => { clearTimeout(h); h = setTimeout(fn, ms); }; };

  // ── data layer (pure static fetch) ──────────────────────────────────────────
  const base = "./portal-data";
  // Pages answers a missing file with an HTML page, so without the ok check the
  // failure arrives as a SyntaxError quoting "<!DOCTYPE " instead of the status.
  const grab = u => fetch(u).then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
  const DATA = {
    meta: () => grab(`${base}/meta.json`),
    manifest: () => grab(`${base}/manifest.json`),
    rec: id => grab(`${base}/rec/${encodeURIComponent(id)}.json`),
  };

  const state = {
    meta: null, recs: [], filtered: [], sel: null, detail: null,
    view: null,        // {t0,t1} analysis window, null = whole recording
    drag: null,        // {t0,t1} window being dragged out right now
    edge: null,        // "t0" | "t1" — which edge the pointer is dragging
    sortDesc: true,
    f: { q: "", tour: new Set(), st: new Set(), qual: new Set(), df: "", dt: "" },
  };

  const PLOT = { d: null, l: null, r: null };   // the three live canvases
  let pendingView = null;                                  // ?t0/?t1 from the URL, applied once

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
  const EVCOL = { left: THEME.evL, right: THEME.evR, rest: THEME.evRest, marker: THEME.evO };
  const EVGLYPH = { left: "←", right: "→", rest: "·" };   // marker gets no glyph: its tick already is the mark
  // The plot and the legend must agree on which marks exist, so both go through here.
  function evPick(events, x0, x1) {
    let evs = events || [];
    if (evs.length > 50) evs = evs.filter(e => e.kind !== "marker");   // a cue flood is unreadable; the gaze kinds carry the meaning
    const shown = evs.filter(e => e.t >= x0 && e.t <= x1);
    return { shown, dropped: (events ? events.length : 0) - shown.length };
  }
  function drawTrace(cv, t, series, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(cv, opts.height || 70);
    if (!t || !t.length) { ctx.fillStyle = THEME.panel; ctx.fillRect(0,0,w,h); return; }
    const padT = 8, padB = opts.ruler ? 18 : 8, x0 = t[0], x1 = t[t.length-1];
    const PADL = padL(w), narrow = w < NARROW;
    // The time axis is ALWAYS the whole recording — it never stretches. A window
    // only changes the y-scale, so the highlighted span is stretched vertically to
    // fill the panel while every sample stays at the same x position.
    let slo = 0, shi = t.length - 1;
    if (opts.scaleTo) {
      const a = Math.min(opts.scaleTo.t0, opts.scaleTo.t1), b = Math.max(opts.scaleTo.t0, opts.scaleTo.t1);
      while (slo < shi && t[slo + 1] <= a) slo++;
      while (shi > slo && t[shi - 1] >= b) shi--;
    }
    const xm = tt => lerp(PADL, w - PADR, (tt - x0) / (x1 - x0 || 1));   // PADL shadows the const: gutter is width-dependent
    let off = 0;
    if (opts.center) { const m=[]; for (const s of series) for (let i=slo;i<=shi;i++) m.push((s.mn[i]+s.mx[i])/2); off = median(m); }
    let A;
    if (opts.robust) { const av=[]; for (const s of series) for (let i=slo;i<=shi;i++) av.push(Math.abs(s.mn[i]-off),Math.abs(s.mx[i]-off)); A=pctl(av,99)*1.15; }
    else { A=1e-6; for (const s of series) for (let i=slo;i<=shi;i++) A=Math.max(A,Math.abs(s.mn[i]-off),Math.abs(s.mx[i]-off)); A*=1.12; }
    A = Math.max(A, 1e-6);
    const ym = v => lerp(padT, h - padB, (A - (v - off)) / (2 * A));

    ctx.fillStyle = THEME.panel; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;
    for (const f of [0.25,0.5,0.75]){const y=lerp(padT,h-padB,f);ctx.beginPath();ctx.moveTo(PADL,y);ctx.lineTo(w-PADR,y);ctx.stroke();}
    // The panel is centred on off, so 0 is often nowhere near the middle — and on a
    // railed lead it is off the panel entirely. Draw it only where it really falls.
    if (off-A <= 0 && 0 <= off+A) { ctx.strokeStyle = THEME.zero; ctx.beginPath(); ctx.moveTo(PADL, ym(0)); ctx.lineTo(w-PADR, ym(0)); ctx.stroke(); }

    if (opts.ceil) {
      // A raw voltage cannot be compared against A, which is a half-range about off. Map
      // each rail into plot coordinates instead and test it alone: centred on off, one
      // rail is often on the panel while the other is far outside it. ym inverts: yh < yl.
      const bot = h - padB, yh = ym(opts.ceil), yl = ym(-opts.ceil);
      const hiIn = yh > padT && yh < bot, loIn = yl > padT && yl < bot;
      const ch = clamp(yh, padT, bot), cl = clamp(yl, padT, bot);
      // A panel must never look clean when the bake has called the recording clipped, and
      // three things conspire to make it: robust autoscale sets A from the 99th percentile,
      // which excludes the railed samples by construction; the envelope is clipped to the
      // box, so the excursion is cut away rather than shown; and the bake calls a recording
      // railing at 0.9*ceil (bake_portal.py:66) while the band below is drawn at 1.0*ceil,
      // so a lead resting at 0.93*ceil never reaches the line at all. The R−L panel is worst
      // hit: its bound is 2*ceil, which only 2 of 237 recordings ever reach.
      // So take the answer from the badge's own source instead of re-deriving it here.
      if (opts.railing && !hiIn && !loIn) {
        ctx.fillStyle = THEME.rail; ctx.textAlign = "right"; ctx.font = "9px "+THEME.mono;
        ctx.fillRect(PADL, padT, w-PADR-PADL, 2); ctx.fillRect(PADL, bot-2, w-PADR-PADL, 2);
        ctx.fillText("CLIPPED · limit off scale", w-PADR-3, padT+12);
      }
      if (hiIn || loIn) {
        ctx.fillStyle = THEME.railFill;
        if (hiIn) ctx.fillRect(PADL, padT, w-PADR-PADL, ch-padT);
        if (loIn) ctx.fillRect(PADL, cl, w-PADR-PADL, bot-cl);
        ctx.strokeStyle = THEME.rail; ctx.setLineDash([4,3]);
        for (const y of [hiIn?ch:null, loIn?cl:null]) { if (y===null) continue; ctx.beginPath(); ctx.moveTo(PADL,y); ctx.lineTo(w-PADR,y); ctx.stroke(); }
        ctx.setLineDash([]); ctx.fillStyle=THEME.rail; ctx.font="10px "+THEME.mono; ctx.textAlign="right";
        ctx.fillText("LIMIT", w-PADR-3, clamp(hiIn ? ch+11 : cl-4, padT+10, bot-2));
      }
    }
    if (opts.events && opts.events.length) {
      const evs = evPick(opts.events, x0, x1).shown, glyphs = evs.length <= 30;
      for (const e of evs) { const x=xm(e.t);
        const col = EVCOL[e.kind] || THEME.evO;
        ctx.strokeStyle=col; ctx.globalAlpha=.28; ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,h-padB); ctx.stroke(); ctx.globalAlpha=1;
        if (glyphs && EVGLYPH[e.kind]){ ctx.fillStyle=col; ctx.font="10px "+THEME.ui; ctx.textAlign="center"; ctx.fillText(EVGLYPH[e.kind], x, padT+9); }
      }
    }
    // robust autoscale puts ~1% of the samples outside the box by construction, and
    // unclipped they paint over the unit label and the time ruler.
    ctx.save(); ctx.beginPath(); ctx.rect(PADL, padT, w-PADR-PADL, (h-padB)-padT); ctx.clip();
    for (const s of series) envelope(ctx, t, s.mn, s.mx, xm, ym, s.fill, s.stroke);
    ctx.restore();

    if (opts.band) {   // dim the context, keep the window bright (studio viewer)
      const t0 = Math.min(opts.band.t0, opts.band.t1), t1 = Math.max(opts.band.t0, opts.band.t1);
      const xa = clamp(xm(t0), PADL, w-PADR), xb = clamp(xm(t1), PADL, w-PADR);
      ctx.fillStyle = THEME.dim;
      if (t0 > x0) ctx.fillRect(PADL, padT, xa-PADL, (h-padB)-padT);
      if (t1 < x1) ctx.fillRect(xb, padT, (w-PADR)-xb, (h-padB)-padT);
      if (opts.handles) for (const tt of [t0, t1]) {
        if (tt <= x0 || tt >= x1) continue; const x = xm(tt);
        ctx.strokeStyle = THEME.edge; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, h-padB); ctx.stroke();
        ctx.fillStyle = THEME.edge; ctx.fillRect(x-3, padT, 6, 13);
      }
    }

    ctx.fillStyle = THEME.muted; ctx.font = "10px "+THEME.mono; ctx.textAlign = "right";
    // The edges are off+A and off-A, not ±A. Printing them as ±A hides the offset, and
    // a lead pinned to the rail then reads "+0.0 / -0.0" — the flattest-looking panel on
    // the site. 1000 µV is 1 mV: switch the unit with the number so the two agree.
    const hi = off + A, lo = off - A, mv = Math.max(Math.abs(hi), Math.abs(lo)) >= 1000;
    const sc = mv ? 1000 : 1, unit = mv ? "mV" : "µV", span = 2 * A / sc;
    // Decimals come from the SPAN, not from each edge on its own. Per-edge precision let a
    // 200 µV window about -175 mV print "-175" at both ends, and a rail-pinned lead print
    // "+0.00 / -0.00" — the very string this labelling replaced.
    const dp = span >= 100 ? 0 : span >= 10 ? 1 : span >= 1 ? 2 : 3;
    const lab = (v, d = dp) => (v < 0 ? "-" : "+") + Math.abs(v / sc).toFixed(d);
    let top = lab(hi), bottom = lab(lo);
    // A constant trace has no range to print. Give its value once and name it, rather than
    // print one number twice and imply a span that is not there.
    if (top === bottom) { top = lab(off, Math.abs(off / sc) >= 100 ? 1 : 2); bottom = "flat"; }
    ctx.fillText(top, PADL-5, padT+9); ctx.fillText(bottom, PADL-5, h-padB-2);
    // The unit is not optional: mV and µV differ by 1000x, and the rotated label is hidden
    // on every phone (a 390px viewport gives a 340px canvas, under NARROW). Put it under the
    // top tick there instead of dropping it.
    if (opts.unit) {
      ctx.fillStyle = THEME.muted; ctx.font = "9px "+THEME.mono;
      if (narrow) { ctx.textAlign = "right"; ctx.fillText(unit, PADL-5, padT+20); }
      else { ctx.save(); ctx.translate(11,(padT+h-padB)/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center"; ctx.fillText(unit,0,0); ctx.restore(); }
    }
    if (opts.name){ ctx.fillStyle=opts.nameColor||THEME.text; ctx.textAlign="left"; ctx.font="600 11px "+THEME.ui; ctx.fillText(opts.name, PADL+4, padT+11); }
    if (opts.ruler){ ctx.fillStyle=THEME.muted; ctx.font="10px "+THEME.mono; ctx.textAlign="center";
      const nt = narrow ? 4 : 8;
      for(let i=0;i<=nt;i++){const tt=lerp(x0,x1,i/nt);ctx.fillText(tt.toFixed(0)+"s",xm(tt),h-4);} }
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
    if (f.tour.size && !f.tour.has(t.tournament || "none")) return false;
    if (f.st.size && !f.st.has(t.session_type)) return false;
    if (f.qual.size && !f.qual.has(r.status)) return false;
    if (f.df && r.date < f.df) return false;
    if (f.dt && r.date > f.dt) return false;
    return true;
  }
  function recompute() {
    state.filtered = state.recs.filter(passes)
      .sort((a, b) => state.sortDesc ? b.id.localeCompare(a.id) : a.id.localeCompare(b.id));
    syncURL();
    renderList();
  }
  // An empty list must name what emptied it: a junk ?df= hid all 191 rows while both
  // date boxes looked blank, because <input type=date> discards a bad value in silence.
  function activeFilters() {
    const f = state.f, a = [];
    if (f.q) a.push(`search "${f.q}"`);
    if (f.tour.size) a.push([...f.tour].map(tourLabel).join(" or "));
    if (f.st.size) a.push([...f.st].join(" or "));
    if (f.qual.size) a.push([...f.qual].map(v => QUALITY[v] || v).join(" or "));
    if (f.df) a.push(`from ${f.df}`);
    if (f.dt) a.push(`to ${f.dt}`);
    return a;
  }
  function resetFilters() {          // six active filters took six actions to clear
    const f = state.f;
    f.q = ""; f.tour.clear(); f.st.clear(); f.qual.clear(); f.df = ""; f.dt = "";
    const s = $("#search"); if (s) s.value = "";
    recompute(); renderFilters();    // the selected recording is deliberately left alone
    if (s) s.focus();                // the button that called this just removed itself
  }

  // ── URL state (shareable filtered views) ────────────────────────────────────
  function syncURL() {
    const f = state.f, p = new URLSearchParams();
    if (f.q) p.set("q", f.q);
    const setCSV = (k, s) => { if (s.size) p.set(k, [...s].join(",")); };
    setCSV("tour", f.tour); setCSV("st", f.st); setCSV("qual", f.qual);
    if (f.df) p.set("df", f.df); if (f.dt) p.set("dt", f.dt);
    if (state.sel) p.set("rec", state.sel);
    if (state.sel && state.view) { p.set("t0", state.view.t0.toFixed(3)); p.set("t1", state.view.t1.toFixed(3)); }
    const qs = p.toString();
    history.replaceState(null, "", qs ? "?" + qs : location.pathname);
  }
  // Every value is checked against what the corpus actually has: an unrecognized one
  // is dropped, never turned into a filter. ?tour=true used to mean tour=false and hide
  // every tournament recording, and ?qual=bogus emptied the list with no chip lit.
  // state.meta must already be set when this runs.
  function loadURL() {
    const p = new URLSearchParams(location.search), f = state.f;
    f.q = p.get("q") || "";
    const getCSV = (k, s, ok) => { s.clear(); const v = p.get(k); if (v) v.split(",").forEach(x => { if (ok.has(x)) s.add(x); }); };
    // ?tour= used to be the boolean 1/0 of a single "tournament" tag. Links shared
    // before the two nights were split still open the view they promised: 1 = both
    // tournaments, 0 = everything else.
    const tr = p.get("tour");
    if (tr === "1" || tr === "0") {
      f.tour.clear();
      for (const { v } of state.meta.tags.tournament.values) if ((v !== null) === (tr === "1")) f.tour.add(v || "none");
    } else {
      getCSV("tour", f.tour, new Set(state.meta.tags.tournament.values.map(x => x.v || "none")));
    }
    getCSV("st", f.st, new Set(state.meta.tags.session_type.values.map(x => x.v)));
    getCSV("qual", f.qual, new Set(Object.keys(QUALITY)));
    const day = v => ISODATE.test(v || "") ? v : "";
    f.df = day(p.get("df")); f.dt = day(p.get("dt"));
    const t0 = parseFloat(p.get("t0")), t1 = parseFloat(p.get("t1"));
    pendingView = (isFinite(t0) && isFinite(t1) && t1 > t0) ? { t0, t1 } : null;
    return p.get("rec");
  }

  // ── corpus stats (top bar) ──────────────────────────────────────────────────
  function renderStats() {
    const c = state.meta.corpus;
    // total_hours is the sum of all 191 durations, and 70 of the 121 sessions ran two
    // recordings over the same wall clock. "of signal" reads as elapsed time and is
    // wrong by 57%; the sum of durations is exactly "recorded hours".
    const y0 = c.date_start.slice(0, 4), y1 = c.date_end.slice(0, 4);
    const items = [
      [c.n_recordings, "recordings"], [c.n_named_subjects, "people"],
      [c.total_hours + "h", "recorded hours"],
      [`${c.date_start.slice(5)} – ${c.date_end.slice(5)}`, y0 === y1 ? y0 : `${y0} – ${y1}`],
    ];
    const box = $("#stats"); box.innerHTML = "";
    for (const [v, k] of items) { const c2 = el("div", "cstat"); c2.appendChild(el("div", "v", v)); c2.appendChild(el("div", "k", k)); box.appendChild(c2); }
  }

  // ── filter UI ───────────────────────────────────────────────────────────────
  // renderFilters() destroys the very button the reader just pressed, and focus falls
  // to <body>. Every count here comes from state.meta, which is fetched once and never
  // written again, so nothing in a group moves when a filter changes except which chips
  // are lit — repainting them in place is the whole update.
  function chipGroup(title, values, sel, labelFn, statusClass) {
    const g = el("div", "fgroup");
    g.setAttribute("role", "group"); g.setAttribute("aria-label", title);
    const h = el("div", "fh", title);
    const c = el("button", "clr", "clear"); c.type = "button";
    h.appendChild(c); g.appendChild(h);
    const chips = el("div", "chips"), items = [];
    const paint = () => {
      for (const [b, v] of items) { const on = sel.has(v); b.classList.toggle("on", on); b.setAttribute("aria-pressed", on); }
      c.hidden = !sel.size;
    };
    c.onclick = () => { sel.clear(); recompute(); paint(); const f0 = chips.firstElementChild; if (f0) f0.focus(); };
    for (const { v, count } of values) {
      const label = labelFn ? labelFn(v) : String(v);
      const b = el("button", "chip" + (statusClass ? " st-" + v : ""), `${label}<span class="n">${count}</span>`);
      b.type = "button";
      b.onclick = () => { sel.has(v) ? sel.delete(v) : sel.add(v); recompute(); paint(); };
      items.push([b, v]); chips.appendChild(b);
    }
    paint();
    g.appendChild(chips); return g;
  }
  function renderFilters() {
    const wrap = $("#filters"); wrap.innerHTML = "";
    const meta = state.meta, f = state.f;

    // Event: one chip per tournament plus "Other days" — a boolean toggle could not
    // say which night, and the two are what a reader wants to compare.
    const tv = meta.tags.tournament.values.map(x => ({ v: x.v || "none", count: x.count }));
    wrap.appendChild(chipGroup(meta.tags.tournament.label, tv, f.tour, tourLabel));

    wrap.appendChild(chipGroup("Session type", meta.tags.session_type.values, f.st));

    const qv = ["ok", "railing", "flat"].map(s => ({ v: s, count: (meta.corpus.quality[s] || 0) })).filter(x => x.count);
    wrap.appendChild(chipGroup("Signal quality", qv, f.qual, v => QUALITY[v] || v, true));

    // date range — the one group built outside chipGroup, so it was the one group
    // with no way back. Rebuilding it on change would take focus out of the box
    // mid-entry, so the link is built once and only shown when it has work to do.
    const dg = el("div", "fgroup"), dh = el("div", "fh", "Date range");
    dg.setAttribute("role", "group"); dg.setAttribute("aria-label", "Date range");
    const dc = el("button", "clr", "clear"); dc.type = "button";
    dh.appendChild(dc); dg.appendChild(dh);
    const syncClr = () => { dc.style.display = (f.df || f.dt) ? "" : "none"; };
    const dr = el("div", "daterow");
    // both boxes announce as "date" with nothing to tell them apart, so each is named
    const di = (val, on, id, name) => { const i = el("input"); i.type = "date"; i.id = id; i.setAttribute("aria-label", name); i.value = val; i.min = meta.corpus.date_start; i.max = meta.corpus.date_end; i.onchange = () => { on(i.value); syncClr(); recompute(); }; return i; };
    const d0 = di(f.df, v => f.df = v, "dateFrom", "Earliest date"), d1 = di(f.dt, v => f.dt = v, "dateTo", "Latest date");
    dc.onclick = () => { f.df = ""; f.dt = ""; d0.value = ""; d1.value = ""; syncClr(); recompute(); d0.focus(); };
    dr.appendChild(d0); dr.appendChild(el("span", null, "→")); dr.appendChild(d1);
    syncClr(); dg.appendChild(dr); wrap.appendChild(dg);
  }

  // ── sidebar list ────────────────────────────────────────────────────────────
  const sparkCol = r => THEME.status[r.status] || "#b388ff";
  function renderList() {
    const meta = $("#listmeta"); meta.innerHTML = "";
    const why = activeFilters();
    const left = el("span"); left.style.cssText = "display:flex;align-items:center;gap:9px";
    left.appendChild(el("span", null, `<b style="color:var(--ink)">${state.filtered.length}</b> of ${state.recs.length} recordings`));
    if (why.length) { const rb = el("button", "sortbtn", "Reset all"); rb.onclick = resetFilters; left.appendChild(rb); }
    meta.appendChild(left);
    const sb = el("button", "sortbtn", state.sortDesc ? "↓ Newest" : "↑ Oldest");
    sb.onclick = () => { state.sortDesc = !state.sortDesc; recompute(); };
    meta.appendChild(sb);

    const list = $("#reclist"); list.innerHTML = "";
    if (!state.filtered.length) {
      const li = el("li", "empty");   // textContent: the search term is whatever the reader typed
      li.textContent = why.length ? `No recordings match ${why.join(" · ")}.` : "No recordings.";
      list.appendChild(li); return;
    }
    for (const r of state.filtered) {
      // a <button> so the row takes Enter, Space and focus; the children are spans
      // because a button may only hold phrasing content.
      const li = el("li"), b = el("button", "recitem" + (state.sel === r.id ? " sel" : ""));
      b.type = "button"; b.dataset.id = r.id;
      const head = el("span", "rihead");
      head.appendChild(el("span", "rdot " + r.status));
      head.appendChild(el("span", "rsub", r.subject + (r.opponent ? ` <span style="color:var(--faint);font-weight:400">vs ${r.opponent}</span>` : "")));
      head.appendChild(el("span", "rdate", r.date.slice(5)));
      head.appendChild(el("span", "rlen", humanDur(r.duration)));
      b.appendChild(head);
      const tags = el("span", "ritags");
      if (r.tags.tournament) tags.appendChild(el("span", "ttag tour", tourLabel(r.tags.tournament).toLowerCase()));
      tags.appendChild(el("span", "ttag", r.tags.session_type));
      if (r.n_players === 2) tags.appendChild(el("span", "ttag", "2-player"));
      // quality was a 7px dot and a stroke colour and nothing else
      tags.appendChild(el("span", "ttag q-" + r.status, QUALITY[r.status] || r.status));
      b.appendChild(tags);
      const spark = el("canvas", "spark"); spark.setAttribute("aria-hidden", "true"); b.appendChild(spark);
      b.onclick = () => selectRec(r.id);
      li.appendChild(b); list.appendChild(li);
      // two rebuilds in one task (init does it on every deep link) leaves the first
      // 191 canvases detached with 191 draws still queued against them.
      requestAnimationFrame(() => { if (spark.isConnected) drawSpark(spark, r.spark, sparkCol(r)); });
    }
  }

  // Move the selection highlight in place. Selecting a recording must NOT go
  // through renderList(): that clears #reclist, which collapses the scroller and
  // throws the sidebar back to the top (and redraws all 191 sparklines for nothing).
  function markSelected() {
    for (const b of $("#reclist").querySelectorAll(".recitem")) b.classList.toggle("sel", b.dataset.id === state.sel);
  }
  // A spark's backing store is sized in device px when the row is built, so a width
  // change leaves every one of them stretched until an unrelated filter rebuilds the list.
  function redrawSparks() {
    const by = new Map(state.recs.map(r => [r.id, r]));
    for (const b of $("#reclist").querySelectorAll(".recitem")) {
      const cv = b.querySelector("canvas"), r = by.get(b.dataset.id);
      if (cv && r) drawSpark(cv, r.spark, sparkCol(r));
    }
  }

  // ── analysis window (studio viewer behaviour) ───────────────────────────────
  // Press and drag on the R−L plot to mark a window. The window is drawn over the
  // plot — dimmed outside, handles on its edges — and it stretches the trace
  // VERTICALLY only: the y-axis scales to the window while the time axis keeps
  // showing the whole recording. Press without dragging to clear it.
  const MIN_SPAN = 2;                      // seconds; viewer.js uses the same floor
  function fullRange(r) { const t = r.t; return [t[0], t[t.length-1]]; }
  function clampView(v, r) {
    if (!v || !r || !r.t || r.t.length < 2) return null;
    const [a, b] = fullRange(r);
    const t0 = clamp(Math.min(v.t0, v.t1), a, b), t1 = clamp(Math.max(v.t0, v.t1), a, b);
    return (t1 - t0) >= MIN_SPAN ? { t0, t1 } : null;
  }
  // applyView redraws without touching the URL, so a live drag can rescale every
  // frame; only the committed window (pointerup) is written to the address bar.
  function applyView(v) { state.view = v; syncZoomUI(); drawPlots(); }
  function setView(v) { applyView(v); syncURL(); }

  function syncZoomUI() {
    const box = $("#zoombox"); if (!box) return;
    box.innerHTML = ""; box.hidden = !state.view;
    if (!state.view) return;
    const { t0, t1 } = state.view, dp = (t1 - t0) >= 8 ? 1 : 2;
    box.appendChild(el("span", "zrange", `${t0.toFixed(dp)}–${t1.toFixed(dp)}s`));
    const b = el("button", "zreset", "reset");
    b.onclick = () => setView(null);
    box.appendChild(b);
  }

  // Only the R−L plot takes the drag — it is the one the window is drawn on.
  function attachWindow(cv) {
    const toT = e => {
      const rect = cv.getBoundingClientRect(), [a, b] = fullRange(state.detail), pl = padL(rect.width);
      return clamp(a + (b - a) * (e.clientX - rect.left - pl) / (rect.width - pl - PADR), a, b);
    };
    cv.addEventListener("pointerdown", e => {
      if (e.button || !state.detail) return;
      const tt = toT(e), [a, b] = fullRange(state.detail);
      const near = x => Math.abs(x - tt) < (b - a) * 0.02, v = state.view;
      // Pressing within 2% of an edge grabs THAT edge; anywhere else starts a new
      // window at the press point and drags its right edge (viewer.js:295-297).
      if (v && near(v.t0))      { state.drag = { t0: v.t0, t1: v.t1 }; state.edge = "t0"; }
      else if (v && near(v.t1)) { state.drag = { t0: v.t0, t1: v.t1 }; state.edge = "t1"; }
      else                      { state.drag = { t0: tt,   t1: tt   }; state.edge = "t1"; }
      cv.setPointerCapture(e.pointerId);
      drawPlots(); e.preventDefault();
    });
    cv.addEventListener("pointermove", e => {
      if (!state.edge) return;
      // The dragged edge is clamped 1 s clear of the fixed one, so the handles can
      // never cross and the window never flips (viewer.js:409).
      const tt = toT(e), d = state.drag;
      if (state.edge === "t0") d.t0 = Math.min(tt, d.t1 - 1);
      else                     d.t1 = Math.max(tt, d.t0 + 1);
      const v = clampView(d, state.detail);
      if (v) applyView(v); else drawPlots();      // live: the y-scale follows the drag
    });
    const end = () => {
      if (!state.edge) return;
      const d = state.drag; state.drag = null; state.edge = null;
      setView(clampView(d, state.detail));        // narrower than MIN_SPAN clears it
    };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
  }

  function drawPlots() {
    const r = state.detail; if (!r || !PLOT.d) return;
    // Mid-drag the band follows the cursor even before it is wide enough to commit,
    // so the edges track the pointer; the y-scale only follows a committed window.
    const band = state.drag || state.view, scale = state.view;
    // R − L is a difference of two electrodes, so it can swing to twice one electrode's
    // full scale before anything clips. Its LIMIT belongs there, not at ceil_uv. robust
    // keeps one rail excursion from squashing the real eye movements into a hairline.
    drawTrace(PLOT.d, r.t, [{ mn: r.diff.mn, mx: r.diff.mx, fill: THEME.ribFill, stroke: THEME.rib }], {
      height: 240, center: true, robust: true, ruler: true, ceil: r.ceil_uv * 2, events: r.events,
      railing: r.status === "railing",
      unit: "µV", name: "R − L (right minus left)", nameColor: THEME.rib,
      band, scaleTo: scale, handles: true,
    });
    drawTrace(PLOT.l, r.t, [{ mn: r.channels.l.mn, mx: r.channels.l.mx, fill: hexA(THEME.L, .12), stroke: THEME.L }],
      { height: 116, center: true, robust: true, ceil: r.ceil_uv, unit: "µV", name: "Left electrode", nameColor: THEME.L, band, scaleTo: scale, railing: r.status === "railing" });
    drawTrace(PLOT.r, r.t, [{ mn: r.channels.r.mn, mx: r.channels.r.mx, fill: hexA(THEME.R, .12), stroke: THEME.R }],
      { height: 116, center: true, robust: true, ceil: r.ceil_uv, ruler: true, unit: "µV", name: "Right electrode", nameColor: THEME.R, band, scaleTo: scale, railing: r.status === "railing" });
  }

  // ── detail view ─────────────────────────────────────────────────────────────
  // textContent, never innerHTML: a fetch failure can carry markup in its message,
  // and the parser then swallows the status code and welds the sentences together.
  function fail(msg) { const m = $("#main"); m.innerHTML = ""; const d = el("div", "empty"); d.textContent = msg; m.appendChild(d); }
  async function selectRec(id) {
    state.sel = id;
    $("#main").innerHTML = '<div class="empty">loading…</div>';
    markSelected();
    // Two fast clicks race. The id is held here and re-checked after the await, so a
    // late fetch can never paint over the row the reader actually chose, and the URL
    // is written on both paths — list, panel and address bar always name the same id.
    let rec;
    try { rec = await DATA.rec(id); }
    catch (e) {
      if (state.sel !== id) return;
      console.error("recording did not load:", id, e);
      state.detail = null; state.view = null; state.drag = null; state.edge = null;
      PLOT.d = PLOT.l = PLOT.r = null;      // the resize redraw must not find a stale canvas
      fail("This recording could not be loaded. It may no longer be part of the corpus.");
      syncURL();
      return;
    }
    if (state.sel !== id) return;
    state.detail = rec;
    state.view = clampView(pendingView, state.detail); pendingView = null; state.drag = null; state.edge = null;
    syncURL();
    renderDetail();
    // On a phone the sidebar sits above the detail, so a selection would otherwise
    // leave the reader looking at the list they just tapped.
    if (isPhone()) $("#main").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  // Everything this page has to say is on three canvases, and a canvas says nothing.
  function plotAlt(cv, r, name, nev) {
    cv.setAttribute("role", "img");
    cv.setAttribute("aria-label",
      `${name}: ${humanDur(r.duration)} at ${r.fs} samples per second, ` +
      `${(r.rail_pct || 0).toFixed(1)}% of samples at the amplifier limit, ` +
      `${nev} event mark${nev === 1 ? "" : "s"} drawn.`);
  }
  function renderDetail() {
    const r = state.detail, main = $("#main"); main.innerHTML = "";
    const row = state.recs.find(x => x.id === r.id) || {};
    const wrap = el("div", "detail");

    const head = el("div", "dhead");
    const ttl = el("div"); ttl.appendChild(el("h2", "dtitle", r.subject + (row.opponent ? ` <span class="opp">vs ${row.opponent}</span>` : "")));
    ttl.appendChild(el("div", "dsub", `${r.date} · ${r.time} · ${humanDur(r.duration)} · ${r.fs} samples/s`));
    head.appendChild(ttl); wrap.appendChild(head);

    const badges = el("div", "badges");
    if (r.tags.tournament) badges.appendChild(el("span", "badge tour", "🏆 " + tourLabel(r.tags.tournament)));
    badges.appendChild(el("span", "badge", `<b>${r.tags.session_type}</b>`));
    if (r.n_players === 2) badges.appendChild(el("span", "badge", "2-player"));
    const q = QUALITY[r.status] || r.status;
    const qb = el("span", "badge q-" + r.status, `<b>${q}</b>` + (r.rail_pct > 0 ? ` · ${r.rail_pct.toFixed(1)}% at limit` : ""));
    badges.appendChild(qb);
    wrap.appendChild(badges);

    // ── the three raw traces ──
    const card = el("div", "card");
    const ch = el("div", "cardhead");
    ch.appendChild(el("span", "clabel", "RAW SIGNAL"));
    const zb = el("span", "zoombox"); zb.id = "zoombox"; zb.hidden = true; ch.appendChild(zb);
    card.appendChild(ch);

    const pp = el("div", "plot");
    const dcv = el("canvas"); pp.appendChild(dcv);
    card.appendChild(pp);
    const stack = el("div", "stack");
    const lrow = el("div", "chrow"), lcv = el("canvas"); lrow.appendChild(lcv);
    const rrow = el("div", "chrow"), rcv = el("canvas"); rrow.appendChild(rcv);
    // The aria-label and the legend must report what the plot actually draws, so both
    // read the same selection. Only the R−L plot is given the events, so only it marks.
    const [ea, eb] = fullRange(r), ev = evPick(r.events, ea, eb);
    plotAlt(dcv, r, "R minus L, the right electrode minus the left", ev.shown.length);
    plotAlt(lcv, r, "Left electrode alone", 0);
    plotAlt(rcv, r, "Right electrode alone", 0);
    stack.appendChild(lrow); stack.appendChild(rrow); card.appendChild(stack);

    if (ev.shown.length || ev.dropped) {
      const key = el("div", "eventkey");
      const names = [...new Set(ev.shown.map(e => e.text))];
      key.appendChild(el("span", null, `<b>${ev.shown.length}</b> events drawn`));
      for (const k of names.slice(0, 6)) key.appendChild(el("span", null, k));
      if (names.length > 6) key.appendChild(el("span", null, `+${names.length - 6} more`));
      if (ev.dropped) key.appendChild(el("span", null, `${ev.dropped} not drawn`));   // never let the count overstate the plot
      card.appendChild(key);
    }
    wrap.appendChild(card);
    main.appendChild(wrap);

    PLOT.d = dcv; PLOT.l = lcv; PLOT.r = rcv;
    attachWindow(dcv);          // the window is drawn on the R−L plot, so it owns the drag
    syncZoomUI();
    requestAnimationFrame(drawPlots);
  }

  // ── dashboard (nothing selected) ────────────────────────────────────────────
  function renderDashboard() {
    const c = state.meta.corpus, q = c.quality || {}, tot = c.n_recordings || 1;
    const main = $("#main"); main.innerHTML = "";
    const d = el("div", "dash");
    d.appendChild(el("h2", null, "Play Pong with your eyes."));
    d.appendChild(el("p", null,
      `Horizontal <b>EOG</b>: the small voltage beside your eyes when you glance left or right. ` +
      `Two electrodes read it, and it drives a Pong paddle. Every name here is a pseudonym.`));
    // quality bar
    const bar = el("div", "qbar");
    for (const [k, col] of [["ok", THEME.status.ok], ["railing", THEME.status.railing], ["flat", THEME.status.flat]]) {
      if (!q[k]) continue; const seg = el("div", "qseg"); seg.style.width = (100 * q[k] / tot) + "%"; seg.style.background = col; bar.appendChild(seg);
    }
    d.appendChild(bar);
    const leg = el("div", "qlegend");
    leg.appendChild(el("span", null, `<b style="color:${THEME.status.ok}">${q.ok||0}</b> good`));
    leg.appendChild(el("span", null, `<b style="color:${THEME.status.railing}">${q.railing||0}</b> clipped`));
    leg.appendChild(el("span", null, `<b style="color:${THEME.status.flat}">${q.flat||0}</b> flat`));
    d.appendChild(leg);
    main.appendChild(d);
  }
  // The dashboard is the only place that says what EOG is, and state.sel was a one-way
  // door: opening one recording hid that explanation for the rest of the session.
  function goHome() {
    state.sel = null; state.detail = null; state.view = null; state.drag = null; state.edge = null;
    PLOT.d = PLOT.l = PLOT.r = null; pendingView = null;
    markSelected();
    syncURL();          // state.sel is null, so rec/t0/t1 fall out of the query string
    renderDashboard();
  }
  window.__goHome = goHome;   // the brand link calls this; still exported so a test can drive it

  // ── about modal ─────────────────────────────────────────────────────────────
  function showAbout() {
    const m = $("#aboutModal");
    m.innerHTML = "";
    const box = el("div", "box");
    box.innerHTML = `<h2 id="aboutTitle">About this site</h2>
      <p><b>BrainPong</b> is a hobby project: play Pong with your eyes. Two electrodes beside the eyes read
      <b>electrooculography</b> (EOG) — the natural voltage of the eye, which shifts when you glance left or right.
      A Cerelog X8 board digitizes the voltage, and the game moves the paddle from it in real time.
      The sensor only receives; nothing is ever applied to a person.</p>
      <p><b>How to read a recording:</b></p>
      <ul>
        <li><b style="color:#7ef0b0">R − L</b> — the right electrode minus the left electrode. This is the signal
        the game reads. A glance right moves the trace up; a glance left moves it down.</li>
        <li><b style="color:#4ea8ff">Left</b> / <b style="color:#ff9d5c">Right</b> — each electrode alone, unfiltered.</li>
        <li><b style="color:#ff6b81">LIMIT</b> — red dashed lines mark how far that plot's trace can go before it
        clips: one electrode's full scale on <b>Left</b> and <b>Right</b>, and twice that on <b>R − L</b>, because the
        difference of two electrodes can reach twice as far. A trace pinned on the lines is clipped, not real signal.
        Recordings marked <i>clipped</i> or <i>flat</i> had electrode problems.</li>
        <li>Vertical colored lines are game or cue events (for example a <b style="color:#5b8cff">LEFT</b> /
        <b style="color:#ff9d5c">RIGHT</b> cue during a training run).</li>
      </ul>
      <p>Each plot is a min/max envelope of the full recording, so short spikes stay visible.
      All subjects played with consent, and every name here is a pseudonym &mdash; the project
      owner's included. <i>Unattributed</i> covers recordings made at a station where no name was
      entered; those cannot be traced to a person at all, and are pooled rather than counted as
      individuals.</p>
      <p>More: <a href="cmrr/">why bad electrode contact ruins the signal ↗</a> ·
      <a href="https://github.com/hoeksemaa/brain-pong">source code on GitHub ↗</a></p>
      <button class="tbtn close" type="button" autofocus>Close</button>`;
    m.appendChild(box);
    // showModal(), not a hidden attribute: Escape, the top layer, focus containment and
    // focus restore to the About button all come from the platform this way.
    m.showModal();
    box.querySelector(".close").onclick = () => m.close();
    m.onclick = e => { if (e.target === m) m.close(); };   // the dialog fills the viewport, so it is the scrim
  }

  const isPhone = () => window.matchMedia("(max-width:860px)").matches;
  let wasPhone = null;
  function syncFilterBox() {           // open on desktop, a disclosure on a phone
    const fb = $("#filterbox"); if (!fb) return;
    const phone = isPhone();
    // Only on an actual breakpoint crossing. Setting it on every resize would slam
    // the filters shut mid-scroll, because a phone fires resize when the address
    // bar hides.
    if (phone !== wasPhone) { fb.open = !phone; wasPhone = phone; }
  }

  // ── init ────────────────────────────────────────────────────────────────────
  async function init() {
    let meta, manifest;
    try { [meta, manifest] = await Promise.all([DATA.meta(), DATA.manifest()]); }
    catch (e) {
      // Whoever reads this screen followed a shared link and cannot run a bake script.
      console.error("portal data did not load:", e);
      fail("The recordings could not be loaded. Please reload the page, or try again shortly.");
      return;
    }
    state.meta = meta; state.recs = manifest.recordings;   // loadURL checks the taxonomy, so meta must be in place first
    const selId = loadURL();
    renderStats();
    renderFilters();
    $("#search").value = state.f.q;
    const search = debounce(recompute, 150);               // otherwise every keystroke rebuilds 191 rows and 191 canvases
    $("#search").oninput = e => { state.f.q = e.target.value; search(); };
    $("#aboutBtn").onclick = showAbout;
    // the brand is the only way back to the dashboard once a recording is open
    $("#homeLink").onclick = e => { e.preventDefault(); goHome(); };
    syncFilterBox();
    recompute();
    if (selId && state.recs.some(r => r.id === selId)) await selectRec(selId);
    else renderDashboard();
    window.addEventListener("popstate", () => {
      const id = loadURL();
      renderFilters(); $("#search").value = state.f.q; recompute();
      if (!id || !state.recs.some(r => r.id === id)) { goHome(); return; }
      // Same recording, different window: rescale in place rather than re-fetching it.
      if (id === state.sel && state.detail) { applyView(clampView(pendingView, state.detail)); pendingView = null; }
      else selectRec(id);
    });
    let lastW = window.innerWidth;
    const onResize = debounce(() => {
      // Height-only resizes are the phone address bar showing/hiding. Acting on them
      // would throw away the reader's scroll position on a scroll.
      if (window.innerWidth === lastW) return;
      lastW = window.innerWidth;
      if (state.detail) drawPlots();   // redraw the canvases; renderDetail would rebuild #main and reset the scroll
      redrawSparks();
    }, 150);
    window.addEventListener("resize", () => { syncFilterBox(); onResize(); });
    window.__ready = true;
  }
  document.addEventListener("DOMContentLoaded", init);
})();
