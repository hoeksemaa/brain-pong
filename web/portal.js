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
    evL: "#5b8cff", evR: "#ff9d5c", evO: "#5b6b86",
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
  const QUALITY = {   // plain-language names for the bake's status field
    ok: "good", railing: "clipped", flat: "flat",
  };

  // ── data layer (pure static fetch) ──────────────────────────────────────────
  const base = "./portal-data";
  const DATA = {
    meta: () => fetch(`${base}/meta.json`).then(r => r.json()),
    manifest: () => fetch(`${base}/manifest.json`).then(r => r.json()),
    rec: id => fetch(`${base}/rec/${encodeURIComponent(id)}.json`).then(r => { if (!r.ok) throw r.status; return r.json(); }),
  };

  const state = {
    meta: null, recs: [], filtered: [], sel: null, detail: null,
    sortDesc: true,
    f: { q: "", tour: null, st: new Set(), qual: new Set(), df: "", dt: "" },
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
      ctx.setLineDash([]); ctx.fillStyle=THEME.rail; ctx.font="10px "+THEME.mono; ctx.textAlign="right"; ctx.fillText("LIMIT",w-PADR-3,ym(opts.ceil)+11);
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

  // ── URL state (shareable filtered views) ────────────────────────────────────
  function syncURL() {
    const f = state.f, p = new URLSearchParams();
    if (f.q) p.set("q", f.q);
    if (f.tour !== null) p.set("tour", f.tour ? "1" : "0");
    const setCSV = (k, s) => { if (s.size) p.set(k, [...s].join(",")); };
    setCSV("st", f.st); setCSV("qual", f.qual);
    if (f.df) p.set("df", f.df); if (f.dt) p.set("dt", f.dt);
    if (state.sel) p.set("rec", state.sel);
    const qs = p.toString();
    history.replaceState(null, "", qs ? "?" + qs : location.pathname);
  }
  function loadURL() {
    const p = new URLSearchParams(location.search), f = state.f;
    f.q = p.get("q") || "";
    f.tour = p.has("tour") ? p.get("tour") === "1" : null;
    const getCSV = (k, s) => { const v = p.get(k); if (v) v.split(",").forEach(x => s.add(x)); };
    getCSV("st", f.st); getCSV("qual", f.qual);
    f.df = p.get("df") || ""; f.dt = p.get("dt") || "";
    return p.get("rec");
  }

  // ── corpus stats (top bar) ──────────────────────────────────────────────────
  function renderStats() {
    const c = state.meta.corpus;
    const items = [
      [c.n_recordings, "recordings"], [c.n_subjects, "people"],
      [c.total_hours + "h", "of signal"],
      [`${c.date_start.slice(5)} – ${c.date_end.slice(5)}`, "2026"],
    ];
    const box = $("#stats"); box.innerHTML = "";
    for (const [v, k] of items) { const c2 = el("div", "cstat"); c2.appendChild(el("div", "v", v)); c2.appendChild(el("div", "k", k)); box.appendChild(c2); }
  }

  // ── filter UI ───────────────────────────────────────────────────────────────
  function chipGroup(title, values, sel, labelFn, statusClass) {
    const g = el("div", "fgroup");
    const h = el("h4", null, title);
    if (sel.size) { const c = el("span", "clr", "clear"); c.onclick = () => { sel.clear(); recompute(); renderFilters(); }; h.appendChild(c); }
    g.appendChild(h);
    const chips = el("div", "chips");
    for (const { v, count } of values) {
      const label = labelFn ? labelFn(v) : String(v);
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

    // tournament day (boolean toggle)
    const tv = meta.tags.tournament.values;
    const tg = el("div", "fgroup"); tg.appendChild(el("h4", null, "Event"));
    const tchips = el("div", "chips");
    for (const { v, count } of tv) {
      const b = el("button", "chip" + (f.tour === v ? " on tour" : ""), `${v ? "Tournament day" : "Other days"}<span class="n">${count}</span>`);
      b.onclick = () => { f.tour = f.tour === v ? null : v; recompute(); renderFilters(); };
      tchips.appendChild(b);
    }
    tg.appendChild(tchips); wrap.appendChild(tg);

    wrap.appendChild(chipGroup("Session type", meta.tags.session_type.values, f.st));

    const qv = ["ok", "railing", "flat"].map(s => ({ v: s, count: (meta.corpus.quality[s] || 0) })).filter(x => x.count);
    wrap.appendChild(chipGroup("Signal quality", qv, f.qual, v => QUALITY[v] || v, true));

    // date range
    const dg = el("div", "fgroup"); dg.appendChild(el("h4", null, "Date range"));
    const dr = el("div", "daterow");
    const di = (val, on) => { const i = el("input"); i.type = "date"; i.value = val; i.min = meta.corpus.date_start; i.max = meta.corpus.date_end; i.onchange = () => { on(i.value); recompute(); }; return i; };
    dr.appendChild(di(f.df, v => f.df = v)); dr.appendChild(el("span", null, "→")); dr.appendChild(di(f.dt, v => f.dt = v));
    dg.appendChild(dr); wrap.appendChild(dg);
  }

  // ── sidebar list ────────────────────────────────────────────────────────────
  function renderList() {
    const meta = $("#listmeta"); meta.innerHTML = "";
    meta.appendChild(el("span", null, `<b style="color:var(--ink)">${state.filtered.length}</b> of ${state.recs.length} recordings`));
    const sb = el("button", "sortbtn", state.sortDesc ? "↓ Newest" : "↑ Oldest");
    sb.onclick = () => { state.sortDesc = !state.sortDesc; recompute(); };
    meta.appendChild(sb);

    const list = $("#reclist"); list.innerHTML = "";
    if (!state.filtered.length) { list.appendChild(el("li", "empty", "No recordings match these filters.")); return; }
    for (const r of state.filtered) {
      const li = el("li", "recitem" + (state.sel === r.id ? " sel" : ""));
      li.dataset.id = r.id;
      const head = el("div", "rihead");
      head.appendChild(el("span", "rdot " + r.status));
      head.appendChild(el("span", "rsub", r.subject + (r.opponent ? ` <span style="color:var(--faint);font-weight:400">vs ${r.opponent}</span>` : "")));
      head.appendChild(el("span", "rdate", r.date.slice(5)));
      head.appendChild(el("span", "rlen", humanDur(r.duration)));
      li.appendChild(head);
      const tags = el("div", "ritags");
      if (r.tags.tournament) tags.appendChild(el("span", "ttag tour", "tournament"));
      tags.appendChild(el("span", "ttag", r.tags.session_type));
      if (r.n_players === 2) tags.appendChild(el("span", "ttag", "2-player"));
      li.appendChild(tags);
      const spark = el("canvas", "spark"); li.appendChild(spark);
      li.onclick = () => selectRec(r.id);
      list.appendChild(li);
      requestAnimationFrame(() => drawSpark(spark, r.spark, THEME.status[r.status] || "#b388ff"));
    }
  }

  // Move the selection highlight in place. Selecting a recording must NOT go
  // through renderList(): that clears #reclist, which collapses the scroller and
  // throws the sidebar back to the top (and redraws all 191 sparklines for nothing).
  function markSelected() {
    for (const li of $("#reclist").children) li.classList.toggle("sel", li.dataset.id === state.sel);
  }

  // ── detail view ─────────────────────────────────────────────────────────────
  async function selectRec(id) {
    state.sel = id;
    $("#main").innerHTML = '<div class="empty">loading…</div>';
    markSelected();
    try { state.detail = await DATA.rec(id); }
    catch (e) { $("#main").innerHTML = `<div class="empty">Could not load recording (${e}).</div>`; return; }
    syncURL();
    renderDetail();
  }
  function renderDetail() {
    const r = state.detail, main = $("#main"); main.innerHTML = "";
    const row = state.recs.find(x => x.id === r.id) || {};
    const wrap = el("div", "detail");

    const head = el("div", "dhead");
    const ttl = el("div"); ttl.appendChild(el("div", "dtitle", r.subject + (row.opponent ? ` <span class="opp">vs ${row.opponent}</span>` : "")));
    ttl.appendChild(el("div", "dsub", `${r.date} at ${r.time} · ${humanDur(r.duration)} · ${r.fs} samples/s`));
    head.appendChild(ttl); wrap.appendChild(head);

    const badges = el("div", "badges");
    if (r.tags.tournament) badges.appendChild(el("span", "badge tour", "🏆 Tournament"));
    badges.appendChild(el("span", "badge", `Session <b>${r.tags.session_type}</b>`));
    if (r.n_players === 2) badges.appendChild(el("span", "badge", "2-player match"));
    const q = QUALITY[r.status] || r.status;
    const qb = el("span", "badge q-" + r.status, `Signal <b>${q}</b>` + (r.rail_pct > 0 ? ` · ${r.rail_pct.toFixed(1)}% at limit` : ""));
    badges.appendChild(qb);
    wrap.appendChild(badges);

    // ── the three raw traces ──
    const card = el("div", "card");
    const ch = el("div", "cardhead");
    ch.appendChild(el("span", "clabel", "RAW SIGNAL"));
    ch.appendChild(el("span", "csub", "three traces, no filtering — exactly as recorded, in microvolts (µV)"));
    card.appendChild(ch);

    card.appendChild(el("div", "plotnote",
      `<b style="color:${THEME.rib}">R − L</b> is the difference between the two electrodes — the signal the game reads. ` +
      `A glance to the right moves it up; a glance to the left moves it down. Below it: each electrode alone.`));

    const pp = el("div", "plot");
    const dcv = el("canvas"); pp.appendChild(dcv);
    card.appendChild(pp);
    const stack = el("div", "stack");
    const lrow = el("div", "chrow"), lcv = el("canvas"); lrow.appendChild(lcv);
    const rrow = el("div", "chrow"), rcv = el("canvas"); rrow.appendChild(rcv);
    stack.appendChild(lrow); stack.appendChild(rrow); card.appendChild(stack);

    if (r.events && r.events.length) {
      const key = el("div", "eventkey");
      const kinds = [...new Set(r.events.map(e => e.label))];
      const shown = kinds.slice(0, 6);
      key.appendChild(el("span", null, `<b>${r.events.length}</b> game/cue events marked as vertical lines:`));
      for (const k of shown) key.appendChild(el("span", null, k));
      if (kinds.length > 6) key.appendChild(el("span", null, `+${kinds.length - 6} more`));
      card.appendChild(key);
    }
    wrap.appendChild(card);
    main.appendChild(wrap);

    requestAnimationFrame(() => {
      drawTrace(dcv, r.t, [{ mn: r.diff.mn, mx: r.diff.mx, fill: THEME.ribFill, stroke: THEME.rib }], {
        height: 240, center: true, ruler: true, ceil: r.ceil_uv, events: r.events,
        unit: "µV", name: "R − L (right minus left)", nameColor: THEME.rib,
      });
      drawTrace(lcv, r.t, [{ mn: r.channels.l.mn, mx: r.channels.l.mx, fill: hexA(THEME.L, .12), stroke: THEME.L }],
        { height: 116, center: true, robust: true, ceil: r.ceil_uv, unit: "µV", name: "Left electrode", nameColor: THEME.L });
      drawTrace(rcv, r.t, [{ mn: r.channels.r.mn, mx: r.channels.r.mx, fill: hexA(THEME.R, .12), stroke: THEME.R }],
        { height: 116, center: true, robust: true, ceil: r.ceil_uv, ruler: true, unit: "µV", name: "Right electrode", nameColor: THEME.R });
    });
  }

  // ── dashboard (nothing selected) ────────────────────────────────────────────
  function renderDashboard() {
    const c = state.meta.corpus, q = c.quality || {}, tot = c.n_recordings || 1;
    const main = $("#main"); main.innerHTML = "";
    const d = el("div", "dash");
    d.appendChild(el("h2", null, "Play Pong with your eyes."));
    d.appendChild(el("p", null,
      `Every recording here is horizontal <b>EOG</b> (electrooculography): the small voltage that appears ` +
      `beside your eyes when you glance left or right. Two electrodes read that voltage, and it drives a Pong paddle. ` +
      `This site holds the full recording corpus of the <a href="https://github.com/hoeksemaa/brain-pong">BrainPong</a> project: ` +
      `<b>${c.n_recordings}</b> recordings from <b>${c.n_named_subjects}</b> people ` +
      `plus <b>${c.n_slot_recordings}</b> recorded at unnamed stations, ` +
      `${c.total_hours} hours of signal, collected ${fmtSpan(c.date_start, c.date_end)}. ` +
      `Every name here is a pseudonym.`));
    d.appendChild(el("p", null,
      `<b>Pick a recording from the list on the left.</b> Each one shows three raw traces: the right-minus-left ` +
      `difference (the signal the game reads), and each electrode on its own. ` +
      `Use the filters to narrow the list, or open <b>About</b> for help reading the plots.`));
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
  function fmtSpan(a, b) {
    const MON = ["January","February","March","April","May","June","July","August","September","October","November","December"];
    const f = d => { const [y,m] = d.split("-"); return `${MON[+m-1]} ${y}`; };
    return f(a) === f(b) ? `in ${f(a)}` : `from ${f(a)} to ${f(b)}`;
  }

  // ── about modal ─────────────────────────────────────────────────────────────
  function showAbout() {
    const m = $("#aboutModal");
    m.innerHTML = "";
    const box = el("div", "box");
    box.innerHTML = `<h2>About this site</h2>
      <p><b>BrainPong</b> is a hobby project: play Pong with your eyes. Two electrodes beside the eyes read
      <b>electrooculography</b> (EOG) — the natural voltage of the eye, which shifts when you glance left or right.
      A Cerelog X8 board digitizes the voltage, and the game moves the paddle from it in real time.
      The sensor only receives; nothing is ever applied to a person.</p>
      <p><b>How to read a recording:</b></p>
      <ul>
        <li><b style="color:#7ef0b0">R − L</b> — the right electrode minus the left electrode. This is the signal
        the game reads. A glance right moves the trace up; a glance left moves it down.</li>
        <li><b style="color:#4ea8ff">Left</b> / <b style="color:#ff9d5c">Right</b> — each electrode alone, unfiltered.</li>
        <li><b style="color:#ff6b81">LIMIT</b> — red dashed lines mark the amplifier's range. A trace pinned there
        is clipped, not real signal. Recordings marked <i>clipped</i> or <i>flat</i> had electrode problems.</li>
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
