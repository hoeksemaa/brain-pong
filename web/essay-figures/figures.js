"use strict";

/* Inline essay figures — one per step of the EOG pipeline.
 *
 * Data is the full 2m13s recording, baked to int16 µV arrays by
 * scripts/bake_essay_figures.py — including every filter stage, so a figure
 * shows the same arithmetic the pipeline runs and can never drift from it. A
 * 10 s window scrolls through the recording at SCROLL_RATE seconds of recording
 * per real second, and loops. The browser only interpolates and draws.
 *
 * Each figure gets one control. The control drives the TRANSITION between two
 * real states, never a fraction of the arithmetic — except where the parameter
 * is genuinely continuous (σ multiplier, baseline duration), which is what
 * earns a slider instead of a toggle.
 *
 *   black = base data    green = secondary base    red = what the figure derives
 */

/* Role-based palette, resolved from CSS so the stylesheet stays the one place
 * colours are defined:  ink = base data, green = secondary base, red = derived.
 * axis/grid are neutral so they never read as a data series.
 *
 * The properties are namespaced --eog-* because this ships into a page it does
 * not own, and a bare --red on :root is exactly the kind of thing that collides
 * with a host stylesheet two years later. */
// Set two-line, "look" over the direction — roughly half the width of one line,
// which is what lets the labels survive the fast block's 0.55 s cue spacing.
const CUE_TEXT = { L: "left", R: "right", C: "center" };

const C = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["ink", "green", "red", "axis", "grid", "cue", "rule", "bg",
                   "iris"]) {
    C[k] = cs.getPropertyValue("--eog-" + k).trim();
  }
}

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ease = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;

/* Blend two #rrggbb values. Used where a trace is mid-operation and is neither
   the input nor the output yet, so its colour should not claim to be either. */
const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const mix = (a, b, t) => {
  const [r1, g1, b1] = hex(a), [r2, g2, b2] = hex(b);
  return `rgb(${Math.round(lerp(r1, r2, t))},${Math.round(lerp(g1, g2, t))},` +
         `${Math.round(lerp(b1, b2, t))})`;
};
const font = () => getComputedStyle(document.body).fontFamily;

/* ── plotting ────────────────────────────────────────────────────────────── */

class Plot {
  constructor(ctx, box, xr, yr) {
    this.ctx = ctx;
    this.x = box.x; this.y = box.y; this.w = box.w; this.h = box.h;
    this.xmin = xr[0]; this.xmax = xr[1];
    this.ymin = yr[0]; this.ymax = yr[1];
  }
  px(t) { return this.x + (t - this.xmin) / (this.xmax - this.xmin) * this.w; }
  py(v) { return this.y + this.h - (v - this.ymin) / (this.ymax - this.ymin) * this.h; }

  grid(tk, fmt, unit) {
    const { ctx } = this;
    ctx.save();
    ctx.font = "10px " + font();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const base = ctx.globalAlpha;
    for (const v of tk) {
      // Deliberately NOT snapped to whole pixels. The centre glides
      // continuously, so rounding each line independently makes them jump a
      // pixel at a time at different moments — visibly ticking out of step with
      // each other and with the trace, which is drawn at sub-pixel positions.
      const yy = this.py(v);
      if (yy < this.y || yy > this.y + this.h) continue;
      // Fade near the panel edges so a line entering or leaving the visible
      // range dissolves instead of popping.
      const edge = Math.min(yy - this.y, this.y + this.h - yy);
      ctx.globalAlpha = base * clamp(edge / EDGE_FADE_PX, 0, 1);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(this.x, yy); ctx.lineTo(this.x + this.w, yy); ctx.stroke();
      ctx.fillStyle = C.axis; ctx.fillText(fmt(v), this.x - 9, yy);
    }
    ctx.globalAlpha = base;
    if (unit) { ctx.fillStyle = C.axis; ctx.fillText(unit, this.x - 9, this.y - 8); }
    ctx.restore();
  }

  /* Ticks land on whole seconds of the recording and slide with the window.
     `step` goes finer only where the window is short enough that whole seconds
     would leave a panel with one label on it, or none. */
  xAxis(unit, step = 1) {
    const { ctx } = this;
    const yy = this.y + this.h;
    const dp = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    ctx.save();
    ctx.font = "10px " + font();
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = C.axis;
    const base = ctx.globalAlpha;
    // Range extended a step either side so labels fade in from beyond the
    // plot rather than appearing abruptly at its edge.
    for (let k = Math.ceil(this.xmin / step) - 1; k * step <= this.xmax + step; k += 1) {
      const t = k * step;
      const xx = this.px(t);
      const edge = Math.min(xx - this.x, this.x + this.w - xx);
      ctx.globalAlpha = base * clamp(edge / 18, 0, 1);
      ctx.fillText(t.toFixed(dp), xx, yy + 6);
    }
    ctx.globalAlpha = base;
    ctx.restore();
    if (unit) {
      ctx.save();
      ctx.font = "10px " + font();
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = C.axis;
      ctx.fillText(unit, this.x + this.w / 2, yy + 21);
      ctx.restore();
    }
  }

  /* `get(i)` is indexed by absolute sample number; x comes from i/fs, so the
     window scrolls smoothly instead of stepping one sample at a time. */
  trace(get, i0, i1, fs, color, width) {
    const { ctx } = this;
    ctx.save();
    ctx.beginPath(); ctx.rect(this.x, this.y - 3, this.w, this.h + 6); ctx.clip();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = i0; i < i1; i++) {
      const xx = this.px(i / fs), yy = this.py(get(i));
      if (i === i0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
    ctx.restore();
  }

  /* Trace a function of continuous TIME rather than of sample index, at `steps`
     points per sample of `fs`.

     Only figures 2 and 3 need this, and only because they draw a pre-ADC signal.
     Their interference is analytic, so it can be evaluated anywhere — and it has
     to be: the 5th mains harmonic is 300 Hz against the recording's 125 Hz
     Nyquist, so drawing it on sample positions would render it as a 50 Hz alias.
     Sampling the analytic part finely is not a liberty, it is the whole point of
     being on the analogue side of the hinge. The recorded part of `f` is still
     only known at 250 Hz and is interpolated between samples, which is what
     trace() draws anyway. */
  traceT(f, t0, t1, fs, steps, color, width) {
    const { ctx } = this;
    ctx.save();
    ctx.beginPath(); ctx.rect(this.x, this.y - 3, this.w, this.h + 6); ctx.clip();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
    ctx.beginPath();
    const dt = 1 / (fs * steps);
    let first = true;
    for (let t = t0; t <= t1; t += dt) {
      const xx = this.px(t), yy = this.py(f(t));
      if (first) { ctx.moveTo(xx, yy); first = false; } else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
    ctx.restore();
  }

  label(text, color) {
    const { ctx } = this;
    ctx.save();
    ctx.font = "15px " + font();
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(text, 6, this.y + this.h / 2);
    ctx.restore();
  }

  /* label(), but stacked lines at a caller-chosen size. Exists for names too
     long for the gutter at the standard 15 px — "amplitude · 0.1–30 Hz" would
     cross x=63 into the tick labels, so figure 8.5 stacks the name over its
     band at a size that stays clear. */
  label2(lines, color, px) {
    const { ctx } = this;
    ctx.save();
    ctx.font = px + "px " + font();
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const lh = px + 3;
    const y0 = this.y + this.h / 2 - (lines.length - 1) * lh / 2;
    lines.forEach((t, j) => ctx.fillText(t, 6, y0 + j * lh));
    ctx.restore();
  }

  /* A short readout in the panel's top right — which filters are on, what σ
     came out, how many cues the threshold caught. Reserved for a figure's
     actual output: a number the step produces and the next step consumes, which
     the traces genuinely cannot show. Everything else belongs in the prose.

     Knocked out of the background, because this lands on top of gridlines and
     dashed cue guides and grey-on-grey turns it to mush. */
  note(lines, alpha = 1) {
    const { ctx } = this;
    ctx.save();
    ctx.font = "10px " + font();
    ctx.textAlign = "right"; ctx.textBaseline = "top";
    lines.forEach(([text, color, a], j) => {
      const eff = (a === undefined ? 1 : a) * alpha;
      if (eff <= 0.004 || !text) return;
      const y = this.y + 8 + j * 13;
      ctx.globalAlpha = eff;
      const w = ctx.measureText(text).width;
      ctx.fillStyle = C.bg;
      ctx.fillRect(this.x + this.w - w - 3, y - 2, w + 6, 14);
      ctx.fillStyle = color;
      ctx.fillText(text, this.x + this.w, y);
    });
    ctx.restore();
  }

  /* Gaze-cue guides: where the subject was told to look left, right or back to
     centre. Dashed so they read as annotation rather than as a second axis, and
     spanning the whole panel stack so they hold still through the converge
     animation instead of moving with any one panel.

     Labels are set on two staggered rows. Spelled out they are ~6x wider than a
     single letter, and in the fast block cues land ~1 s apart — one row would
     overlap. Anything that still collides on both rows is dropped rather than
     drawn on top of its neighbour. */
  cues(list, yTop, yBot) {
    const { ctx } = this;
    ctx.save();
    ctx.font = "9.5px " + font();
    ctx.textAlign = "center"; ctx.textBaseline = "top";

    const vis = [];
    for (const c of list) {
      if (c.t < this.xmin - 0.5 || c.t > this.xmax + 0.5) continue;
      const xx = this.px(c.t);
      const edge = Math.min(xx - this.x, this.x + this.w - xx);
      const a = clamp(edge / 18, 0, 1);
      if (a > 0.01) vis.push({ ...c, xx, a });
    }

    // Every cue gets a line — no event is ever unmarked.
    ctx.strokeStyle = C.cue; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    for (const c of vis) {
      ctx.globalAlpha = c.a;
      ctx.beginPath(); ctx.moveTo(c.xx, yTop); ctx.lineTo(c.xx, yBot); ctx.stroke();
    }
    ctx.setLineDash([]);

    // TWO staggered rows, lower row first. A label needs ~46 px and the fast
    // block puts cues 0.55 s apart, which at a 10 s window is ~24 px — so one
    // row can only ever label every other cue, and the return-to-centre of each
    // pair went unlabelled while its dashed line stayed. Alternating rows
    // doubles the spacing available to 48 px, which just fits.
    //
    // Still two-pass by priority: directional cues claim space first, so if
    // anything ever has to be dropped it is the return-to-centre and not the
    // "look left" that says which way the trace is about to move. Every cue
    // keeps its line either way.
    const ROWS = [[yTop - 27, yTop - 16], [yTop - 49, yTop - 38]];
    const taken = [[], []];
    const place = (xx, half) => {
      for (let r = 0; r < ROWS.length; r++) {
        if (taken[r].every(([s, e]) => xx + half < s || xx - half > e)) {
          taken[r].push([xx - half, xx + half]);
          return r;
        }
      }
      return -1;
    };
    ctx.fillStyle = C.cue;
    for (const pass of ["LR", "C"]) {
      for (const c of vis) {
        if ((c.dir === "C") !== (pass === "C")) continue;
        const word = CUE_TEXT[c.dir] || c.dir;
        const w = Math.max(ctx.measureText("look").width, ctx.measureText(word).width);
        const row = place(c.xx, w / 2 + 6);
        if (row < 0) continue;
        ctx.globalAlpha = c.a * 0.85;
        ctx.fillText("look", c.xx, ROWS[row][0]);
        ctx.fillText(word, c.xx, ROWS[row][1]);
      }
    }
    ctx.restore();
  }
}

/* Step ladder, with a 4 rung added to the usual 1/2/5.
 *
 * The gap between 2 and 5 is a factor of 2.5, which is too coarse here: panel
 * spans are fixed by the signal, so a span landing mid-gap has to fall a whole
 * rung and doubles its line count. The 950 µV differential panel is exactly
 * that — 500 won't fit twice readably, and without a 4 it drops to 200 and
 * shows twice as many lines as its neighbours. 4 × 10^k still labels cleanly
 * (0.40 / 0.80 / 1.20). */
const STEP_LADDER = [1, 2, 4, 5, 10];

function finerStep(step) {
  const mag = Math.pow(10, Math.floor(Math.log10(step) + 1e-9));
  const m = step / mag;
  if (m >= 7.5) return 5 * mag;
  if (m >= 4.5) return 4 * mag;
  if (m >= 3) return 2 * mag;
  if (m >= 1.5) return 1 * mag;
  return 0.5 * mag;
}

/* Gridline step for one panel.
 *
 * Chosen so the on-screen SPACING is about the same in every panel regardless
 * of how different their µV scales are — a panel showing 950 µV and one showing
 * 2750 µV should have the same visual rhythm, otherwise the denser one reads as
 * a different kind of chart. Pixel spacing is the invariant; the µV value falls
 * out of it, snapped to the 1/2/5 ladder so labels stay round.
 *
 * Then forced finer if that would leave fewer than `minCount` lines solidly
 * inside the panel. Derived from the span alone, never from the current centre:
 * a per-frame choice would flip spacing as the axis drifts and pop extra lines
 * into the middle of the panel. An interval of width `usable` always holds at
 * least floor(usable / step) multiples of step, whatever the offset. */
function gridStep(span, panelPx, padUV, minCount) {
  const ideal = span * TARGET_GRID_PX / panelPx;
  const mag = Math.pow(10, Math.floor(Math.log10(ideal)));
  let step = mag, err = Infinity;
  for (const m of STEP_LADDER) {
    const e = Math.abs(Math.log((m * mag) / ideal));   // ratio error, not absolute
    if (e < err) { err = e; step = m * mag; }
  }
  const maxStep = (span - 2 * padUV) / minCount;
  let guard = 0;
  while (step > maxStep && guard++ < 8) step = finerStep(step);
  return step;
}

function ticksAt(lo, hi, step) {
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

/* ── figure shell ────────────────────────────────────────────────────────── */

const TRANSITION_S = 1.0;     // how long a toggle takes to complete its morph,
                              // the same in every figure so the steps read as
                              // one sequence rather than a set of one-offs
const SCROLL_RATE = 1;        // seconds of recording per real second
const EDGE_FADE_PX = 14;      // gridlines dissolve within this of a panel edge
const PANEL_H = 150;          // px per signal panel
// One canvas height for every figure, set by the tallest requirement: figure 6
// stacks two 150 px panels plus the gap they converge across. Single-panel
// figures centre their panel in the same box, so every plot in the series sits
// at the same height on the page and the reader's eye does not have to move
// between steps.
const FIG_H = 430;
// Clearance above a panel: the upper of the two staggered cue-label rows sits
// 49 px up, so 60 is that plus a little air. Nothing else needs space up there,
// so this is also where a single-panel figure starts its plot — no figure
// carries headroom it does not use.
const CUE_HEADROOM = 60;
const PLOT_TOP = CUE_HEADROOM;
const PLOT_FOOT = 36;         // px below a panel for the x-axis and its unit
// A figure is exactly as tall as its content. What is standardised is the PLOT
// — same width, same height, same distance from the top of the figure in every
// step — which is what makes the traces comparable between steps. Padding the
// canvas out to figure 6's height on top of that bought nothing but whitespace.
const SINGLE_H = PLOT_TOP + PANEL_H + PLOT_FOOT;
// Gap between stacked panels, taken from figure 6: its two sit at PLOT_TOP and at
// PLOT_TOP + avail − PANEL_H, which leaves exactly this between them.
const PANEL_GAP = FIG_H - PLOT_TOP - PLOT_FOOT - 2 * PANEL_H;
// Figure 2 stacks THREE — the two canthi and the reference they are measured
// against — so it is the one figure taller than figure 6. The plot box is still
// 430x150 and the top two panels sit at exactly figure 3's y, so the seam holds;
// what grows is only the canvas.
const TRIPLE_H = PLOT_TOP + 3 * PANEL_H + 2 * PANEL_GAP + PLOT_FOOT;
const TARGET_GRID_PX = 52;    // aimed-for gridline spacing, identical in every panel
// Two labelled lines are always on screen, so any screenshot at any scroll
// position carries its own absolute scale.
const MIN_GRIDLINES = 2;

class Figure {
  /* opts:
   *   height   canvas height, px
   *   aria     what the control does, for screen readers
   *   control  "toggle" (default) or {slider: [min, max], start, step}
   *   toggles  [{key, label}] — several independent toggles, each carrying its
   *            own printed label. Mutually exclusive with control.
   *   draw(ctx, W, H, value, clock)
   *
   * `value` is the control's position: 0..1 for a toggle (eased toward its
   * target so a flip morphs rather than snaps), the raw slider value, or — for
   * `toggles` — an object of those keyed the same way. */
  constructor(mount, opts) {
    this.opts = opts;
    const sl = opts.control && opts.control.slider;
    const multi = opts.toggles || null;
    this.slider = sl || null;
    this.multi = multi;
    this.value = sl ? (opts.control.start ?? sl[0]) : 0;
    this.target = this.value;   // toggles ease toward this; sliders sit on it
    this.clock = 0;             // seconds of window travel
    this.running = true;

    const fig = document.createElement("div");
    fig.className = "fig";

    // Canvas gets its own box so it can flex while the control sits beside it.
    const box = document.createElement("div");
    box.className = "fig-canvas";
    this.canvas = document.createElement("canvas");
    box.appendChild(this.canvas);
    fig.appendChild(box);

    if (multi) {
      // Each control names itself, so the panel does not have to. A figure with
      // more than one knob cannot label its state on the trace without a legend,
      // and a legend is ink the prose beside it already spends.
      this.values = {};
      this.targets = {};
      const col = document.createElement("div");
      col.className = "fig-toggles";
      for (const s of multi) {
        this.values[s.key] = 0;
        this.targets[s.key] = 0;
        const r = document.createElement("div");
        r.className = "fig-toggle-row";
        const inp = document.createElement("input");
        inp.type = "checkbox";
        inp.checked = false;         // browsers restore state across a reload
        inp.setAttribute("aria-label", s.label);
        inp.addEventListener("input", () => {
          this.targets[s.key] = inp.checked ? 1 : 0;
        });
        const lab = document.createElement("span");
        lab.className = "fig-toggle-label";
        lab.textContent = s.label;
        r.appendChild(inp); r.appendChild(lab);
        col.appendChild(r);
      }
      fig.appendChild(col);
      mount.appendChild(fig);
      this.ctx = this.canvas.getContext("2d");
      this.finish(mount);
      return;
    }

    // Some figures have nothing to control: where the loop itself is the
    // animation, a toggle would be a control that does nothing, and a reader
    // reasonably spends time wondering what it is for.
    if (opts.control === "none") {
      // An empty column of the same width as everyone else's controls, so this
      // figure's plot is exactly as wide as theirs.
      const pad = document.createElement("div");
      pad.className = "fig-nocontrol";
      fig.appendChild(pad);
      mount.appendChild(fig);
      this.ctx = this.canvas.getContext("2d");
      this.finish(mount);
      return;
    }

    const below = sl && opts.control.below;
    const row = document.createElement("div");
    row.className = sl ? (below ? "fig-slider-below" : "fig-slider") : "fig-toggle";
    this.input = document.createElement("input");
    if (sl) {
      // Vertical by default, so it sits level with the panel it acts on and
      // reads as a magnitude rather than as a timeline running against the
      // x-axis. `below` lays it horizontally under the plot instead, for a
      // parameter that IS a level on the y-axis — there it can be aligned to
      // the plot box and read as the line it moves.
      this.input.type = "range";
      this.input.min = sl[0]; this.input.max = sl[1];
      this.input.step = opts.control.step ?? (sl[1] - sl[0]) / 200;
      this.input.value = this.value;
      if (!below) this.input.setAttribute("orient", "vertical");
      if (below && opts.control.label) {
        this.readout = document.createElement("span");
        this.readout.className = "fig-slider-readout";
        this.readout.textContent = opts.control.label(this.value);
      }
    } else {
      this.input.type = "checkbox";
      // Explicit, because browsers restore form-control state across a reload:
      // the toggle would come back checked while the canvas redrew from value 0,
      // leaving the control saying one thing and the figure showing another.
      this.input.checked = false;
    }
    this.input.setAttribute("aria-label", opts.aria);
    row.appendChild(this.input);
    if (this.readout) row.appendChild(this.readout);
    fig.appendChild(row);
    if (below) fig.classList.add("fig-stacked");
    mount.appendChild(fig);

    this.ctx = this.canvas.getContext("2d");
    this.input.addEventListener("input", () => {
      this.target = sl ? parseFloat(this.input.value) : (this.input.checked ? 1 : 0);
      if (sl) this.value = this.target;   // a slider is where you put it, at once
      if (this.readout) this.readout.textContent = opts.control.label(this.value);
    });
    this.finish(mount);
  }

  /* Sizing, visibility and the animation loop — shared by both control shapes. */
  finish() {

    this.resize = this.resize.bind(this);
    window.addEventListener("resize", this.resize);
    this.resize();

    // Don't burn frames on a figure that isn't on screen.
    new IntersectionObserver((es) => {
      this.running = es[0].isIntersecting;
      if (this.running) { this.last = performance.now(); this.tick(); }
    }, { threshold: 0 }).observe(this.canvas);

    this.last = performance.now();
    this.tick = this.tick.bind(this);
    requestAnimationFrame(this.tick);
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth || 700, h = this.opts.height;
    this.canvas.style.height = h + "px";
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = h;
    this.draw();
  }

  tick(now) {
    if (!this.running) return;
    now = now || performance.now();
    const dt = Math.min(0.1, (now - this.last) / 1000);   // clamp tab-switch jumps
    this.last = now;
    this.clock += dt * SCROLL_RATE;

    // Ease toward whatever the toggle last asked for. Flipping mid-morph just
    // reverses from where it is, so it never snaps. Independent toggles each
    // ease on the same clock, so two flipped together stay in step.
    const step = dt / (this.opts.morph_s || TRANSITION_S);
    const toward = (v, t) => (t > v ? Math.min(t, v + step) : Math.max(t, v - step));
    if (this.multi) {
      for (const k in this.values) {
        if (this.values[k] !== this.targets[k]) {
          this.values[k] = toward(this.values[k], this.targets[k]);
        }
      }
    } else if (!this.slider && this.value !== this.target) {
      this.value = toward(this.value, this.target);
    }
    this.draw();
    requestAnimationFrame(this.tick);
  }

  draw() {
    readColors();
    this.ctx.clearRect(0, 0, this.W, this.H);
    this.opts.draw(this.ctx, this.W, this.H,
                   this.multi ? this.values : this.value, this.clock);
  }
}

/* ── the data ────────────────────────────────────────────────────────────── */

/* One decoded recording, shared by every figure on the page.
 *
 * `sig(key)` returns a sample accessor for any baked signal, indexed by
 * absolute sample number, in µV. `step(key)` is that signal's gridline step,
 * fixed for the life of the figure — see gridStep(). */
class Data {
  constructor(meta, buf) {
    this.meta = meta;
    this.fs = meta.fs;
    this.n = meta.n;
    this.view = meta.view_s;
    this.arr = {};
    meta.layout.forEach((k, i) => {
      this.arr[k] = new Int16Array(buf, i * meta.n * 2, meta.n);
    });
    this._step = {};
  }

  sig(key) {
    // The differential is the one signal not stored: it is ch_R − ch_L, and
    // reconstructing it in the browser costs one subtraction per sample.
    if (key === "diff") {
      const R = this.arr.ch_R, L = this.arr.ch_L, m = this.meta.mean.diff;
      return (i) => R[i] - L[i] + m;
    }
    const a = this.arr[key], m = this.meta.mean[key] || 0;
    return (i) => a[i] + m;
  }

  span(key) { return this.meta.span[key]; }

  step(key, span) {
    span = span ?? this.meta.span[key];
    const ck = key + ":" + span;
    // Half the fade band, not all of it: a line at ~50% alpha is still legible
    // enough to read scale off a screenshot, and demanding two FULLY solid
    // lines forces a finer step than the figure needs.
    if (!(ck in this._step)) {
      this._step[ck] = gridStep(span, PANEL_H, span * (EDGE_FADE_PX / 2) / PANEL_H,
                                MIN_GRIDLINES);
    }
    return this._step[ck];
  }

  /* Window bounds for a scroll clock, in absolute sample indices.
   *
   * `from` skips the head of the record where a figure shows a filtered stage:
   * the offline high-pass opens cold on the 28 mV DC step and rings down
   * through the first second, which never happens live. */
  window(clock, from = 0, view = this.view) {
    const travel = this.meta.duration_s - from - view;
    const t0 = from + (clock % travel);
    return {
      t0,
      i0: Math.max(0, Math.floor(t0 * this.fs) - 1),
      i1: Math.min(this.n, Math.ceil((t0 + view) * this.fs) + 1),
    };
  }
}

const winMean = (get, i0, i1) => {
  let a = 0;
  for (let i = i0; i < i1; i++) a += get(i);
  return a / (i1 - i0);
};

/* ── the pre-ADC interference (figures 2 and 3) ───────────────────────────── */

/* The common-mode waveform both pre-ADC figures scale, built once from
 * meta.interference so the two cannot drift apart. Unit peak by construction —
 * the bake normalises the harmonic amplitudes — which is what lets V_cm be quoted
 * as a peak and stay inside the amplifier's input window.
 *
 * Mains is not a sine. It carries odd-harmonic distortion, and capacitive
 * coupling amplifies the problem because the displacement current into the body
 * is C·dV/dt and so rises with frequency.
 *
 * No amplitude envelope, though the real interference has one. It is measurable
 * in this recording — the 58-62 Hz Hilbert envelope has sd/mean 0.22 on R and
 * 0.42 on L, power peaking at 0.061 Hz — but only a fifth of its variance sits
 * below 0.3 Hz, so the real thing jumps when someone moves and then holds. A
 * single slow sinusoid got the depth right and the character wrong.
 *
 * Everything NOT here — electrode-skin noise at 1-20 µV rms with a 1/f^1.5-2
 * spectrum, EMG, drift, lead-loop pickup — is already in the recording. Adding it
 * would count it twice. These figures model only what gets removed. */
function makeInterference(meta) {
  const H = meta.interference.harmonics.map(
    ([k, a, p]) => [2 * Math.PI * k * meta.interference.f_hz, a, p]);
  return (t) => {
    let v = 0;
    for (const [w, a, p] of H) v += a * Math.sin(w * t + p);
    return v;
  };
}

/* Read a baked signal at a non-integer sample position. The recorded half of a
 * pre-ADC figure is only known at 250 Hz; between samples it is a straight line,
 * which is exactly what Plot.trace draws. */
const sampleAt = (get, n) => (t, fs) => {
  const x = t * fs;
  const i = Math.floor(x);
  if (i < 0) return get(0);
  if (i >= n - 1) return get(n - 1);
  const fr = x - i;
  return get(i) * (1 - fr) + get(i + 1) * fr;
};

/* Every figure, keyed by the id of the element it mounts into. Only the ones
   present in the DOM are built, so the essay can embed them one at a time. */
const FIGURES = {};

/* ── 01 · eyes move — the dipole rotates ─────────────────────────────────── */

/* Where the signal comes from, and the only figure in the series that is a diagram
 * rather than a trace.
 *
 * THE EYE IS A BATTERY. The retinal pigment epithelium holds a standing potential
 * across the globe — cornea POSITIVE, retina negative, a few hundred µV to a
 * millivolt end to end. It is always there; nothing about a saccade creates it. All
 * a saccade does is ROTATE it.
 *
 * WHICH IS WHY THE SIGNS ONLY APPEAR AS THE EYES TURN. Looking straight at the
 * viewer the dipole points along the line of sight, so both poles project to the
 * same spot on screen and both canthi sit equidistant from it — the battery is
 * there and the difference it produces is zero. Rotating it swings the positive
 * pole toward one canthus and the negative pole toward the other, and only then is
 * there anything to measure. The signs fading in with the rotation is that, not a
 * charge appearing out of nothing.
 *
 * VERIFIED AGAINST THE RECORDING, which is the one thing here that is not a
 * drawing. Averaged over the cued glances: a LEFT cue puts ch_L at +39.0 µV and
 * ch_R at −25.4 µV; a RIGHT cue puts ch_R at +29.6 µV and ch_L at −34.2 µV. So
 * R − L is −64 µV one way and +64 µV the other, symmetric to within a microvolt,
 * and the canthus the cornea swings toward is the one that goes positive. The
 * geometry drawn here predicts that sign, and the data confirms it.
 *
 * HANDEDNESS, and it is worth being explicit because it is easy to get backwards.
 * These are eyes looking OUT at the reader, so screen-left is the subject's RIGHT.
 * GAZE_DIR = −1 sweeps the pupils to screen-left, which is the subject glancing to
 * their own right, and by the numbers above that drives R − L POSITIVE. Flip
 * GAZE_DIR to +1 and it becomes a leftward glance, matching the "look left" cue
 * guides in figures 2 onward. No electrodes are drawn: this figure is the source,
 * and where the signal gets measured from is figure 2's business.
 *
 * NO AXIS LINE BETWEEN THE POLES. It was drawn and cut: through a globe that is
 * itself outlined in grey, a third horizontal line reads as anatomy the eye does
 * not have. The two signs moving apart in step already say they are one object.
 *
 * THE SIGNS SIT OUTSIDE THE GLOBES, which is a legibility choice and not a
 * projection. The true poles stay inside the outline — at 30° they are only
 * Rg·sin 30° = 22 px off centre against a 44 px radius — so drawn faithfully they
 * would sit on top of the iris. They are pushed out to the rim along the same axis
 * and in the same proportion, so the direction and the ordering are honest even
 * though the distance is not. */
FIGURES["fig-01"] = (mount, data) => {
  const R_GLOBE = 40;          // px, the eyeball
  const R_IRIS = 17;
  const R_PUPIL = 9;
  // Wider than anatomy — real eyes sit ~2.6 globe-radii apart and this is 4.2 —
  // because each globe needs a pole drawn either side of it and at true spacing the
  // left eye's minus lands on top of the right eye's plus. A schematic, not a
  // portrait, so the spacing gives way before the poles do.
  const EYE_DX = 84;           // half the distance between the two globes
  const MAX_DEG = 30;          // gaze excursion at full turn
  const SIGN_OUT = 13;         // pole clearance from the globe
  const GAZE_DIR = -1;         // −1 sweeps the pupils to screen-left; see above
  const MAX_RAD = MAX_DEG * Math.PI / 180;

  new Figure(mount, {
    height: 210,
    aria: "turn the eyes and watch the corneoretinal dipole rotate",
    draw: (ctx, W, H, s) => {
      const e = ease(s);
      const phi = GAZE_DIR * MAX_RAD * e;
      // Rotation as a fraction of full turn, signed. Drives where the poles sit.
      const t = Math.sin(phi) / Math.sin(MAX_RAD);
      // Signs come in with the rotation, because before it there is no difference
      // for the electrodes to see.
      const aSign = ease(s / 0.35);
      const cy = H / 2 - 6;

      const eye = (cx) => {
        // Globe.
        ctx.save();
        ctx.fillStyle = C.bg; ctx.strokeStyle = C.cue; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(cx, cy, R_GLOBE, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.restore();

        // Iris and pupil, both foreshortened by cos of the rotation and both
        // centred on the cornea — the positive pole itself, at its true projected
        // position. Clipped to the globe so a wide rotation cannot push them past
        // the outline.
        const xCornea = cx + R_GLOBE * Math.sin(phi);
        const squash = Math.abs(Math.cos(phi));
        ctx.save();
        ctx.beginPath(); ctx.arc(cx, cy, R_GLOBE - 1, 0, Math.PI * 2); ctx.clip();
        for (const [r, fill] of [[R_IRIS, C.iris], [R_PUPIL, C.ink]]) {
          ctx.fillStyle = fill;
          ctx.beginPath();
          ctx.ellipse(xCornea, cy, r * squash, r, 0, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();

        if (aSign <= 0.004) return;
        const xPlus = cx + (R_GLOBE + SIGN_OUT) * t;
        const xMinus = cx - (R_GLOBE + SIGN_OUT) * t;

        ctx.save();
        ctx.font = "600 21px " + font();
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillStyle = C.red;
        ctx.globalAlpha = aSign;
        ctx.fillText("+", xPlus, cy);
        // Dimmer: the retina is on the far side of the globe.
        ctx.globalAlpha = aSign * 0.75;
        ctx.fillText("−", xMinus, cy);
        ctx.restore();
      };

      eye(W / 2 - EYE_DX);
      eye(W / 2 + EYE_DX);
    },
  });
};

/* ── 02 · measured against SRB1 ──────────────────────────────────────────── */

/* The step that makes a measurement exist at all.
 *
 * There is no such thing as the voltage AT a point — only the voltage BETWEEN
 * two. Every channel on this rig is one canthus measured against the reference
 * earlobe, and this figure is that subtraction. Before it, each amplifier pin
 * sits at whatever the body sits at; after it, there is a signal.
 *
 * Same invented open-loop scenario as figure 3, sharing every parameter, so the
 * arithmetic closes exactly:
 *
 *     V_pin_i = V_cm·(1 − Z_i/Z_in)    + true_i
 *     V_srb1  = V_cm·(1 − Z_srb1/Z_in)
 *     V_pin_i − V_srb1 = V_cm·(Z_srb1 − Z_i)/Z_in + true_i
 *
 * — and that last line IS figure 3's opening trace. So this figure opens on
 * `figure 3's opening trace + V_srb1` and the toggle removes V_srb1, landing on
 * figure 3 with nothing left over.
 *
 * THREE PANELS, figure 6's structure: the operands are drawn, then the result.
 * R and L are the two canthi as their amplifier pins see them; SRB1 is the
 * reference earlobe, drawn as its own trace because a subtraction whose subtrahend
 * is invisible is not a subtraction the reader can follow. It is the one figure in
 * the series taller than figure 6 — three 150 px panels instead of two — and the
 * top two sit at exactly figure 3's y, so the seam holds and only the canvas grows.
 *
 * SRB1 IS DRAWN IN RED, with the results it produces. Red is this series' colour
 * for what a step contributes, and the subtrahend and the two differences it makes
 * are one operation — so the reference, its two travelling copies and both output
 * traces are all the same colour. It also carries no biopotential and no drift,
 * only the common-mode, which is exactly why subtracting it takes the one and
 * leaves the other.
 *
 * THE REFERENCE SITS BETWEEN THE TWO CHANNELS, and doubles. One earlobe is
 * subtracted from two canthi, so its trace leaves the middle panel as TWO copies
 * that slide out in opposite directions — up into R, down into L — and dissolve as
 * they land. Putting it in the middle is what makes "at the same time" literal:
 * the distances are equal, so one progress variable gives both copies the same
 * speed AND the same arrival. Everything they overlay is at the same volts scale on
 * the way, and at that scale a channel and the reference agree to 0.013 %, so each
 * copy lands directly on top of the trace it is cancelling.
 *
 * L THEN CLOSES THE GAP. Once the reference is spent its panel is dead space, so L
 * rises into it and the figure finishes as the two panels figure 5 opens on, in
 * figure 5's own positions. Figure 6 converges two panels into one for the same
 * reason: a panel that has done its work should not go on occupying the page.
 *
 * TWO STAGES ON ONE CLICK, because "it all goes away" and "no, look closer, it
 * didn't" are two different claims and the second is the interesting one.
 *
 *   stage 1  subtract the reference. Its copies slide out into the two channels,
 *            its own panel empties and L closes the gap, and the axis HOLDS at the
 *            volts scale — so both channels collapse from a 3 V band of mains to a
 *            hairline on zero.
 *   stage 2  the axis zooms ~2000x, and the hairline turns out to have structure
 *            — the 392 µV and 168 µV that survived because the bridge is
 *            unbalanced. Subtraction gets you to a floor; it does not get you to
 *            zero. Figure 3 then lowers the floor.
 *
 * The zoom is LOGARITHMIC. A linear ramp from 3.73 V to 1550 µV spends 99 % of
 * its second above 40 mV and then snaps through the last three decades; equal
 * ratios per unit time is what reads as a steady zoom.
 *
 * AT THE VOLTS SCALE ALL THREE TRACES ARE INDISTINGUISHABLE. The channels differ
 * from the reference by ~400 µV out of 3 V — 0.013 % — so the figure opens on
 * three panels that look like the same trace, and that trace looks like mains.
 * That is the point: the eye signal is in there, four decades down, and nothing
 * about the raw pin potentials hints at it.
 *
 * THE NAMES CHANGE WITH THE ARITHMETIC, as figure 6's do when its R and L become
 * R − L: R and L are what went in, R−SRB1 and L−SRB1 are what came out, switching
 * on the subtraction itself so the name and the contents change together.
 *
 * NAMES AT THE SERIES' 15 px, like every other figure. "R−SRB1" measures to x=60
 * against tick labels that start at x=63, so it fits where a longer form would not:
 * the tick gutter is 108 px with the labels right-aligned into it, and anything
 * past x=63 lands on top of "−12.50".
 *
 * THE HALF-CELLS ARE NOT SPOOFED. V_srb1 carries the earlobe's own half-cell
 * potential too, but the recording's DC offsets — −11.1 mV on R, −38.9 mV on L —
 * already ARE (electrode half-cell − earlobe half-cell). Adding the earlobe's
 * offset to both channels and subtracting it again is a no-op, so what is
 * subtracted here is the common-mode alone. */
FIGURES["fig-02"] = (mount, data) => {
  const meta = data.meta, fs = data.fs;
  const sr = meta.srb1, bi = meta.bias, dg = meta.digitize;
  const R = data.sig("ch_R"), L = data.sig("ch_L");

  const VIEW = dg.view_s;                     // 3 s — the series' window
  const X_STEP = [1, 0.5, 0.2, 0.1].find((v) => VIEW / v >= 6) ?? 0.1;
  const CM = makeInterference(meta);          // unit-peak, shared with figure 3
  const SUB = meta.interference.substeps;
  const atR = sampleAt(R, data.n), atL = sampleAt(L, data.n);

  const A_CM = sr.subtracted_uv;              // the reference's own potential, µV
  // What survives the subtraction on each channel, and so what figure 3 inherits.
  const LEAK = { ch_R: bi.ch.ch_R.removed_uv, ch_L: bi.ch.ch_L.removed_uv };
  const SPAN_PIN = sr.span;                   // volts — before subtraction
  const SPAN_REFD = bi.span;                  // µV — figure 3's opening panel
  const gpad = (span) => span * (EDGE_FADE_PX / 2) / PANEL_H;
  // Above this span the panel is labelled in volts. One relabel, and it lands
  // mid-zoom where the gridlines are sweeping anyway, so it is masked by motion.
  const V_SCALE_UV = 1e5;

  new Figure(mount, {
    height: TRIPLE_H,
    morph_s: TRANSITION_S * 2,                // a second per stage
    aria: "subtract the reference electrode from both channels, "
        + "then zoom in on what is left",
    draw: (ctx, W, H, s, clock) => {
      const ph = PANEL_H;
      // R top, the reference BETWEEN them, L bottom — so the two copies travel
      // equal distances in opposite directions and "at the same time" needs no
      // qualification. yR and yMid are figure 5's two panel positions, which is
      // where R and L end up once L has closed the gap.
      const yR = PLOT_TOP;
      const yMid = PLOT_TOP + PANEL_H + PANEL_GAP;
      const yL0 = yMid + PANEL_H + PANEL_GAP;
      const { t0, i0, i1 } = data.window(clock, 0, VIEW);

      const k = ease(s / 0.45);               // the subtraction
      const m = ease(s / 0.40);               // the copies' flight
      // They dissolve from early in the flight, so they read as merging into the
      // trace they cancel rather than stopping on top of it.
      const aCopy = 0.55 * (1 - ease((s - 0.10) / 0.28));
      // The reference's panel empties as its trace departs: once subtracted it is
      // spent, and drawing it still would claim it is an input.
      const aMid = 1 - ease((s - 0.32) / 0.16);
      // L closes the gap the reference leaves, landing on figure 5's lower panel.
      const conv = ease((s - 0.50) / 0.22);
      const z = ease((s - 0.72) / 0.28);      // volts → mV
      const yL = lerp(yL0, yMid, conv);

      // (1−k) of the reference still present, so k=1 is figure 3's opening trace
      // and every intermediate k is a real partial referencing.
      const fn = (at, key) => {
        const a = LEAK[key] + (1 - k) * A_CM;
        return (t) => at(t, fs) + a * CM(t);
      };
      const fR = fn(atR, "ch_R"), fL = fn(atL, "ch_L");
      const fSR = (t) => A_CM * CM(t);
      // Sample-indexed versions, for the window mean the camera centres on.
      const gR = (i) => R(i) + (LEAK.ch_R + (1 - k) * A_CM) * CM(i / fs);
      const gL = (i) => L(i) + (LEAK.ch_L + (1 - k) * A_CM) * CM(i / fs);
      const gSR = (i) => A_CM * CM(i / fs);

      // Equal ratios per unit time — see the note above. The reference's own panel
      // never zooms: it is gone long before the zoom starts.
      const span = (key) => key === "srb1" ? SPAN_PIN.srb1
        : Math.exp(lerp(Math.log(SPAN_PIN[key]), Math.log(SPAN_REFD[key]), z));

      const mk = (y, ctr, sp) => new Plot(ctx,
        { x: 108, y, w: W - 120, h: ph },
        [t0, t0 + VIEW], [ctr - sp / 2, ctr + sp / 2]);

      // Cues span the stack down to whichever panel is currently lowest.
      mk(yR, 0, 1).cues(meta.cues, yR, yL + ph);

      const panel = (y, key, g, f, base, nameIn, nameOut, alpha) => {
        if (alpha <= 0.004) return;
        const sp = span(key);
        const p = mk(y, winMean(g, i0, i1), sp);
        const volts = sp > V_SCALE_UV;
        ctx.globalAlpha = alpha;
        p.grid(ticksAt(p.ymin, p.ymax, gridStep(sp, ph, gpad(sp), MIN_GRIDLINES)),
               volts ? (u) => (u / 1e6).toFixed(2) : (u) => (u / 1e3).toFixed(2),
               volts ? "V" : "mV");
        const col = mix(base, C.red, ease(s));
        if (nameOut) {
          // NOT cross-faded. The two names share an origin at x=6 and have
          // different lengths, so any overlap in their alphas renders both sets
          // of glyphs on top of each other. One leaves before the
          // other arrives, with a beat of no name in between.
          ctx.globalAlpha = alpha * (1 - ease(k / 0.45));
          p.label(nameIn, col);
          ctx.globalAlpha = alpha * ease((k - 0.55) / 0.45);
          p.label(nameOut, col);
        } else {
          p.label(nameIn, col);
        }
        ctx.globalAlpha = alpha;
        if (f) p.traceT(f, t0, t0 + VIEW, fs, SUB, col, 1.2);
        ctx.globalAlpha = 1;
      };
      panel(yR, "ch_R", gR, fR, C.ink, "R", "R−SRB1", 1);
      panel(yL, "ch_L", gL, fL, C.green, "L", "L−SRB1", 1);
      // Grid and name only — the reference's trace leaves as the two copies below.
      // Red, because it is what this step contributes: the subtrahend and the two
      // results it produces are one operation, and in this series that is red.
      panel(yMid, "srb1", gSR, null, C.red, "SRB1", null, aMid);

      // The reference, doubled, sliding into both channels at once — up into R and
      // down into L, equal distances from the middle. Drawn last so each copy is
      // visible over the trace it is landing on.
      if (aCopy > 0.004) {
        const ctr = winMean(gSR, i0, i1);
        for (const yTo of [yR, yL0]) {
          ctx.globalAlpha = aCopy;
          mk(lerp(yMid, yTo, m), ctr, SPAN_PIN.srb1)
            .traceT(fSR, t0, t0 + VIEW, fs, SUB, C.red, 0.9);
          ctx.globalAlpha = 1;
        }
      }

      // Pinned under the lowest panel, which rides up with L.
      mk(yL, 0, 1).xAxis("seconds", X_STEP);
    },
  });
};

/* ── 03 · bias driven into the body ──────────────────────────────────────── */

/* Active cancellation, and the only figure in the series built on invented
 * numbers.
 *
 * WHY IT HAS TO BE INVENTED. The recording only ever contains SRB1-referenced,
 * bias-ON channels; the corpus has no PD_BIAS-off session, so the common-mode the
 * loop suppresses was never measured. Everything from figure 5 down is the
 * committed recording and this is not. What makes it honest anyway is that the
 * model is an INVERTIBLE function of the real data: the figure adds a leak term
 * and the toggle takes it back out, so at s=1 the trace is the recording bit for
 * bit. meta.bias carries every parameter, flagged `invented: true`.
 *
 * THE MECHANISM, which is not what most people expect. Bias does not subtract
 * anything from the measurement. Electrode impedances and the amplifier's input
 * impedance form a BRIDGE, and common-mode leaks into the difference in
 * proportion to how unbalanced that bridge is:
 *
 *     ch_i = ch_i_true + V_cm · (Z_srb1 − Z_i) / Z_in
 *
 * The loop drives an inverted copy of V_cm back into the BODY, which shrinks the
 * multiplier on that whole term. Sound-cancelling headphones, except the
 * anti-signal goes into the medium every microphone is sitting in rather than
 * into one pair of ears — which is exactly why one electrode on one earlobe helps
 * both channels at once. Not finite CMRR: the ADS1299's is −110 dB, far too good
 * to explain any visible hum.
 *
 * Z_in IS NOT THE CHIP. Its input impedance is gigaohms and irrelevant at 60 Hz.
 * The bottom of the bridge is the capacitance from each input node to ground,
 * dominated by unshielded electrode leads — 100 pF is 26.5 MΩ at 60 Hz.
 *
 * WHAT THIS FIGURE DOES NOT CLAIM. That the 60 Hz left in the recording is what
 * bias left behind. With the loop closed this model leaks 1.7–4 µV, and the
 * measured hum is several times that, 4× larger on R than on L, and correlated
 * between the channels at only +0.24 — none of which looks like a shared
 * common-mode. Lead-loop magnetic pickup is differential by cable geometry and
 * immune to both bias and CMRR. So the figure shows what bias REMOVES and says
 * nothing about the composition of what remains.
 *
 * THREE PANELS, LIKE FIGURE 2, and for the same reason: the anti-signal the loop
 * drives into the body sits BETWEEN the two channels and doubles, sliding up into R
 * and down into L over equal distances. Drawn in red with the results it produces,
 * since the anti-signal and the two suppressions it causes are one operation. Once
 * it is spent L closes the gap, and the figure finishes as figure 5's two panels in
 * figure 5's positions.
 *
 * ONE LOOP CANNOT NULL TWO CHANNELS EXACTLY. The two canthi sit behind different
 * mismatches, so the amount cancelled differs — 392 µV on R against 168 µV on L.
 * The middle panel is drawn at the mean of the two, and the ARITHMETIC applied to
 * each channel stays per-channel exact; only the display trace is averaged. What
 * that per-channel difference leaves behind is the (Z_L − Z_R) term figure 6 spends.
 *
 * SEAM. At s=1 this is figure 5's opening frame — same two panels, same baked
 * 3 s spans, same window-mean centring, same 1 s-per-second scroll. The toggle
 * runs BACKWARDS relative to the other figures for once: 0 is the loop open,
 * which is the state the reader arrives from, and 1 is the real recording. Black
 * and green go in, red comes out, on the same s as the arithmetic — figure 7's
 * contract. */
FIGURES["fig-03"] = (mount, data) => {
  const meta = data.meta, fs = data.fs;
  const bi = meta.bias, dg = meta.digitize;
  const R = data.sig("ch_R"), L = data.sig("ch_L");

  const VIEW = dg.view_s;                // 3 s — figure 5's window
  const X_STEP = [1, 0.5, 0.2, 0.1].find((v) => VIEW / v >= 6) ?? 0.1;
  const CM = makeInterference(meta);     // unit-peak, the same one figure 2 scales
  const SUB = meta.interference.substeps;
  const atR = sampleAt(R, data.n), atL = sampleAt(L, data.n);

  // The amplitude the loop removes from each channel, and the panel each needs
  // with the loop open. Both baked; see bias_model() for the derivation.
  const AMP = { ch_R: bi.ch.ch_R.removed_uv, ch_L: bi.ch.ch_L.removed_uv };
  const MID = bi.mid_uv;                 // the anti-signal, as the middle draws it
  const SPAN_OPEN = bi.span;             // loop open — the wider panel
  const SPAN_SHUT = dg.span;             // loop closed — figure 5's panel, the seam
  const gpad = (span) => span * (EDGE_FADE_PX / 2) / PANEL_H;

  new Figure(mount, {
    height: TRIPLE_H,
    morph_s: TRANSITION_S * 2,
    aria: "close the bias loop and drive the mains interference out of the body",
    draw: (ctx, W, H, s, clock) => {
      const ph = PANEL_H;
      // Same stack as figure 2: R top, the thing being subtracted BETWEEN them, L
      // bottom. yR and yMid are figure 5's two positions, which is where R and L
      // finish once L has closed the gap the anti-signal leaves.
      const yR = PLOT_TOP;
      const yMid = PLOT_TOP + PANEL_H + PANEL_GAP;
      const yL0 = yMid + PANEL_H + PANEL_GAP;
      const yf = (u) => (u / 1000).toFixed(2);
      const { t0, i0, i1 } = data.window(clock, 0, VIEW);

      const k = ease(s / 0.45);          // the suppression
      const m = ease(s / 0.40);          // the copies' flight
      const aCopy = 0.55 * (1 - ease((s - 0.10) / 0.28));
      const aMid = 1 - ease((s - 0.32) / 0.16);
      const conv = ease((s - 0.50) / 0.30);
      const yL = lerp(yL0, yMid, conv);

      // (1−k) of the leak, so k=1 is the recording untouched and every intermediate
      // k is a loop with less than full gain — a real state, not a crossfade
      // between two pictures. Time is absolute recording time, so the interference
      // travels with the samples it corrupts instead of standing still while they
      // slide underneath.
      const fn = (at, key) => {
        const a = (1 - k) * AMP[key];
        return (t) => at(t, fs) + a * CM(t);
      };
      const fR = fn(atR, "ch_R"), fL = fn(atL, "ch_L");
      // The anti-signal itself: an inverted copy of the interference, which is what
      // the loop drives into the body. Sound-cancelling headphones, except the
      // anti-signal goes into the medium every electrode is sitting in.
      const fMid = (t) => -MID * CM(t);
      // Sample-indexed versions, for the window mean the camera centres on.
      const gR = (i) => R(i) + (1 - k) * AMP.ch_R * CM(i / fs);
      const gL = (i) => L(i) + (1 - k) * AMP.ch_L * CM(i / fs);
      const gMid = (i) => -MID * CM(i / fs);

      const span = (key) => key === "mid" ? SPAN_OPEN.mid
        : lerp(SPAN_OPEN[key], SPAN_SHUT[key], k);

      const mk = (y, ctr, sp) => new Plot(ctx,
        { x: 108, y, w: W - 120, h: ph },
        [t0, t0 + VIEW], [ctr - sp / 2, ctr + sp / 2]);

      mk(yR, 0, 1).cues(meta.cues, yR, yL + ph);

      const panel = (y, key, g, f, base, nameIn, nameOut, alpha) => {
        if (alpha <= 0.004) return;
        const sp = span(key);
        const p = mk(y, winMean(g, i0, i1), sp);
        ctx.globalAlpha = alpha;
        p.grid(ticksAt(p.ymin, p.ymax, gridStep(sp, ph, gpad(sp), MIN_GRIDLINES)),
               yf, "mV");
        // Black/green in, red out, carrying the same k as the arithmetic.
        const col = mix(base, C.red, k);
        if (nameOut) {
          // Not cross-faded — see figure 2. One name leaves before the next lands.
          ctx.globalAlpha = alpha * (1 - ease(k / 0.45));
          p.label(nameIn, col);
          ctx.globalAlpha = alpha * ease((k - 0.55) / 0.45);
          p.label(nameOut, col);
        } else {
          p.label(nameIn, col);
        }
        ctx.globalAlpha = alpha;
        if (f) p.traceT(f, t0, t0 + VIEW, fs, SUB, col, 1.2);
        ctx.globalAlpha = 1;
      };
      // Tight around the minus, not "R − Bias": the tick gutter stops a name at
      // x=63, and spaced it measures to 78 at the series' 15 px while tight it
      // measures to 60. Figure 2 sets its longer names smaller for the same reason;
      // here the shorter name means the size can stay standard instead.
      panel(yR, "ch_R", gR, fR, C.ink, "R", "R−Bias", 1);
      panel(yL, "ch_L", gL, fL, C.green, "L", "L−Bias", 1);
      // Grid and name only; its trace leaves as the two copies below. Red, because
      // the anti-signal and the two results it produces are one operation.
      panel(yMid, "mid", gMid, null, C.red, "Bias", null, aMid);

      // The anti-signal, doubled, sliding into both channels at once — up into R
      // and down into L, equal distances from the middle.
      if (aCopy > 0.004) {
        const ctr = winMean(gMid, i0, i1);
        for (const yTo of [yR, yL0]) {
          ctx.globalAlpha = aCopy;
          mk(lerp(yMid, yTo, m), ctr, SPAN_OPEN.mid)
            .traceT(fMid, t0, t0 + VIEW, fs, SUB, C.red, 0.9);
          ctx.globalAlpha = 1;
        }
      }

      mk(yL, 0, 1).xAxis("seconds", X_STEP);
    },
  });
};

/* ── 05 · digitized ──────────────────────────────────────────────────────── */

/* The ADC: the hinge of the whole essay. Everything above this figure is
 * analogue physics reconstructed backwards; everything below it is arithmetic on
 * an array, and this is where the array first exists.
 *
 * THE RATE IS WRONG ON PURPOSE, and it is the one deliberate lie in the series.
 * The board runs at 250 Hz, so the 2 s window holds 500 samples per channel —
 * well over one per pixel. Drawn as dots they merge back into the line they came
 * from and the reader learns only that a trace is made of ink. So the figure
 * samples at 10 Hz instead: one dot per 100 ms, each the mean of 25 real samples.
 * What survives the exaggeration is the only thing this step has to teach — that
 * a continuous quantity has been replaced by one number per interval, and the
 * intervals are all there is from here on. meta.digitize carries both rates so
 * the prose can say which it is showing.
 *
 * ONE NUMBER PER INTERVAL, and the number is the interval's mean. Which makes
 * each band the whole truth about where its dot came from — the point of drawing
 * the band at all. The real converter does NOT do this: an ADS1299 decimates
 * with a third-order sinc, so a sample is a bell-weighted mean 3 output periods
 * wide whose centre of mass sits 1.5 periods behind the timestamp it is reported
 * under, and consecutive samples overlap three deep. A one-slot band would lie
 * about that. meta.digitize.adc carries the real filter — support, group delay,
 * -3 dB corner, all derived from f_CLK and the decimation ratio in the bake —
 * for prose that wants to name the gap. See adc_chain() there.
 *
 * THE CAMERA DOES NOT MOVE. Same 3 s window, same span, same scroll rate at both
 * ends of the toggle, so the only thing the control changes is the sampling. An
 * earlier cut opened on figure 6's 10 s frame and zoomed in as the dots appeared,
 * which made the seam exact but moved the camera and the operator together — and
 * a reader watching the picture change cannot then tell which of the two did it.
 * The cost is the seam: figure 6 opens five times wider, so the step from 5 to 6
 * is a zoom-out this figure no longer performs.
 *
 * COLOUR. Black and green are the two channels going in, red is the samples
 * coming out — one panel each, so the two red series are told apart by position
 * rather than by hue. The continuous traces do not morph toward red the way
 * figure 7's does: the dots REPLACE the curve rather than being the curve
 * transformed, and keeping R and L in their own colours is what lets a reader
 * follow which is which while both are on screen.
 *
 * ORDER OF APPEARANCE. Slots, then the samples that land in them, and only then
 * does the curve leave. A reader shown the dots before the intervals has no
 * reason to believe one-dot-per-interval is what happened rather than a
 * thinning-out of the line. */
FIGURES["fig-05"] = (mount, data) => {
  const meta = data.meta, fs = data.fs;
  const dg = meta.digitize;
  const R = data.sig("ch_R"), L = data.sig("ch_L");

  const VIEW = dg.view_s;                // 2 s, at both ends of the toggle
  const SLOT = dg.slot_s;                // 0.1 s of recording per sample -> 10 Hz
  const DOT_R = 3.4;                     // px. Thirty of these at 14 px apart,
                                         // not figure 10's five event markers, so
                                         // well under its 5.2.

  // Panel heights for a 2 s window, baked by the same worst-slice rule as the
  // top-level spans but over two seconds. The 10 s spans are several times too
  // tall here — a short slice of a raw channel moves a fraction of what a 10 s
  // slice does, and the samples flatten into a straight row with nothing to
  // sample.
  const SPAN = dg.span;
  // Fixed for the life of the figure, since the span is. Same padding rule as
  // Data.step(), which cannot be used directly: its cache is keyed on the
  // top-level span, and this figure's is the 1 s one.
  const STEP = {
    ch_R: gridStep(SPAN.ch_R, PANEL_H, SPAN.ch_R * (EDGE_FADE_PX / 2) / PANEL_H,
                   MIN_GRIDLINES),
    ch_L: gridStep(SPAN.ch_L, PANEL_H, SPAN.ch_L * (EDGE_FADE_PX / 2) / PANEL_H,
                   MIN_GRIDLINES),
  };
  // Coarsest rung that still puts at least eight labels on the panel. Derived
  // rather than set, because the window has been retuned twice and a hand-set
  // step silently stops fitting: whole seconds on a 2 s panel is two labels.
  const X_STEP = [1, 0.5, 0.2, 0.1].find((v) => VIEW / v >= 6) ?? 0.1;

  new Figure(mount, {
    height: FIG_H,
    // TWO STEPS ON ONE CLICK. They are two different claims — the axis divides
    // into intervals; only one number per interval survives — so they are staged
    // in sequence rather than blended, with a beat between them where step 1 is
    // fully up and nothing has happened to the data yet. One toggle, because a
    // reader should not have to work out that two controls are meant to be
    // pressed in order.
    //
    // morph_s is TRANSITION_S x 2, so each of the two steps gets the one second
    // every other transition in the series takes. The stages below are fractions
    // of that doubled clock: step 1 lands by 0.35, holds to 0.5, step 2 runs to 1.
    morph_s: TRANSITION_S * 2,
    aria: "mark the sampling intervals, then keep one sample per interval",
    draw: (ctx, W, H, s, clock) => {
      const top = CUE_HEADROOM, foot = PLOT_FOOT, ph = PANEL_H;
      const avail = H - top - foot;
      const yf = (u) => (u / 1000).toFixed(2);

      // Straight off the shell's clock, which advances at SCROLL_RATE — one
      // second of recording per real second, the same as every other figure. So
      // the panel's contents turn over every 3 s and a new sample lands every
      // 100 ms: the real thing at real speed.
      const { t0, i0, i1 } = data.window(clock, 0, VIEW);

      // Step 1: stripes in, data untouched and in front. Step 2: stripes and data
      // both out, dots in — the stripes have done their job once the samples
      // exist, and step 2's whole point is that only the samples survive.
      const step2 = ease((s - 0.5) / 0.5);
      const aBand = ease(s / 0.35) * (1 - step2);
      const aCont = 1 - ease((s - 0.5) / 0.45);
      const aDot = ease((s - 0.55) / 0.45);

      const mk = (y, ctr, span) => new Plot(ctx,
        { x: 108, y, w: W - 120, h: ph },
        [t0, t0 + VIEW], [ctr - span / 2, ctr + span / 2]);

      // Slot bounds in ABSOLUTE recording time — band k is [k·SLOT, (k+1)·SLOT)
      // of the record, not of the screen. So a band travels with the samples it
      // covers and its dot can never migrate into a neighbour, which is figure
      // 9's rule for its grey window and holds here for the same reason.
      const kRange = () => [Math.floor(t0 / SLOT), Math.ceil((t0 + VIEW) / SLOT)];
      // Fade at the panel edges on the same 18 px as cue guides and x-axis
      // labels, so a band or a dot entering the window dissolves in.
      const edgeFade = (p, xx) =>
        clamp(Math.min(xx - p.x, p.x + p.w - xx) / 18, 0, 1);

      const slots = (p, alpha) => {
        const [k0, k1] = kRange();
        ctx.save();
        ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
        ctx.fillStyle = C.grid;
        // STRIPED, not every column: alternate slots are filled and the ones
        // between are left as background, so the boundary between two intervals
        // is a colour change rather than a gap. Filling all thirty needed a
        // pixel gap to separate them, and at 14 px a column that gap is a
        // meaningful slice of the interval it is supposed to be delimiting.
        // Parity comes off the ABSOLUTE slot index, so the stripes scroll with
        // the data instead of shimmering as the window slides past them.
        for (let k = k0; k <= k1; k++) {
          if (k % 2) continue;
          const xa = p.px(k * SLOT), xb = p.px((k + 1) * SLOT);
          ctx.globalAlpha = alpha * edgeFade(p, (xa + xb) / 2);
          ctx.fillRect(xa, p.y, xb - xa, p.h);
        }
        ctx.restore();
      };

      const dots = (p, get, alpha) => {
        const [k0, k1] = kRange();
        ctx.save();
        ctx.beginPath(); ctx.rect(p.x, p.y - 4, p.w, p.h + 8); ctx.clip();
        ctx.fillStyle = C.red;
        for (let k = k0; k <= k1; k++) {
          // Sample bounds from the same slot edges the band was drawn between,
          // so what is averaged and what is highlighted are one interval. Half
          // -open at both ends, so no sample is counted in two slots.
          const a = Math.max(0, Math.ceil(k * SLOT * fs));
          const b = Math.min(data.n, Math.ceil((k + 1) * SLOT * fs));
          if (b <= a) continue;
          const xx = p.px((k + 0.5) * SLOT);
          ctx.globalAlpha = alpha * edgeFade(p, xx);
          ctx.beginPath();
          ctx.arc(xx, p.py(winMean(get, a, b)), DOT_R, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      };

      // Figure 6's two panels in figure 6's two places, on figure 6's
      // window-mean centring. Both built before anything is drawn, because the
      // slot fills go UNDERNEATH the gridlines — unlike figure 9's single grey
      // wash, which covers them. Striped bands laid on top would erase the scale
      // across most of the panel; underneath, it survives the highlight.
      const build = (y, key, get) => mk(y, winMean(get, i0, i1), SPAN[key]);
      const pR = build(top, "ch_R", R);
      const pL = build(top + avail - ph, "ch_L", L);

      if (aBand > 0.004) { slots(pR, aBand); slots(pL, aBand); }

      // Cues over the slots but under the gridlines, which is figure 6's
      // stacking — so at s=0 the two figures are pixel-for-pixel the same frame.
      mk(top, 0, 1).cues(meta.cues, top, top + avail);

      const finish = (p, key, get, color, name) => {
        const sp = SPAN[key], ctr = (p.ymin + p.ymax) / 2;
        p.grid(ticksAt(ctr - sp / 2, ctr + sp / 2, STEP[key]), yf, "mV");
        p.label(name, color);
        if (aCont > 0.004) {
          ctx.globalAlpha = aCont;
          p.trace(get, i0, i1, fs, color, 1.2);
          ctx.globalAlpha = 1;
        }
        if (aDot > 0.004) dots(p, get, aDot);
      };
      finish(pR, "ch_R", R, C.ink, "R");
      finish(pL, "ch_L", L, C.green, "L");

      // One shared time axis, pinned to the bottom panel — figure 6's again.
      mk(top + avail - ph, 0, 1).xAxis("seconds", X_STEP);
    },
  });
};

/* ── 06 · subtract left from right ───────────────────────────────────────── */

/* The toggle drives only the two-panel → one-panel transition. The difference
   is always the full ch_R − ch_L; there is no half-subtracted state. */
FIGURES["fig-06"] = (mount, data) => {
  const meta = data.meta, fs = data.fs, view = data.view;
  const R = data.sig("ch_R"), L = data.sig("ch_L"), D = data.sig("diff");

  new Figure(mount, {
    height: FIG_H, aria: "combine the two channels into their difference",
    draw: (ctx, W, H, s, clock) => {
      // top leaves room for two staggered rows of gaze-cue labels above panel 1
      // top is CUE_HEADROOM so the upper cue-label row is not clipped; foot 36
      // is what the x-axis needs. Together they put yCentre on PLOT_TOP, so the
      // converged differential sits where every other figure's panel sits.
      const top = CUE_HEADROOM, foot = PLOT_FOOT, ph = PANEL_H;
      const avail = H - top - foot;
      const yCentre = top + (avail - ph) / 2;
      const yf = (v) => (v / 1000).toFixed(2);

      const { t0, i0, i1 } = data.window(clock);

      const converge = ease(s / 0.65);
      const aPair = 1 - ease((s - 0.10) / 0.55);
      const aDiff = ease((s - 0.35) / 0.50);
      // Grids fade out early and back in late: three panels converging on one
      // spot would otherwise stack three sets of tick labels on top of each other.
      const aPairGrid = 1 - ease(s / 0.25);
      const aDiffGrid = ease((s - 0.70) / 0.30);

      const mk = (y, ctr, span) => new Plot(ctx,
        { x: 108, y, w: W - 120, h: ph },
        [t0, t0 + view], [ctr - span / 2, ctr + span / 2]);

      const panel = (y, key, get, color, name, width, aTrace, aGrid) => {
        // Span is fixed per signal (baked from the worst slice in the whole
        // recording, so nothing can overflow); the centre follows the window,
        // which slides the slow electrode drift out of view and leaves the
        // glance filling a readable share of the panel.
        const span = data.span(key);
        const ctr = winMean(get, i0, i1);
        const p = mk(y, ctr, span);
        if (aGrid > 0.004) {
          ctx.globalAlpha = aGrid;
          p.grid(ticksAt(ctr - span / 2, ctr + span / 2, data.step(key)), yf, "mV");
        }
        if (aTrace > 0.004) {
          ctx.globalAlpha = aTrace;
          p.label(name, color);
          p.trace(get, i0, i1, fs, color, width);
        }
        ctx.globalAlpha = 1;
        return p;
      };

      // Gaze cues first, so every trace draws over them.
      mk(top, 0, 1).cues(meta.cues, top, top + avail);

      // R above L so the subtraction reads down the page. The recording's
      // canonical differential is ch_R − ch_L (rightward gaze positive).
      panel(lerp(top, yCentre, converge), "ch_R", R, C.ink, "R", 1.2,
            aPair, aPairGrid);
      panel(lerp(top + avail - ph, yCentre, converge), "ch_L", L, C.green, "L", 1.2,
            aPair, aPairGrid);
      panel(yCentre, "diff", D, C.red, "R − L", 1.5, aDiff, aDiffGrid);

      // One shared time axis, pinned to the bottom, labelled in recording seconds.
      mk(top + avail - ph, 0, 1).xAxis("seconds");
    },
  });
};

/* ── 07 · detrend ────────────────────────────────────────────────────────── */

/* detrend(subtract(R, L)) — the game's own detrend, not a stand-in for it.
 *
 * `_poll_eog` hands `_eog_filter` a 125-sample buffer and keeps only the newest
 * 25; `_eog_filter` opens with DataFilter.detrend(CONSTANT). So one constant —
 * the mean of the 125 samples ending at that poll — comes off all 25 samples
 * the poll delivers. live_detrend() reproduces that poll by poll, verified
 * sample-for-sample against the BrainFlow call.
 *
 * No text on the figure. It is read inline beside the prose that explains it,
 * so anything the sentence next to it already says is ink competing with the
 * trace.
 *
 * SEAM. At s=0 this is figure 6's closing panel — same signal, same span, same
 * centring rule — so the toggle is the only thing that moves. Every figure
 * opens where the last one closed. The one thing that does change at the seam
 * is the colour: figure 6's output is red, and arriving here the same array is
 * this figure's input, so it is black.
 *
 * MOTION. The trace stretches slowly; the axis chases it. Centre and gridline
 * step are recomputed each frame from what is on screen, so the labels sweep
 * from ~29 mV down to 0 as the offset leaves, and the trace cannot escape the
 * panel. */
FIGURES["fig-07"] = (mount, data) => {
  const meta = data.meta, fs = data.fs, view = data.view;
  const D = data.sig("diff"), DT = data.sig("detrend_live");
  const from = meta.valid_from_s;

  // Both endpoints are baked spans, so the axis never carries a hand-set
  // number. SPAN_OPEN is figure 6's differential panel — the seam.
  const SPAN_OPEN = data.span("diff");
  const SPAN_END = data.span("detrend_live");
  // Padding gridStep() keeps clear of the panel edges, matching Data.step().
  const gpad = (span) => span * (EDGE_FADE_PX / 2) / PANEL_H;

  new Figure(mount, {
    height: SINGLE_H,
    aria: "subtract the mean of the last half second, as the detector does",
    draw: (ctx, W, H, s, clock) => {
      const top = PLOT_TOP, ph = PANEL_H;
      const { t0, i0, i1 } = data.window(clock, from);

      // Not a dissolve: (1−s)·x + s·detrend(x) = x − s·µ, the same subtraction
      // scaled, so every intermediate s is an operator the pipeline could run.
      const get = (i) => lerp(D(i), DT(i), s);

      // Camera, not operator — figure 6's centring rule, so s=0 reproduces its
      // frame. Both spans are baked from the worst slice of their own signal,
      // and a blend cannot exceed the blend of the bounds, so the trace is
      // always inside the panel.
      const span = lerp(SPAN_OPEN, SPAN_END, s);
      const ctr = winMean(get, i0, i1);

      const p = new Plot(ctx, { x: 108, y: top, w: W - 120, h: ph },
                         [t0, t0 + view], [ctr - span / 2, ctr + span / 2]);

      // One grid, step recomputed each frame from the span on screen. Changing
      // rungs mid-morph is wanted here, not avoided; Plot.grid fades lines at
      // the panel edge so a change dissolves rather than snapping.
      const step = gridStep(span, ph, gpad(span), MIN_GRIDLINES);
      p.grid(ticksAt(ctr - span / 2, ctr + span / 2, step),
             (v) => (v / 1000).toFixed(2), "mV");

      // The zero line is the thing the trace is being moved onto, so it is drawn
      // as scale, not as data — one weight above a gridline, no label.
      const yz = p.py(0);
      if (yz > top && yz < top + ph) {
        ctx.save();
        ctx.strokeStyle = C.cue; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(p.x, yz); ctx.lineTo(p.x + p.w, yz); ctx.stroke();
        ctx.restore();
      }

      p.cues(meta.cues, top, top + ph);
      // Black going in, red coming out — the colour carries the same s as the
      // arithmetic, so the trace is the operation's own colour at every point.
      p.trace(get, i0, i1, fs, mix(C.ink, C.red, s), 1.5);
      p.xAxis("seconds");
    },
  });
};

/* ── 08 · filter ─────────────────────────────────────────────────────────── */

/* One toggle, one job — "filter it" — but the morph walks the chain a stage at
 * a time so you can see which filter did what, and each stage names itself as
 * it engages.
 *
 * Black in, red out, on the same s as the arithmetic — the same contract as
 * every other figure.
 *
 * THREE INDEPENDENT SLIDERS, one per filter, so any SUBSET of the chain can be
 * seen — low-pass and notch without the high-pass, notch alone, and so on. That
 * is exact rather than approximate. The filters are linear and commute, so
 *
 *   ∏ᵢ[(1−sᵢ)·I + sᵢ·Hᵢ] = Σ_S (∏_{i∈S} sᵢ)(∏_{i∉S}(1−sᵢ)) · ∏_{i∈S} Hᵢ
 *
 * — a multilinear interpolation of the eight subset outputs, all of which are
 * baked. Every slider position is therefore a filter the pipeline could have
 * run, not a crossfade between two pictures, and the corners of the cube are
 * the real thing.
 *
 * The corners are the ESSAY's, not the game's: low-pass 100 Hz, high-pass
 * 0.5 Hz, 4th order throughout, against the game's 30 / 0.1 and a 3rd-order
 * notch. 100 Hz is what keeps mains inside the passband, so the notch has
 * something visible to take out; at the game's 30 Hz corner both notch bands
 * sit above the low-pass and the toggle moves the trace a quarter of a pixel.
 * The chain is applied the way the game applies it — per 100 ms poll, to the
 * 125-sample buffer, IIR state starting from zero each time — so what diverges
 * is the numbers, deliberately, and nothing else. meta.game_filters carries the
 * real ones for prose that wants to name the difference.
 *
 * The axis is absolute — no window re-centring — so a gridline reading 0.00 mV
 * is 0 µV. That is only affordable because every subset here already starts
 * from figure 7's detrended array. */
FIGURES["fig-08"] = (mount, data) => {
  const meta = data.meta, fs = data.fs, view = data.view;
  const f = meta.filters;
  const ORDER = f.order;                       // ["lp", "notch", "hp"]
  // Layout key for a bitmask over ORDER — the empty set is figure 7's output.
  const KEY = (bits) => ORDER.filter((_, j) => bits >> j & 1).join("_") || "detrend_live";
  const CUBE = Array.from({ length: 8 }, (_, bits) => data.sig(KEY(bits)));
  // One span for the whole figure, big enough for every corner of the cube. A
  // blend is a convex combination of corners, so it cannot exceed their bound.
  const SPAN = Math.max(...Array.from({ length: 8 }, (_, b) => data.span(KEY(b))));
  const STEP = data.step("detrend_live", SPAN);
  const from = meta.valid_from_s;

  new Figure(mount, {
    height: SINGLE_H,
    toggles: ORDER.map((k) => ({ key: k, label: f[k].label })),
    draw: (ctx, W, H, v, clock) => {
      const top = PLOT_TOP, ph = PANEL_H;
      const { t0, i0, i1 } = data.window(clock, from);

      const s = ORDER.map((k) => clamp(v[k], 0, 1));
      // Multilinear weights over the cube's corners: sᵢ where the bit is set,
      // (1−sᵢ) where it is not. They sum to 1 by construction.
      const w = Array.from({ length: 8 }, (_, bits) =>
        s.reduce((p, si, j) => p * ((bits >> j & 1) ? si : 1 - si), 1));
      const get = (i) => {
        let a = 0;
        for (let b = 0; b < 8; b++) if (w[b] > 1e-9) a += w[b] * CUBE[b](i);
        return a;
      };

      const p = new Plot(ctx, { x: 108, y: top, w: W - 120, h: ph },
                         [t0, t0 + view], [-SPAN / 2, SPAN / 2]);
      p.grid(ticksAt(-SPAN / 2, SPAN / 2, STEP), (u) => (u / 1000).toFixed(2), "mV");
      p.cues(meta.cues, top, top + ph);
      // Black in, red out. With three knobs "out" is all three applied, so the
      // colour follows how much of the chain is engaged.
      const col = mix(C.ink, C.red, s.reduce((a, b) => a + b, 0) / s.length);
      p.trace(get, i0, i1, fs, col, 1.5);
      p.xAxis("seconds");
    },
  });
};

/* ── 08.5 · threshold amplitude, or threshold velocity ───────────────────── */

/* Three rows off ONE poll-by-poll computation: the R − L differential, the
 * game's filtered amplitude, and the same chain's velocity — rows 2 and 3
 * diverge only at the final step, and each carries its own ±6σ threshold from
 * the recording's own 5 s calibration. No control: the loop is the animation
 * (figure 9's precedent), and the argument is the comparison itself.
 *
 * TWO DELIBERATE DEPARTURES from the rest of the series, both the point:
 *
 * - THE GAME'S CORNERS, not the essay's. The data is baked through
 *   eog_core._eog_filter verbatim (LP 30 / HP 0.1, notch 3rd order). At the
 *   essay's LP 100 the derivative amplifies 40-100 Hz broadband so much that
 *   velocity LOSES on the 12-recording corpus (531/960 cued glances vs
 *   amplitude's 708/960 at 6σ) — PR #36's own "velocity performs best at
 *   ~35 Hz" warning, measured. At these corners: amplitude 526/960,
 *   velocity 926/960.
 *
 * - NOT the committed Player G recording. On Player G both rows catch 79/80 — the one
 *   subject in the corpus where nothing separates them. Player C (run #3 of the
 *   mass collection) is the honest witness: BOTH rows draw their σ from the
 *   same noisy opening — each lands at the 100th percentile of its own
 *   recording's 5 s blocks (σ_amp 51.3 µV vs blockwise median 14.9; σ_vel
 *   586.6 µV/s vs median 324) — so each detector gets the worst calibration
 *   it could have drawn. Amplitude's 6σ (±308 µV) sits ABOVE the recording's
 *   peak (265 µV): it cannot fire at all, 0/80 cued glances. Velocity's
 *   slimmer margin (peaks 2.2× its 6σ) still clears: 77/80. The bake
 *   (scripts/bake_ampvel_figure.py) carries the full numbers.
 *
 * Row names sit in the gutter per figures 2/3, but stacked at 10.5 px over
 * their band (label2) — "amplitude" at the standard 15 px would cross x=63
 * into the tick labels. Thresholds are figure 10's idiom: 1 px red rules,
 * both signs, under the trace. */
FIGURES["fig-085"] = (mount, data) => {
  const av = data.ampvel;
  if (!av) return;                       // sidecar bake not present
  const m = av.meta, fs = m.fs, view = m.view_s, from = m.valid_from_s;
  const DIFF = (i) => av.arrays.diff[i] + m.mean.diff;
  const AMP = (i) => av.arrays.amp[i];
  const VEL = (i) => av.arrays.vel[i] * m.vel_scale;
  const THR_A = m.sigma_thr * m.sigma.amp_uv;
  const THR_V = m.sigma_thr * m.sigma.vel_uvs;
  const S = m.span;
  const gpad = (span) => span * (EDGE_FADE_PX / 2) / PANEL_H;
  const STEP_D = gridStep(S.diff, PANEL_H, gpad(S.diff), MIN_GRIDLINES);
  const STEP_A = gridStep(S.amp, PANEL_H, gpad(S.amp), MIN_GRIDLINES);
  const STEP_V = gridStep(S.vel, PANEL_H, gpad(S.vel), MIN_GRIDLINES);

  new Figure(mount, {
    height: TRIPLE_H,
    control: "none",
    aria: "the same ten seconds three ways: raw differential, filtered "
        + "amplitude against its six-sigma threshold, and the same chain's "
        + "velocity against its own",
    draw: (ctx, W, H, _v, clock) => {
      const travel = m.duration_s - from - view;
      const t0 = from + (clock % travel);
      const i0 = Math.max(0, Math.floor(t0 * fs) - 1);
      const i1 = Math.min(m.n, Math.ceil((t0 + view) * fs) + 1);
      const xr = [t0, t0 + view];
      const y1 = PLOT_TOP;
      const y2 = y1 + PANEL_H + PANEL_GAP;
      const y3 = y2 + PANEL_H + PANEL_GAP;
      const box = { x: 108, w: W - 120 };

      // Row 1 centres on the window mean (figure 6/7's camera — drift dwarfs
      // any one slice); rows 2/3 are absolute, zero is zero (figure 8's rule).
      const ctr = winMean(DIFF, i0, i1);
      const p1 = new Plot(ctx, { ...box, y: y1, h: PANEL_H }, xr,
                          [ctr - S.diff / 2, ctr + S.diff / 2]);
      p1.grid(ticksAt(ctr - S.diff / 2, ctr + S.diff / 2, STEP_D),
              (v) => (v / 1000).toFixed(2), "mV");
      const p2 = new Plot(ctx, { ...box, y: y2, h: PANEL_H }, xr,
                          [-S.amp / 2, S.amp / 2]);
      p2.grid(ticksAt(-S.amp / 2, S.amp / 2, STEP_A),
              (v) => (v / 1000).toFixed(2), "mV");
      const p3 = new Plot(ctx, { ...box, y: y3, h: PANEL_H }, xr,
                          [-S.vel / 2, S.vel / 2]);
      p3.grid(ticksAt(-S.vel / 2, S.vel / 2, STEP_V),
              (v) => (v / 1000).toFixed(0), "mV/s");

      // Cue guides span the whole stack (figure 2/3's rule); labels above the
      // top panel only.
      p1.cues(m.cues, y1, y3 + PANEL_H);

      // Thresholds under the traces, both signs — _sustained_crossing tests
      // |x| and reads direction off the crossing afterwards.
      ctx.save();
      ctx.strokeStyle = C.red; ctx.lineWidth = 1;
      for (const [p, thr] of [[p2, THR_A], [p3, THR_V]]) {
        for (const v of [thr, -thr]) {
          const yy = p.py(v);
          if (yy < p.y || yy > p.y + p.h) continue;
          ctx.beginPath(); ctx.moveTo(p.x, yy); ctx.lineTo(p.x + p.w, yy); ctx.stroke();
        }
      }
      ctx.restore();

      p1.trace(DIFF, i0, i1, fs, C.ink, 1.5);
      p2.trace(AMP, i0, i1, fs, C.red, 1.5);
      p3.trace(VEL, i0, i1, fs, C.red, 1.5);

      p1.label2(["R − L"], C.ink, 15);
      p2.label2(["amplitude", "0.1–30 Hz"], C.red, 10.5);
      p3.label2(["velocity", "0.1–30 Hz"], C.red, 10.5);

      // σ is this figure's genuine output — the number the threshold lines
      // are drawn from — under the same rule as figure 9's readout.
      p2.note([[`6σ = ${THR_A.toFixed(1)} µV`, C.red]]);
      p3.note([[`6σ = ${Math.round(THR_V)} µV/s`, C.red]]);

      p3.xAxis("seconds");
    },
  });
};

/* ── 09 · calibrate → σ ──────────────────────────────────────────────────── */

/* What `_run_eog_sm` does in CALIBRATING, and the one number the next step needs.
 *
 * The trace is figure 8's output, unchanged — the same `lp_notch_hp` array it
 * ends on, in µV — and σ is the MAD of that. The live detector calibrates on
 * the VELOCITY instead (`_poll_eog` returns `_eog_velocity(filtered)[-n_new:]`,
 * so its σ is 342 µV/s), and the essay deliberately does not: keeping one
 * quantity and one unit the whole way down is worth more here than matching a
 * derivative the reader has not been shown. meta.calib.sigma_velocity carries
 * the detector's real number for prose that wants to name the gap.
 *
 * NO CONTROL. The loop is the animation. It runs the recording forward from
 * t=0 the way the board delivers it: a play-head advances in real time, the
 * window follows it once the recording is longer than the window, and every
 * 0.1 s a poll lands and σ is recomputed from everything counted so far. There
 * is nothing for a reader to toggle, so there is no toggle.
 *
 * The grey is anchored in RECORDING time, not screen position — it starts where
 * counting starts and ends at the play-head, so it travels with the samples it
 * marks rather than sitting still while they slide underneath.
 *
 * Black is the signal; red is the part of it that has been counted, tracking
 * the leading edge of the grey window, plus σ itself. The number earns its ink
 * under the rule the rest of the figures follow: it is a value this step
 * produces and the next step consumes, and no trace can show it.
 *
 * ONE DEPARTURE FROM THE CODE, deliberate and worth knowing. The state machine
 * evaluates 1.4826 × MAD exactly ONCE, when the buffer first reaches
 * EOG_BASELINE_S × sr; there is no running estimate in the game. Watching it
 * converge is the essay's construction, not the detector's — σ(k) for k < 50 is
 * a real function of the real data (`sigma_by_poll` in the bake) but it is a
 * number the software never materialises. What the game commits to is the last
 * value, 342 µV/s, and that is where the loop rests. */
FIGURES["fig-09"] = (mount, data) => {
  const meta = data.meta, fs = data.fs;
  const cal = meta.calib;
  const AMP = data.sig("lp_notch_hp");   // exactly what figure 8 ends on

  const NPOLL = cal.sigma_by_poll.length;      // 50 polls = 5 s
  const pollN = Math.round(cal.poll_s * fs);   // samples one poll delivers
  const calPoll0 = Math.round(cal.i0 / pollN); // poll index counting starts on
  // Wide enough that the finished calibration and its 3 s of run-out are on
  // screen together at the end of the loop.
  const VIEW = cal.t1 + cal.pad_s - cal.t0;
  const END_S = cal.t1 + cal.pad_s;            // play-head stops here
  const HOLD_S = 1.5;                          // dwell on the answer before looping
  const LOOP_S = END_S + HOLD_S;
  const SPAN = cal.span;
  const STEP = gridStep(SPAN, PANEL_H, SPAN * (EDGE_FADE_PX / 2) / PANEL_H,
                        MIN_GRIDLINES);

  new Figure(mount, {
    height: SINGLE_H, control: "none",
    draw: (ctx, W, H, _v, clock) => {
      const top = PLOT_TOP, ph = PANEL_H;

      // The play-head: where the board has streamed to. The window trails it
      // once there is more recording than fits, and never runs off the front.
      const now = Math.min(END_S, clock % LOOP_S);
      const t0 = clamp(now - VIEW, 0, Math.max(0, END_S - VIEW));
      const i0 = Math.max(0, Math.floor(t0 * fs) - 1);
      // Every moving edge is one sample index, and they are all derived from
      // the same one, so they cannot drift apart. Sample resolution, not poll
      // resolution: quantising the geometry to whole 0.1 s polls moves it in
      // 5 px steps and reads as chop, where a sample is 0.2 px and reads as
      // motion. σ still steps per poll, because σ genuinely is computed once a
      // poll — but a number ticking is not the same as a line stuttering.
      //
      // Integer arithmetic throughout, deliberately. Doing this in seconds puts
      // floor((now − 2.7) / 0.1) and floor(now / 0.1) on opposite sides of a
      // boundary — 4.5 − 2.7 is 1.7999999999999998 — and the highlight ends up
      // a whole poll behind the trace.
      const iNow = clamp(Math.floor(now * fs), 0, meta.n - 1);
      const iCal = Math.min(iNow, cal.i1 - 1);        // counting stops at t1
      const k = clamp(Math.floor((iNow - cal.i0) / pollN), 0, NPOLL);
      const sigma = k > 0 ? cal.sigma_by_poll[k - 1] : 0;
      const i1 = iNow + 1;

      const p = new Plot(ctx, { x: 108, y: top, w: W - 120, h: ph },
                         [t0, t0 + VIEW], [-SPAN / 2, SPAN / 2]);
      p.grid(ticksAt(-SPAN / 2, SPAN / 2, STEP), (v) => (v / 1000).toFixed(2), "mV");

      // The samples that have entered σ so far. Anchored to the recording, so
      // it slides with the data it marks; its right edge is the play-head, and
      // it stops growing when the buffer is full.
      if (iCal > cal.i0) {
        ctx.save();
        ctx.fillStyle = C.grid;
        // Bounds taken from the same sample indices the red trace is drawn
        // between, so the fill edge and the colour change are one edge.
        const xa = p.px(cal.i0 / fs), xb = p.px(iCal / fs);
        ctx.fillRect(xa, top, xb - xa, ph);
        ctx.restore();
      }

      p.cues(meta.cues, top, top + ph);
      // Black is the signal; red is the part of it that has been counted. The
      // boundary is the leading edge of the grey window, so the colour says
      // exactly which samples the number to the right was computed from.
      p.trace(AMP, i0, i1, fs, C.ink, 1.2);
      if (iCal > cal.i0) p.trace(AMP, cal.i0, iCal + 1, fs, C.red, 1.2);
      p.xAxis("seconds");

      if (sigma > 0) p.note([[`σ = ${sigma.toFixed(2)} µV`, C.red]]);
    },
  });
};

/* ── 10 · threshold → spike detected ─────────────────────────────────────── */

/* The last step: turn a signal into events by drawing a line across it.
 *
 * The rule is `_sustained_crossing`'s own — |x| over k·σ, held for at least
 * EOG_MIN_DUR_MS = 12 ms, then one detection per REFRACTORY_S = 0.8 s, which is
 * how `_run_eog_sm` spaces them. σ is the 8.02 µV figure 9 just measured, so the
 * two figures are one continuous argument: 9 finds the noise floor, 10 spends it.
 *
 * What this stops short of is the glance PAIR. A single sustained crossing is a
 * spike, not a command — the game wants an opposite crossing within
 * GLANCE_WINDOW_S before it moves a paddle. That is a further step, and this
 * figure is about the threshold.
 *
 * The slider runs 1σ to 5σ and opens at 2.5σ. Those are historical: sigma_thr
 * has held five values in this repo — 6.0, 5.0, 2.5, 4.0, 6.0 — and 2.5 is the
 * one it was dropped to when 5σ turned out to be unreachable on noisy baselines.
 *
 * Red is the part of the trace that clears the line — the same red as the
 * threshold that selected it and as the count in the corner, because all three
 * are this step's output. Black is what went in. A reader needs to see WHICH
 * excursions counted, not just how many. */
FIGURES["fig-10"] = (mount, data) => {
  const meta = data.meta, fs = data.fs, view = data.view;
  const det = meta.detect;
  const AMP = data.sig("lp_notch_hp");
  const SPAN = data.span("lp_notch_hp");
  const STEP = data.step("lp_notch_hp");
  const from = meta.valid_from_s;
  // EOG_MIN_DUR_MS in samples — a crossing must hold this long to count.
  const minDur = Math.max(1, Math.round(det.min_dur_ms / 1000 * fs));

  new Figure(mount, {
    height: SINGLE_H,
    control: { slider: [det.k_min, det.k_max], start: det.k_start,
               step: det.k_step, below: true, label: (k) => k.toFixed(2) + " σ" },
    aria: "move the detection threshold in units of the baseline sigma",
    draw: (ctx, W, H, k, clock) => {
      const top = PLOT_TOP, ph = PANEL_H;
      const { t0, i0, i1 } = data.window(clock, from);
      const thr = k * det.sigma;

      const p = new Plot(ctx, { x: 108, y: top, w: W - 120, h: ph },
                         [t0, t0 + view], [-SPAN / 2, SPAN / 2]);
      p.grid(ticksAt(-SPAN / 2, SPAN / 2, STEP), (v) => (v / 1000).toFixed(2), "mV");
      p.cues(meta.cues, top, top + ph);

      // The threshold itself, both signs — _sustained_crossing tests |x|, and
      // reads the direction off the crossing afterwards.
      ctx.save();
      ctx.strokeStyle = C.red; ctx.lineWidth = 1;
      for (const v of [thr, -thr]) {
        const y = p.py(v);
        ctx.beginPath(); ctx.moveTo(p.x, y); ctx.lineTo(p.x + p.w, y); ctx.stroke();
      }
      ctx.restore();

      p.trace(AMP, i0, i1, fs, C.ink, 1.5);

      // Runs that clear the line AND hold for min_dur — the second half is the
      // rule `_sustained_crossing` applies to kill single-sample spikes, and the
      // red was previously ignoring it, painting blips the detector rejects.
      const runs = [];
      let s = -1;
      for (let i = i0; i < i1; i++) {
        const on = Math.abs(AMP(i)) > thr;
        if (on && s < 0) s = i;
        else if (!on && s >= 0) { if (i - s >= minDur) runs.push([s, i]); s = -1; }
      }
      if (s >= 0 && i1 - s >= minDur) runs.push([s, i1]);

      ctx.save();
      ctx.beginPath(); ctx.rect(p.x, p.y - 3, p.w, p.h + 6); ctx.clip();
      ctx.strokeStyle = C.red; ctx.lineWidth = 1.5; ctx.lineJoin = "round";
      for (const [a, b] of runs) {
        // One sample either side, so the red meets the black rather than
        // floating off the end of it.
        ctx.beginPath();
        for (let i = Math.max(i0, a - 1); i <= Math.min(i1 - 1, b); i++) {
          const xx = p.px(i / fs), yy = p.py(AMP(i));
          if (i === Math.max(i0, a - 1)) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
        }
        ctx.stroke();
      }

      // The extreme of each run — the sample the crossing peaked at, whichever
      // side of zero it happened on. This is the moment the eye was moving
      // fastest, and it is what a reader is counting when they count spikes.
      ctx.fillStyle = C.red;
      for (const [a, b] of runs) {
        let pk = a;
        for (let i = a; i < b; i++) if (Math.abs(AMP(i)) > Math.abs(AMP(pk))) pk = i;
        ctx.beginPath();
        ctx.arc(p.px(pk / fs), p.py(AMP(pk)), 5.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      p.xAxis("seconds");

      // What the threshold bought, over the whole recording — not just the ten
      // seconds on screen, which is why these numbers hold still while the
      // window scrolls. `sweep` is indexed by slider position.
      // Two numbers, one per direction the threshold can fail: how much of the
      // real thing it found, and how much else it found. Counting CUES rather
      // than detections for the first, because a single glance crosses the line
      // more than once — the spike and its undershoot both count — so a spike
      // total runs to 136 against 80 real events and reads as nonsense.
      //
      // The second number is everything the threshold fired on that had no cue
      // near it. Worth knowing when writing the prose around this: not all of
      // them are the detector's mistakes — a blink, a jaw clench or a genuine
      // spontaneous saccade lands here too, and on 133 s of someone sitting
      // still, some of the 32 at 1.5σ will be real eye movements nobody asked
      // for. The label calls them false because that is what they are relative
      // to the task, not relative to the eye.
      const idx = clamp(Math.round((k - det.k_min) / det.k_step),
                        0, det.sweep.length - 1);
      const [n, hit, bad] = det.sweep[idx];
      p.note([
        [`${hit}/${det.n_cues} glances detected`, C.red],
        [`${bad} false glances`, C.red],
      ]);
    },
  });
};

/* ── bootstrap ───────────────────────────────────────────────────────────── */

/* Where the baked data lives, relative to the PAGE and not to this script.
 *
 * The harness serves the figures from their own directory so "." is right there.
 * An essay does not: a post at /2026/07/14/brain-pong.html would resolve a bare
 * "data/..." to /2026/07/14/data/... and 404. Host pages set
 *
 *     <script>window.EOG_FIG_BASE = "/assets/essay-figures";</script>
 *
 * before loading this file. */
const BASE = String(window.EOG_FIG_BASE || ".").replace(/\/+$/, "");

// Bump on every re-bake. The data files are fetched separately from the script,
// so without this a re-bake silently keeps serving the previous payload.
const DATA_V = 44;

readColors();
// Figure 8.5 draws a different recording through the game's own corners, so its
// data is a sidecar bake (scripts/bake_ampvel_figure.py) fetched only when its
// mount is on the page — an essay embedding figures one at a time pays for it
// only where it appears.
const wantAmpvel = !!document.getElementById("fig-085");
Promise.all([
  fetch(`${BASE}/data/eog-figures.json?v=${DATA_V}`).then((r) => r.json()),
  fetch(`${BASE}/data/eog-full.bin?v=${DATA_V}`).then((r) => r.arrayBuffer()),
  wantAmpvel
    ? fetch(`${BASE}/data/eog-ampvel.json?v=${DATA_V}`).then((r) => r.json())
    : Promise.resolve(null),
  wantAmpvel
    ? fetch(`${BASE}/data/eog-ampvel.bin?v=${DATA_V}`).then((r) => r.arrayBuffer())
    : Promise.resolve(null),
]).then(([meta, buf, avMeta, avBuf]) => {
  const data = new Data(meta, buf);
  if (avMeta && avBuf) {
    const arrays = {};
    avMeta.layout.forEach((k, i) => {
      arrays[k] = new Int16Array(avBuf, i * avMeta.n * 2, avMeta.n);
    });
    data.ampvel = { meta: avMeta, arrays };
  }
  for (const [id, build] of Object.entries(FIGURES)) {
    const mount = document.getElementById(id);
    if (mount) build(mount, data);
  }
});
