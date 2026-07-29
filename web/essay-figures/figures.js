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
 * axis/grid are neutral so they never read as a data series. */
// Set two-line, "look" over the direction — roughly half the width of one line,
// which is what lets the labels survive the fast block's 0.55 s cue spacing.
const CUE_TEXT = { L: "left", R: "right", C: "center" };

const C = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["ink", "green", "red", "axis", "grid", "cue", "rule", "bg"]) {
    C[k] = cs.getPropertyValue("--" + k).trim();
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

  label(text, color) {
    const { ctx } = this;
    ctx.save();
    ctx.font = "15px " + font();
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(text, 6, this.y + this.h / 2);
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

/* Every figure, keyed by the id of the element it mounts into. Only the ones
   present in the DOM are built, so the essay can embed them one at a time. */
const FIGURES = {};

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

// Bump on every re-bake. The data files are fetched separately from the script,
// so without this a re-bake silently keeps serving the previous payload.
const DATA_V = 15;

readColors();
Promise.all([
  fetch(`data/eog-figures.json?v=${DATA_V}`).then((r) => r.json()),
  fetch(`data/eog-full.bin?v=${DATA_V}`).then((r) => r.arrayBuffer()),
]).then(([meta, buf]) => {
  const data = new Data(meta, buf);
  for (const [id, build] of Object.entries(FIGURES)) {
    const mount = document.getElementById(id);
    if (mount) build(mount, data);
  }
});
