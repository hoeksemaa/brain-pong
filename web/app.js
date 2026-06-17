"use strict";

const COLORS  = { L: "#4499ff", R: "#ff8844" };
const CUE_COL = { LEFT: "#4499ff", RIGHT: "#ff8844" };

const sel    = document.getElementById("rec");
const meta   = document.getElementById("meta");
const status = document.getElementById("status");

async function init() {
  let manifest;
  try {
    manifest = await (await fetch("data/manifest.json")).json();
  } catch (e) {
    status.textContent = "Failed to load manifest.json — run `python web/build.py` first.";
    return;
  }
  for (const r of manifest) {
    const o = document.createElement("option");
    o.value = r.id;
    o.textContent = `${r.subject} · ${r.duration_s}s @ ${r.sample_rate}Hz`;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => render(sel.value));
  if (manifest.length) render(manifest[0].id);
}

async function render(id) {
  status.textContent = "Loading…";
  const rec = await (await fetch(`data/${id}.json`)).json();
  meta.textContent =
    `${rec.n_samples.toLocaleString()} samples · ${rec.duration_s}s · raw, unfiltered`;

  const n = rec.channels.length;
  const gap = 0.08;                       // vertical gap between stacked panels
  const h   = (1 - gap * (n - 1)) / n;

  const traces = [];
  const layout = {
    paper_bgcolor: "#111317",
    plot_bgcolor:  "#181b21",
    font: { color: "#e6e8ec" },
    margin: { l: 70, r: 20, t: 16, b: 44 },
    showlegend: false,
    hovermode: "x unified",
    shapes: [],
    annotations: [],
  };

  rec.channels.forEach((ch, i) => {
    const axis = i === 0 ? "" : i + 1;             // y, y2, y3…
    const top  = 1 - i * (h + gap);
    const bot  = top - h;
    traces.push({
      x: ch.t, y: ch.uv,
      type: "scattergl", mode: "lines",
      line: { color: COLORS[ch.role] || "#aaffaa", width: 1 },
      name: ch.name, xaxis: "x", yaxis: "y" + axis,
      hovertemplate: `${ch.name}: %{y:.1f} µV<extra></extra>`,
    });
    layout["yaxis" + axis] = {
      domain: [Math.max(bot, 0), top],
      title: { text: ch.name + "  (µV)", font: { size: 12 } },
      gridcolor: "#23272f", zerolinecolor: "#3a4150", anchor: "x",
    };
  });

  layout.xaxis = {
    domain: [0, 1], anchor: "y" + (n > 1 ? n : ""),
    title: { text: "Time (s)" },
    gridcolor: "#23272f",
  };

  // Event cue lines (LEFT / RIGHT only — skip BASELINE/REST/DONE clutter)
  for (const ev of rec.events) {
    if (ev.label !== "LEFT" && ev.label !== "RIGHT") continue;
    layout.shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: ev.t, x1: ev.t, y0: 0, y1: 1,
      line: { color: CUE_COL[ev.label], width: 1, dash: "dot" },
      opacity: 0.5,
    });
  }

  const config = { responsive: true, scrollZoom: true, displaylogo: false };
  await Plotly.react("plot", traces, layout, config);
  status.textContent =
    `${rec.events.filter(e => e.label === "LEFT" || e.label === "RIGHT").length} gaze cues shown`;
}

init();
