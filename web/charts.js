/* Minimal dependency-free SVG line charts in a "neo broker" style. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(min, max, n, integer) {
    if (!(max > min)) { max = min + 1; }
    const span = max - min;
    const step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    let step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    if (integer && step < 1) step = 1;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const fmt = (v, d) => (v == null || isNaN(v)) ? "–" : (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(d == null ? 2 : d));

  /**
   * lineChart(container, {series:[{name,color,points:[[x,y],...],dashed,fill,dots}], xLabel, yFmt})
   */
  function lineChart(container, cfg) {
    container.innerHTML = "";
    const W = container.clientWidth || 600, H = container.clientHeight || 200;
    const pad = { l: 42, r: 14, t: 10, b: 24 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" }, container);
    const series = cfg.series.filter(s => s.points && s.points.length);
    if (!series.length) {
      const t = el("text", { x: W / 2, y: H / 2, fill: "#8a94a6", "text-anchor": "middle", "font-size": 13 }, svg);
      t.textContent = "Noch keine Daten";
      return;
    }
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of series) for (const [x, y] of s.points) {
      if (y == null || isNaN(y)) continue;
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    if (cfg.yMin != null) ymin = Math.min(ymin, cfg.yMin);
    if (xmax === xmin) xmax = xmin + 1;
    const ypadv = (ymax - ymin) * 0.08 || 1;
    ymin -= ypadv; ymax += ypadv;
    const sx = x => pad.l + (x - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
    const sy = y => pad.t + (1 - (y - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
    // grid
    for (const v of niceTicks(ymin, ymax, 4)) {
      el("line", { x1: pad.l, x2: W - pad.r, y1: sy(v), y2: sy(v), stroke: "#232a38", "stroke-width": 1 }, svg);
      const t = el("text", { x: pad.l - 6, y: sy(v) + 4, fill: "#8a94a6", "font-size": 10, "text-anchor": "end" }, svg);
      t.textContent = fmt(v, cfg.yDigits == null ? 1 : cfg.yDigits);
    }
    for (const v of niceTicks(xmin, xmax, 6, true)) {
      const t = el("text", { x: sx(v), y: H - 6, fill: "#8a94a6", "font-size": 10, "text-anchor": "middle" }, svg);
      t.textContent = fmt(v, 0);
    }
    const defs = el("defs", {}, svg);
    series.forEach((s, i) => {
      const pts = s.points.filter(p => p[1] != null && !isNaN(p[1]));
      if (!pts.length) return;
      const d = pts.map((p, j) => (j ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
      if (s.fill) {
        const gid = "g" + i + Math.random().toString(36).slice(2, 7);
        const g = el("linearGradient", { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 }, defs);
        el("stop", { offset: "0%", "stop-color": s.color, "stop-opacity": .35 }, g);
        el("stop", { offset: "100%", "stop-color": s.color, "stop-opacity": 0 }, g);
        const area = d + ` L${sx(pts[pts.length - 1][0]).toFixed(1)} ${sy(ymin)} L${sx(pts[0][0]).toFixed(1)} ${sy(ymin)} Z`;
        el("path", { d: area, fill: `url(#${gid})` }, svg);
      }
      if (s.dots) {
        for (const p of pts) el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3, fill: s.color }, svg);
      }
      if (!s.dots || pts.length > 1) {
        el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dashed ? "5 5" : "", "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      }
      if (s.endDot !== false && !s.dashed) {
        const last = pts[pts.length - 1];
        el("circle", { cx: sx(last[0]), cy: sy(last[1]), r: 4, fill: s.color, stroke: "#151a23", "stroke-width": 2 }, svg);
      }
    });
    // hover tooltip
    container.style.position = "relative";
    const tip = document.createElement("div");
    tip.className = "chart-tip"; tip.hidden = true; container.appendChild(tip);
    const cursor = el("line", { y1: pad.t, y2: H - pad.b, stroke: "#8a94a6", "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0 }, svg);
    svg.addEventListener("mousemove", ev => {
      const r = svg.getBoundingClientRect();
      const x = xmin + (ev.clientX - r.left) / r.width * W;
      const xv = xmin + ((ev.clientX - r.left) / r.width * W - pad.l) / (W - pad.l - pad.r) * (xmax - xmin);
      cursor.setAttribute("x1", sx(xv)); cursor.setAttribute("x2", sx(xv)); cursor.setAttribute("opacity", 1);
      let html = `<div class="muted">${cfg.xLabel || "x"} ${fmt(xv, 0)}</div>`;
      for (const s of series) {
        let best = null;
        for (const p of s.points) if (p[1] != null && (best == null || Math.abs(p[0] - xv) < Math.abs(best[0] - xv))) best = p;
        if (best) html += `<div><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${s.color};margin-right:6px"></i>${s.name}: <b>${fmt(best[1], s.digits == null ? 2 : s.digits)}</b></div>`;
      }
      tip.innerHTML = html; tip.hidden = false;
      const left = (ev.clientX - r.left);
      tip.style.left = (left > r.width * 0.6 ? left - tip.offsetWidth - 12 : left + 12) + "px";
      tip.style.top = Math.max(0, ev.clientY - r.top - 20) + "px";
    });
    svg.addEventListener("mouseleave", () => { tip.hidden = true; cursor.setAttribute("opacity", 0); });
  }

  function sparkline(container, points, color) {
    container.innerHTML = "";
    const W = container.clientWidth || 120, H = container.clientHeight || 28;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" }, container);
    const pts = points.filter(p => p[1] != null && !isNaN(p[1]));
    if (pts.length < 2) return;
    let ymin = Infinity, ymax = -Infinity;
    for (const p of pts) { ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
    if (ymax === ymin) ymax = ymin + 1;
    const x0 = pts[0][0], x1 = pts[pts.length - 1][0] || 1;
    const sx = x => (x - x0) / ((x1 - x0) || 1) * (W - 2) + 1;
    const sy = y => H - 2 - (y - ymin) / (ymax - ymin) * (H - 4);
    const d = pts.map((p, j) => (j ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
    el("path", { d, fill: "none", stroke: color, "stroke-width": 1.5, "stroke-linejoin": "round" }, svg);
  }

  window.Charts = { lineChart, sparkline, fmt };
})();
