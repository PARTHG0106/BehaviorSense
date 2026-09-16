/* Two figures, drawn as SVG so the marks obey the page's own tokens.
 *
 * Neither needs a colour scale: one hue carries the data and the second series in the
 * anchors plot is distinguished by SHAPE — a filled dot against a hollow ring. Encoding
 * identity in form rather than hue keeps it legible without colour vision and adds
 * nothing new to the palette, which is the point of having a small one.
 */

const svgNS = "http://www.w3.org/2000/svg";

const el = (name, attrs = {}) => {
  const node = document.createElementNS(svgNS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

/** Calibration anchors: injected corruption against measured failure rate. */
export function drawAnchors(mount) {
  const data = [
    { label: "honest", injected: 0, measured: 0.0 },
    { label: "1 in 4", injected: 25, measured: 24.7 },
    { label: "1 in 2", injected: 50, measured: 52.0 },
    { label: "all", injected: 100, measured: 100.0 },
  ];
  const W = 460, rowH = 46, padL = 62, padR = 58, padT = 8;
  const H = padT + data.length * rowH + 26;
  const x = (v) => padL + (v / 100) * (W - padL - padR);

  const svg = el("svg", {
    class: "plot", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Measured failure rate matches injected corruption at 0, 25, 50 and 100 per cent.",
  });

  data.forEach((d, i) => {
    const y = padT + i * rowH + rowH / 2;
    const g = el("g", { class: "plot__row" });
    g.append(el("text", { class: "plot__label", x: padL - 12, y: y + 4, "text-anchor": "end" }));
    g.lastChild.textContent = d.label;
    // The connecting rule is the message: how far the measurement sits from its target.
    g.append(el("line", {
      class: "plot__link", x1: x(Math.min(d.injected, d.measured)), y1: y,
      x2: x(Math.max(d.injected, d.measured)), y2: y,
    }));
    g.append(el("circle", { class: "plot__ring", cx: x(d.injected), cy: y, r: 5 }));
    g.append(el("circle", { class: "plot__dot", cx: x(d.measured), cy: y, r: 4 }));
    const label = el("text", {
      class: "plot__value", x: x(d.measured) + 12, y: y + 4,
    });
    label.textContent = `${d.measured.toFixed(1)}%`;
    g.append(label);
    svg.append(g);
  });
  mount.replaceChildren(svg);
}

/** Bar anchored to the axis, rounded only at the value end. A rectangle with a
 * uniform radius rounds the baseline too, which reads as a floating pill rather than
 * a measurement growing off an axis. */
function barPath(x0, x1, y, h, r = 4) {
  const w = Math.max(1, x1 - x0);
  const rad = Math.min(r, w, h / 2);
  return `M${x0} ${y} H${x0 + w - rad} A${rad} ${rad} 0 0 1 ${x0 + w} ${y + rad} `
    + `V${y + h - rad} A${rad} ${rad} 0 0 1 ${x0 + w - rad} ${y + h} H${x0} Z`;
}

/** Leave-one-corpus-out AUROC. One measure, one hue, chance marked. */
export function drawLodo(mount) {
  const data = [
    { label: "CAUCAFall", value: 0.734, n: 1159 },
    { label: "URFD", value: 0.722, n: 239 },
    { label: "Le2i", value: 0.586, n: 752 },
    { label: "GMDCSA", value: 0.545, n: 914 },
  ];
  const W = 760, rowH = 42, padL = 104, padR = 92, padT = 10;
  const H = padT + data.length * rowH + 30;
  const lo = 0.4, hi = 0.8;
  const x = (v) => padL + ((v - lo) / (hi - lo)) * (W - padL - padR);

  const svg = el("svg", {
    class: "plot", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "AUROC by held-out corpus: CAUCAFall 0.734, URFD 0.722, Le2i 0.586, GMDCSA 0.545. Chance is 0.5.",
  });

  data.forEach((d, i) => {
    const y = padT + i * rowH;
    const g = el("g", { class: "plot__row" });
    const name = el("text", { class: "plot__label", x: padL - 14, y: y + rowH / 2 + 4, "text-anchor": "end" });
    name.textContent = d.label;
    g.append(name);
    g.append(el("path", { class: "plot__bar", d: barPath(padL, x(d.value), y + 11, rowH - 22) }));
    const value = el("text", { class: "plot__value", x: x(d.value) + 12, y: y + rowH / 2 + 4 });
    value.textContent = `${d.value.toFixed(3)}`;
    g.append(value);
    const count = el("text", {
      class: "plot__label", x: W - 6, y: y + rowH / 2 + 4, "text-anchor": "end",
    });
    count.textContent = `${d.n.toLocaleString("en-GB")} windows`;
    g.append(count);
    svg.append(g);
  });

  // Chance is drawn LAST, over the bars, with a surface-coloured halo behind the
  // stroke. Drawn first it disappeared under every bar that crosses 0.5 — which is
  // most of them, and they are exactly the ones the line is there to qualify.
  const cx = x(0.5);
  const y2 = padT + data.length * rowH;
  svg.append(el("line", { class: "plot__gap", x1: cx, y1: padT, x2: cx, y2 }));
  svg.append(el("line", { class: "plot__chance", x1: cx, y1: padT, x2: cx, y2 }));
  const chanceLabel = el("text", {
    class: "plot__label", x: cx, y: y2 + 18, "text-anchor": "middle",
  });
  chanceLabel.textContent = "chance 0.50";
  svg.append(chanceLabel);

  mount.replaceChildren(svg);
}


