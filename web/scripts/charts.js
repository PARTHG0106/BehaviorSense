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

/** Source-stratified AUROC, two models per corpus. Source groups include training
 *  windows for the dedicated fall head, so neither column is a held-out transfer
 *  estimate. The point of the figure is the gap, which is what the two-model design
 *  rests on: the 20-class ADL ensemble's fall-class posterior sits below chance on
 *  one corpus, the dedicated binary head does not. */
export function drawLodo(mount) {
  const data = [
    { label: "CAUCAFall", n: 1159, adl: 0.603, head: 0.970 },
    { label: "GMDCSA",    n: 914,  adl: 0.471, head: 0.928 },
    { label: "Le2i",      n: 752,  adl: 0.571, head: 0.989 },
    { label: "URFD",      n: 239,  adl: 0.560, head: 0.991 },
  ];
  const W = 980, rowH = 50, padL = 104, padR = 96, padT = 10;
  const H = padT + data.length * rowH + 70;
  const lo = 0.4, hi = 1.0;
  const x = (v) => padL + ((v - lo) / (hi - lo)) * (W - padL - padR);
  const barH = 14;
  const gap = 4;

  const svg = el("svg", {
    class: "plot", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Source-stratified AUROC, both models. Dedicated fall head: CAUCAFall 0.970, GMDCSA 0.928, Le2i 0.989, URFD 0.991. 20-class ADL ensemble fall posterior: CAUCAFall 0.603, GMDCSA 0.471, Le2i 0.571, URFD 0.560. Chance is 0.50.",
  });

  data.forEach((d, i) => {
    const y = padT + i * rowH + 6;
    const g = el("g", { class: "plot__row" });
    const name = el("text", { class: "plot__label", x: padL - 14, y: y + rowH / 2 - 2, "text-anchor": "end" });
    name.textContent = d.label;
    g.append(name);

    // ADL-ensemble bar (top): shorter, hollow-toned.
    g.append(el("path", { class: "plot__bar plot__bar--adl", d: barPath(padL, x(d.adl), y + 4, barH) }));
    const adlVal = el("text", { class: "plot__value", x: x(d.adl) + 8, y: y + 4 + barH - 2 });
    adlVal.textContent = `${d.adl.toFixed(3)}`;
    g.append(adlVal);

    // Dedicated fall head bar (bottom): the long one — same model as the AUPRC headline.
    g.append(el("path", { class: "plot__bar", d: barPath(padL, x(d.head), y + 4 + barH + gap, barH) }));
    const headVal = el("text", { class: "plot__value", x: x(d.head) + 8, y: y + 4 + 2 * barH + gap - 2 });
    headVal.textContent = `${d.head.toFixed(3)}`;
    g.append(headVal);

    const count = el("text", {
      class: "plot__label", x: W - 6, y: y + rowH / 2 - 2, "text-anchor": "end",
    });
    count.textContent = `${d.n.toLocaleString("en-GB")} windows`;
    g.append(count);
    svg.append(g);
  });

  // Chance line.
  const cx = x(0.5);
  const y2 = padT + data.length * rowH;
  svg.append(el("line", { class: "plot__gap", x1: cx, y1: padT, x2: cx, y2 }));
  svg.append(el("line", { class: "plot__chance", x1: cx, y1: padT, x2: cx, y2 }));
  const chanceLabel = el("text", {
    class: "plot__label", x: cx, y: y2 + 18, "text-anchor": "middle",
  });
  chanceLabel.textContent = "chance 0.50";
  svg.append(chanceLabel);

  // Legend on its own line, below the chance label, so the two don't collide.
  // Both swatches sit on the left so neither label can fall outside the viewBox.
  const legendY = y2 + 44;
  const swatch1 = el("rect", { class: "plot__swatch plot__swatch--adl", x: padL, y: legendY - 8, width: 12, height: 12, rx: 2 });
  svg.append(swatch1);
  const leg1 = el("text", { class: "plot__label", x: padL + 20, y: legendY + 2, "text-anchor": "start" });
  leg1.textContent = "ADL ensemble (fall posterior)";
  svg.append(leg1);
  const swatch2 = el("rect", { class: "plot__swatch", x: padL + 230, y: legendY - 8, width: 12, height: 12, rx: 2 });
  svg.append(swatch2);
  const leg2 = el("text", { class: "plot__label", x: padL + 250, y: legendY + 2, "text-anchor": "start" });
  leg2.textContent = "Dedicated fall head";
  svg.append(leg2);

  mount.replaceChildren(svg);
}


