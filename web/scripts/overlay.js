/* Skeleton overlay: COCO-17 stick figures drawn over a playing video.
 *
 * The keypoints arrive from the backend in the SOURCE video's pixel coordinates, and the
 * <video> is laid out at whatever width the page gives it, so every position is scaled at
 * draw time. Getting that wrong produces skeletons that are plausibly shaped and floating
 * beside the person — right structure, wrong place, and nothing on screen says so.
 *
 * Identity is distinguished by WEIGHT AND FILL, not colour. The palette has three hues and
 * each already means something: ink is the verified state, pencil is a correction, ochre is
 * "watch this". A skeleton is none of those — it is measurement — so borrowing pencil to mean
 * "second person" would say "this person is a mistake". Instead the resident is drawn heavy
 * with filled joints, everyone else light with hollow ones, and the role is stated in mono
 * type beside the head, which is where this system puts facts.
 */

export const COCO_EDGES = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [0, 5], [0, 6],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

const INK = "#14212a";
const INK_SOFT = "#48595f";
const SHEET = "#e9eeed";
const MIN_SCORE = 0.3;      // matches Keypoints.visible() in the Python

/* Track ids are per-clip and arbitrary; what a viewer needs is "which of these is the
 * resident". Style follows role, and falls back to the track id only for ordering. */
function styleFor(role) {
  if (role === "resident") return { width: 3.0, fill: true, alpha: 1.0 };
  if (role === "visitor") return { width: 1.75, fill: false, alpha: 0.95 };
  return { width: 1.5, fill: false, alpha: 0.7 };   // unknown: drawn, not claimed
}

export class SkeletonOverlay {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.ctx = canvas.getContext("2d");
    this.payload = null;
    this.raf = null;
    this.onFrame = null;
  }

  load(payload) {
    this.payload = payload;
    this.resize();
  }

  /** Do the two decoders agree on the frame size? Null when they do, a message when they don't.
   *
   * The keypoints are in the coordinate space of the frame the SERVER decoded with cv2. The
   * <video> shows the frame the BROWSER decoded. When those disagree — a rotation matrix one
   * honours and the other ignores, non-square pixels, a container that lies about its own
   * dimensions — no scale factor can rescue the drawing, and the failure looks exactly like a
   * correctly-shaped skeleton floating beside the person. Silence there is the worst option:
   * the overlay is the only part of this page a viewer checks by eye, so if it cannot be
   * trusted it has to say so rather than draw a confident lie.
   */
  sizeDisagreement() {
    const pw = this.payload?.width, ph = this.payload?.height;
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!pw || !ph || !vw || !vh) return null;
    if (pw === vw && ph === vh) return null;
    const swapped = pw === vh && ph === vw;
    return `The backend decoded this clip as ${pw}×${ph}; your browser decodes it as `
      + `${vw}×${vh}. ${swapped
        ? "The dimensions are swapped, which means a rotation flag one decoder honours and "
          + "the other ignores — the skeletons below are drawn in the other orientation and "
          + "are not reliable."
        : "The skeleton positions are scaled from the backend's size and may not line up."}`;
  }

  /* The canvas is sized in DEVICE pixels and scaled back down in CSS, or the strokes are
   * soft on any display with a pixel ratio above 1 — which is most of them. */
  resize() {
    const rect = this.video.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cssW = rect.width;
    this.cssH = rect.height;
    this.draw();
  }

  /** Seconds of footage that were actually analysed, from the frames we were given. */
  analysedSpan() {
    const n = this.payload?.frames?.length || 0;
    return n ? n / (this.payload.fps || 15) : 0;
  }

  /** The extracted frame nearest `t`, or null when `t` is past the analysed span.
   *
   * NULL PAST THE END IS THE POINT. This used to be
   * `frames[Math.min(frames.length - 1, Math.round(t * fps))]`, which clamps - so on a 9:56
   * upload analysed to the 900-frame cap, every moment from 0:45 to 9:56 redrew frame 899 and
   * the skeleton sat frozen over the hob for nine minutes while the person walked around. Worse
   * than ugly: it draws pose over footage nothing ever looked at, which is a claim to have
   * tracked what was never decoded.
   */
  frameAt(t) {
    const frames = this.payload?.frames;
    if (!frames?.length) return null;
    const fps = this.payload.fps || 15;
    const i = Math.round(t * fps);
    if (i < 0 || i > frames.length - 1) return null;
    return frames[i];
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.cssW) return;
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    const t = this.video.currentTime || 0;
    const frame = this.frameAt(t);
    if (!frame) {
      // Say why the overlay is empty. A blank canvas over playing video reads as a broken
      // renderer; "analysis ended at 45.0 s" reads as the frame cap, which is what it is.
      const span = this.analysedSpan();
      if (span && t > span) this.notice(`analysis ended at ${span.toFixed(1)} s`);
      return;
    }

    // Source-pixels to CSS-pixels. The backend reports the size of the frame IT DECODED,
    // which is the space the keypoints live in, so this does not have to be guessed.
    const sx = this.cssW / (this.payload.width || this.cssW);
    const sy = this.cssH / (this.payload.height || this.cssH);

    for (const person of frame.people || []) {
      const st = styleFor(person.role);
      ctx.globalAlpha = st.alpha;
      const kp = person.kp;
      // A pose can be PRESENT and yet have no joint the tracker would trust. That used to
      // draw literally nothing — every edge and every joint skipped by the threshold, no box,
      // no label — so a person standing in plain view rendered as empty space while a weaker
      // detection elsewhere in the frame got a full skeleton. "Below threshold" is a fact
      // worth stating, and it is the same fact as "pose unusable", so it is drawn the same way.
      const usable = kp ? kp.some((j) => j && j[2] >= MIN_SCORE) : false;

      if (!usable) {
        // Tracked, but the pose was too occluded or too weak for Agent 2 to use. Drawn as an
        // empty box rather than omitted: "present, unusable" and "absent" are different facts.
        const [x1, y1, x2, y2] = person.box;
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = INK_SOFT;
        ctx.lineWidth = 1;
        ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
        ctx.setLineDash([]);
        this.label(person, x1 * sx, y1 * sy,
          kp ? `pose below ${MIN_SCORE}` : "pose unusable");
        continue;
      }

      ctx.strokeStyle = INK;
      ctx.lineWidth = st.width;
      ctx.lineCap = "round";
      for (const [a, b] of COCO_EDGES) {
        const p = kp[a], q = kp[b];
        if (!p || !q || p[2] < MIN_SCORE || q[2] < MIN_SCORE) continue;
        ctx.beginPath();
        ctx.moveTo(p[0] * sx, p[1] * sy);
        ctx.lineTo(q[0] * sx, q[1] * sy);
        ctx.stroke();
      }

      const r = st.fill ? 2.6 : 2.2;
      for (const j of kp) {
        if (j[2] < MIN_SCORE) continue;
        ctx.beginPath();
        ctx.arc(j[0] * sx, j[1] * sy, r, 0, Math.PI * 2);
        if (st.fill) { ctx.fillStyle = INK; ctx.fill(); }
        else { ctx.fillStyle = SHEET; ctx.fill(); ctx.lineWidth = 1.2; ctx.stroke(); }
      }

      // Anchor the label to the head when it is visible, the box top when it is not.
      const head = kp[0] && kp[0][2] >= MIN_SCORE ? kp[0] : null;
      const lx = head ? head[0] * sx : person.box[0] * sx;
      const ly = head ? head[1] * sy - 14 : person.box[1] * sy;
      this.label(person, lx, ly);
    }
    ctx.globalAlpha = 1;
  }

  /** A caption in the corner, for facts about the whole frame rather than about one person. */
  notice(text) {
    const ctx = this.ctx;
    ctx.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    const w = ctx.measureText(text).width;
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = SHEET;
    ctx.fillRect(6, 6, w + 10, 16);
    ctx.fillStyle = INK;
    ctx.fillText(text, 11, 17);
    ctx.globalAlpha = 1;
  }

  label(person, x, y, suffix = "") {
    const ctx = this.ctx;
    const role = person.role === "unknown" ? "unidentified" : person.role;
    const conf = person.role_confidence;
    const text = `${role}${conf ? ` ${Math.round(conf * 100)}%` : ""}`
      + `${suffix ? ` · ${suffix}` : ""}`;
    ctx.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    const w = ctx.measureText(text).width;
    const px = Math.max(2, Math.min(x - w / 2, this.cssW - w - 6));
    const py = Math.max(11, y);
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = SHEET;
    ctx.fillRect(px - 3, py - 9, w + 6, 12);
    ctx.fillStyle = INK;
    ctx.fillText(text, px, py);
  }

  /* Drive from requestAnimationFrame rather than `timeupdate`: that event fires roughly
   * four times a second, which makes the skeleton visibly lag the body it belongs to. */
  start() {
    const tick = () => {
      this.draw();
      if (this.onFrame) this.onFrame(this.video.currentTime || 0);
      this.raf = requestAnimationFrame(tick);
    };
    if (this.raf == null) this.raf = requestAnimationFrame(tick);
  }

  stop() {
    if (this.raf != null) cancelAnimationFrame(this.raf);
    this.raf = null;
  }
}
