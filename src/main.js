import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { FACEMESH_TESSELATION } from './face-mesh-triangulation.js';

const W = 512;
const H = 512;

const LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const INNER_UPPER_LIP = [13,312,311,310,415,0,185,40,39,37,82,81,80,191];
const INNER_LOWER_LIP = [14,317,402,318,324,17,84,181,91,146,87,178,88,95];
const OUTER_MOUTH = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82,13,312,311,310,415,308,291,409,270,269,267,0,37,39,40,185];

const VISEMES = [
  { ms: 140, open: 0.08, round: 0.00, smile: 0.00 },
  { ms: 175, open: 0.98, round: 0.00, smile: 0.02 },
  { ms: 120, open: 0.34, round: 0.00, smile: 0.05 },
  { ms: 155, open: 0.62, round: 0.00, smile: 0.20 },
  { ms: 95,  open: 0.03, round: 0.00, smile: 0.00 },
  { ms: 170, open: 0.72, round: 0.82, smile: 0.00 },
  { ms: 120, open: 0.26, round: 0.46, smile: 0.00 },
  { ms: 150, open: 0.86, round: 0.05, smile: 0.04 },
  { ms: 90,  open: 0.05, round: 0.00, smile: 0.00 },
  { ms: 150, open: 0.46, round: 0.00, smile: 0.18 },
];

class Ikabot2DRig {
  constructor() {
    this.faceLandmarker = null;
    this.image = null;
    this.landmarks = null;
    this.base = null;
    this.crop = null;
    this.ready = false;
    this.speaking = false;
    this.speechStartedAt = 0;
    this.speechEndsAt = 0;
    this.current = { open: 0, round: 0, smile: 0 };
    this.target = { open: 0, round: 0, smile: 0 };
    this.nextBlinkAt = performance.now() + 1800 + Math.random() * 1800;
    this.blinkStartedAt = 0;
    this.lastFrame = performance.now();
    this.lastPaint = 0;

    this.canvas = document.getElementById('ikabotCanvas');
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.canvas.width = W;
    this.canvas.height = H;
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';

    this.init();
  }

  async init() {
    this.bindUI();
    this.paintEmpty();
    this.status('Chargement du moteur visage…', 'loading');

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
          delegate: 'GPU'
        },
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
        runningMode: 'IMAGE',
        numFaces: 1
      });
      this.status('Moteur prêt. Choisis la photo.', 'success');
    } catch (e) {
      console.error(e);
      this.status('Impossible de charger MediaPipe : ' + e.message, 'error');
    }

    requestAnimationFrame((t) => this.loop(t));
  }

  bindUI() {
    const input = document.getElementById('fileInput');
    const upload = document.getElementById('uploadArea');
    const process = document.getElementById('processBtn');
    const talk = document.getElementById('talkBtn');
    const stop = document.getElementById('stopBtn');

    upload.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) this.loadFile(file);
    });

    upload.addEventListener('dragover', (e) => {
      e.preventDefault();
      upload.classList.add('dragover');
    });
    upload.addEventListener('dragleave', () => upload.classList.remove('dragover'));
    upload.addEventListener('drop', (e) => {
      e.preventDefault();
      upload.classList.remove('dragover');
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file && file.type && file.type.startsWith('image/')) this.loadFile(file);
    });

    process.addEventListener('click', () => this.process());
    talk.addEventListener('click', () => this.startSpeechDemo());
    stop.addEventListener('click', () => this.stopSpeech());
  }

  loadFile(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = url;
      this.image = img;
      this.landmarks = null;
      this.ready = false;
      this.drawFittedImage();
      document.getElementById('processBtn').disabled = false;
      document.getElementById('talkBtn').disabled = true;
      document.getElementById('stopBtn').disabled = true;
      this.status('Photo chargée. Appuie sur « Analyser le visage ».', 'success');
    };
    img.onerror = () => this.status('Impossible de lire cette image.', 'error');
    img.src = url;
  }

  async process() {
    if (!this.image || !this.faceLandmarker) return;
    const button = document.getElementById('processBtn');
    button.disabled = true;
    this.status('Analyse du visage…', 'loading');

    try {
      const results = this.faceLandmarker.detect(this.image);
      if (!results.faceLandmarks || !results.faceLandmarks.length) {
        throw new Error('aucun visage détecté');
      }
      this.landmarks = results.faceLandmarks[0];
      this.prepareRig();
      this.ready = true;
      document.getElementById('talkBtn').disabled = false;
      this.status('Visage prêt. Lance « Test parole » : aucun changement de photo.', 'success');
      this.paintFrame(0);
    } catch (e) {
      console.error(e);
      this.status('Erreur : ' + e.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  prepareRig() {
    const xs = this.landmarks.map((p) => p.x * this.image.width);
    const ys = this.landmarks.map((p) => p.y * this.image.height);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const fw = maxX - minX;
    const fh = maxY - minY;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2 + fh * 0.08;

    let size = Math.max(fw * 1.95, fh * 1.72);
    size = Math.min(size, this.image.width, this.image.height);
    let x = cx - size / 2;
    let y = cy - size / 2;
    x = Math.max(0, Math.min(x, this.image.width - size));
    y = Math.max(0, Math.min(y, this.image.height - size));
    this.crop = { x, y, size };

    this.base = this.landmarks.map((p) => {
      const sx = p.x * this.image.width;
      const sy = p.y * this.image.height;
      return {
        sx,
        sy,
        x: (sx - x) / size * W,
        y: (sy - y) / size * H
      };
    });

    this.mouth = {
      cx: (this.base[61].x + this.base[291].x) / 2,
      cy: (this.base[13].y + this.base[14].y) / 2,
      width: Math.max(20, Math.abs(this.base[291].x - this.base[61].x))
    };

    this.eyeCenters = {
      left: {
        x: (this.base[33].x + this.base[133].x) / 2,
        y: (this.base[159].y + this.base[145].y) / 2,
      },
      right: {
        x: (this.base[362].x + this.base[263].x) / 2,
        y: (this.base[386].y + this.base[374].y) / 2,
      }
    };
  }

  startSpeechDemo() {
    if (!this.ready) return;
    this.speaking = true;
    this.speechStartedAt = performance.now();
    this.speechEndsAt = this.speechStartedAt + 10000;
    document.getElementById('talkBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('state').textContent = 'PARLE';
  }

  stopSpeech() {
    this.speaking = false;
    this.target = { open: 0, round: 0, smile: 0 };
    document.getElementById('talkBtn').disabled = !this.ready;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('state').textContent = 'REPOS';
  }

  updateSpeech(now) {
    if (!this.speaking) {
      this.target = { open: 0, round: 0, smile: 0 };
      return;
    }

    if (now >= this.speechEndsAt) {
      this.stopSpeech();
      return;
    }

    const elapsed = now - this.speechStartedAt;
    const cycle = VISEMES.reduce((a, v) => a + v.ms, 0);
    let t = elapsed % cycle;
    let chosen = VISEMES[0];
    for (const v of VISEMES) {
      if (t <= v.ms) {
        chosen = v;
        break;
      }
      t -= v.ms;
    }
    this.target = chosen;
  }

  blinkAmount(now) {
    if (!this.ready) return 0;
    if (!this.blinkStartedAt && now >= this.nextBlinkAt) this.blinkStartedAt = now;
    if (!this.blinkStartedAt) return 0;

    const d = now - this.blinkStartedAt;
    const duration = 170;
    if (d >= duration) {
      this.blinkStartedAt = 0;
      this.nextBlinkAt = now + 2500 + Math.random() * 2800;
      return 0;
    }
    return Math.sin(Math.PI * d / duration);
  }

  loop(now) {
    const dt = Math.min(50, now - this.lastFrame);
    this.lastFrame = now;
    this.updateSpeech(now);

    const tau = 86;
    const a = 1 - Math.exp(-dt / tau);
    this.current.open += (this.target.open - this.current.open) * a;
    this.current.round += (this.target.round - this.current.round) * a;
    this.current.smile += (this.target.smile - this.current.smile) * a;

    const blink = this.blinkAmount(now);
    const moving =
      this.speaking ||
      blink > 0.002 ||
      this.current.open > 0.003 ||
      this.current.round > 0.003 ||
      this.current.smile > 0.003;

    if (this.ready && moving && now - this.lastPaint >= 30) {
      this.lastPaint = now;
      this.paintFrame(blink);
    } else if (this.ready && !moving && now - this.lastPaint > 250) {
      this.lastPaint = now;
      this.paintFrame(0);
    }

    requestAnimationFrame((t) => this.loop(t));
  }

  deform(blink) {
    const pts = this.base.map((p) => ({ ...p }));
    const { cx, cy, width } = this.mouth;
    const open = this.current.open;
    const round = this.current.round;
    const smile = this.current.smile;

    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const dx = (p.x - cx) / width;
      const dy = (p.y - cy) / width;
      const mouthInfluence = Math.exp(-(dx * dx * 2.1 + dy * dy * 4.8));

      if (mouthInfluence > 0.015) {
        const below = Math.max(0, Math.min(1, (p.y - cy) / (width * 0.62) + 0.22));
        const above = Math.max(0, Math.min(1, (cy - p.y) / (width * 0.45) + 0.12));

        p.y += open * width * 0.155 * mouthInfluence * (0.30 + below * 1.05);
        p.y -= open * width * 0.035 * mouthInfluence * above;
        p.x += (cx - p.x) * round * 0.17 * mouthInfluence;

        if (smile > 0) {
          const side = Math.min(1, Math.abs(p.x - cx) / (width * 0.55));
          p.y -= smile * width * 0.075 * mouthInfluence * side;
          p.x += Math.sign(p.x - cx) * smile * width * 0.025 * mouthInfluence;
        }
      }
    }

    for (const i of INNER_UPPER_LIP) {
      if (pts[i]) pts[i].y -= open * width * 0.045;
    }
    for (const i of INNER_LOWER_LIP) {
      if (pts[i]) pts[i].y += open * width * 0.170;
    }
    for (const i of OUTER_MOUTH) {
      if (pts[i]) pts[i].x += (cx - pts[i].x) * round * 0.09;
    }

    if (blink > 0) {
      this.applyBlink(pts, LEFT_EYE, this.eyeCenters.left, blink);
      this.applyBlink(pts, RIGHT_EYE, this.eyeCenters.right, blink);
    }

    return pts;
  }

  applyBlink(pts, indices, center, amount) {
    for (const i of indices) {
      const p = pts[i];
      if (!p) continue;
      p.y += (center.y - p.y) * amount * 0.92;
    }
  }

  paintFrame(blink) {
    this.drawBase();
    if (!this.base || !this.crop) return;

    const dst = this.deform(blink);
    const moved = new Float32Array(dst.length);
    for (let i = 0; i < dst.length; i++) {
      const dx = dst[i].x - this.base[i].x;
      const dy = dst[i].y - this.base[i].y;
      moved[i] = Math.hypot(dx, dy);
    }

    for (let i = 0; i < FACEMESH_TESSELATION.length; i += 3) {
      const ia = FACEMESH_TESSELATION[i];
      const ib = FACEMESH_TESSELATION[i + 1];
      const ic = FACEMESH_TESSELATION[i + 2];
      if ((moved[ia] + moved[ib] + moved[ic]) < 0.08) continue;
      this.drawTriangle(this.base[ia], this.base[ib], this.base[ic], dst[ia], dst[ib], dst[ic]);
    }
  }

  drawTriangle(sa, sb, sc, da, db, dc) {
    const sx0 = sa.sx, sy0 = sa.sy;
    const sx1 = sb.sx, sy1 = sb.sy;
    const sx2 = sc.sx, sy2 = sc.sy;
    const dx0 = da.x, dy0 = da.y;
    const dx1 = db.x, dy1 = db.y;
    const dx2 = dc.x, dy2 = dc.y;

    const den = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
    if (Math.abs(den) < 0.00001) return;

    const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / den;
    const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / den;
    const e = (dx0 * (sx1 * sy2 - sx2 * sy1) + dx1 * (sx2 * sy0 - sx0 * sy2) + dx2 * (sx0 * sy1 - sx1 * sy0)) / den;
    const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / den;
    const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / den;
    const f = (dy0 * (sx1 * sy2 - sx2 * sy1) + dy1 * (sx2 * sy0 - sx0 * sy2) + dy2 * (sx0 * sy1 - sx1 * sy0)) / den;

    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(dx0, dy0);
    ctx.lineTo(dx1, dy1);
    ctx.lineTo(dx2, dy2);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(a, b, c, d, e, f);
    ctx.drawImage(this.image, 0, 0);
    ctx.restore();
  }

  drawBase() {
    if (!this.image) {
      this.paintEmpty();
      return;
    }

    if (this.crop) {
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.drawImage(
        this.image,
        this.crop.x,
        this.crop.y,
        this.crop.size,
        this.crop.size,
        0,
        0,
        W,
        H
      );
    } else {
      this.drawFittedImage();
    }
  }

  drawFittedImage() {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#020604';
    ctx.fillRect(0, 0, W, H);
    if (!this.image) return;

    const s = Math.min(W / this.image.width, H / this.image.height);
    const dw = this.image.width * s;
    const dh = this.image.height * s;
    ctx.drawImage(this.image, (W - dw) / 2, (H - dh) / 2, dw, dh);
  }

  paintEmpty() {
    const g = this.ctx.createRadialGradient(W / 2, H / 2, 20, W / 2, H / 2, W * 0.7);
    g.addColorStop(0, '#0b2416');
    g.addColorStop(1, '#010302');
    this.ctx.fillStyle = g;
    this.ctx.fillRect(0, 0, W, H);
    this.ctx.fillStyle = 'rgba(80,255,130,.75)';
    this.ctx.font = '600 20px monospace';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('IKABOT // EN ATTENTE', W / 2, H / 2);
  }

  status(message, type) {
    const el = document.getElementById('status');
    el.className = 'status ' + (type || '');
    el.textContent = message;
  }
}

new Ikabot2DRig();
