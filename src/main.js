import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { FACEMESH_TESSELATION } from './face-mesh-triangulation.js';

const SLOTS = [
  { id: 'front', label: 'Face neutre', required: true, yaw: 0 },
  { id: 'left45', label: '3/4 gauche', required: true, yaw: -35 },
  { id: 'right45', label: '3/4 droit', required: true, yaw: 35 },
  { id: 'left70', label: 'Profil gauche', required: true, yaw: -70 },
  { id: 'right70', label: 'Profil droit', required: true, yaw: 70 },
  { id: 'mouthA', label: 'Face — bouche A', required: true, yaw: 0 },
  { id: 'mouthO', label: 'Face — bouche O', required: true, yaw: 0 },
  { id: 'eyesClosed', label: 'Face — yeux fermés', required: true, yaw: 0 },
];

const CENTER_INDEX = 168;
const FACE_TOP = 10;
const FACE_BOTTOM = 152;
const LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const OUTER_MOUTH = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82,13,312,311,310,415,308,291,409,270,269,267,0,37,39,40,185];
const LOWER_FACE = new Set([152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377]);

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const rad = (deg) => deg * Math.PI / 180;

class IkabotMultiView {
  constructor() {
    this.files = {};
    this.images = {};
    this.results = {};
    this.landmarker = null;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.avatar = null;
    this.faceMesh = null;
    this.background = null;

    this.demo = false;
    this.demoStart = 0;
    this.nextBlink = performance.now() + 1800;
    this.blinkStart = 0;
    this.current = { jawOpen: 0, mouthPucker: 0, mouthWide: 0, eyeBlinkLeft: 0, eyeBlinkRight: 0 };
    this.target = { ...this.current };
    this.last = performance.now();

    this.init();
  }

  async init() {
    this.buildSlots();
    this.initThree();
    this.bindButtons();
    this.setStatus('Chargement de MediaPipe…', 'loading');

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
          delegate: 'GPU'
        },
        runningMode: 'IMAGE',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: false
      });
      this.setStatus('Moteur prêt. Charge les 8 vues.', 'success');
    } catch (err) {
      console.error(err);
      this.setStatus('Erreur MediaPipe : ' + err.message, 'error');
    }

    requestAnimationFrame((t) => this.loop(t));
  }

  buildSlots() {
    const host = document.getElementById('photoSlots');
    for (const slot of SLOTS) {
      const wrap = document.createElement('label');
      wrap.className = 'photo-slot';
      wrap.dataset.id = slot.id;
      wrap.innerHTML = `
        <input type="file" accept="image/*" hidden>
        <div class="slot-preview"><span>+</span></div>
        <div class="slot-label">${slot.label}</div>
      `;
      const input = wrap.querySelector('input');
      input.addEventListener('change', (e) => {
        const f = e.target.files?.[0];
        if (f) this.loadSlot(slot.id, f, wrap);
      });
      host.appendChild(wrap);
    }
  }

  loadSlot(id, file, wrap) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      this.files[id] = file;
      this.images[id] = img;
      wrap.classList.add('loaded');
      wrap.querySelector('.slot-preview').innerHTML = '';
      wrap.querySelector('.slot-preview').style.backgroundImage = `url("${url}")`;
      wrap.querySelector('.slot-preview').style.backgroundSize = 'cover';
      wrap.querySelector('.slot-preview').style.backgroundPosition = 'center';
      this.refreshBuildButton();
    };
    img.onerror = () => this.setStatus('Impossible de lire ' + file.name, 'error');
    img.src = url;
  }

  refreshBuildButton() {
    const ready = SLOTS.every((s) => this.images[s.id]);
    document.getElementById('buildBtn').disabled = !ready || !this.landmarker;
  }

  bindButtons() {
    document.getElementById('buildBtn').addEventListener('click', () => this.buildAvatar());
    document.getElementById('demoBtn').addEventListener('click', () => {
      if (!this.faceMesh) return;
      this.demo = !this.demo;
      this.demoStart = performance.now();
      document.getElementById('demoBtn').textContent = this.demo ? '■ Arrêter la parole' : '▶ Test parole';
      document.getElementById('state').textContent = this.demo ? 'PARLE' : 'REPOS';
    });
    document.getElementById('exportBtn').addEventListener('click', () => this.exportGLB());
  }

  initThree() {
    const canvas = document.getElementById('stage');
    const host = canvas.parentElement;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x030604);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
    this.camera.position.set(0, 0, 3.15);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.minDistance = 2.25;
    this.controls.maxDistance = 4.5;
    this.controls.minPolarAngle = Math.PI * 0.35;
    this.controls.maxPolarAngle = Math.PI * 0.65;
    this.controls.minAzimuthAngle = -0.38;
    this.controls.maxAzimuthAngle = 0.38;

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.65));
    const key = new THREE.DirectionalLight(0xffffff, 1.15);
    key.position.set(2.5, 3, 4);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x44ff88, 0.55);
    rim.position.set(-3, 1, -2);
    this.scene.add(rim);

    const resize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h, false);
    };
    window.addEventListener('resize', resize);
    resize();
  }

  async buildAvatar() {
    const button = document.getElementById('buildBtn');
    button.disabled = true;
    document.getElementById('demoBtn').disabled = true;
    document.getElementById('exportBtn').disabled = true;

    try {
      this.setStatus('Analyse des 8 photos…', 'loading');
      this.results = {};

      for (const slot of SLOTS) {
        const result = this.landmarker.detect(this.images[slot.id]);
        if (!result.faceLandmarks?.length) {
          throw new Error('visage non détecté : ' + slot.label);
        }
        this.results[slot.id] = result.faceLandmarks[0];
      }

      this.setStatus('Fusion des vues et reconstruction du relief…', 'loading');
      const avatar = this.createAvatar();
      if (this.avatar) this.scene.remove(this.avatar);
      this.avatar = avatar;
      this.scene.add(avatar);

      document.getElementById('demoBtn').disabled = false;
      document.getElementById('exportBtn').disabled = false;
      document.getElementById('state').textContent = 'REPOS';
      this.setStatus('Ikabot multi-vues prêt. Fais bouger la vue avec le doigt puis lance Test parole.', 'success');
    } catch (err) {
      console.error(err);
      this.setStatus('Erreur : ' + err.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  faceHeight(lm) {
    return Math.max(0.0001, Math.abs(lm[FACE_BOTTOM].y - lm[FACE_TOP].y));
  }

  canonical2D(lm, i) {
    const h = this.faceHeight(lm);
    const c = lm[CENTER_INDEX];
    return {
      x: (lm[i].x - c.x) / h,
      y: -(lm[i].y - c.y) / h,
      z: -(lm[i].z - c.z) / h
    };
  }

  estimateDepth(i) {
    const front = this.results.front;
    const base = this.canonical2D(front, i);
    const estimates = [];

    for (const slot of SLOTS.filter((s) => s.yaw !== 0)) {
      const lm = this.results[slot.id];
      const q = this.canonical2D(lm, i);
      const theta = rad(slot.yaw);
      const s = Math.sin(theta);
      if (Math.abs(s) < 0.1) continue;
      const z = (q.x - base.x * Math.cos(theta)) / s;
      if (Number.isFinite(z)) estimates.push({ z, w: Math.abs(slot.yaw) < 60 ? 1 : 0.55 });
    }

    estimates.push({ z: base.z * 0.9, w: 0.8 });
    const values = estimates.map((e) => e.z).sort((a, b) => a - b);
    const med = values[Math.floor(values.length / 2)];
    const filtered = estimates.filter((e) => Math.abs(e.z - med) < 0.45);

    let num = 0, den = 0;
    for (const e of (filtered.length ? filtered : estimates)) {
      num += e.z * e.w;
      den += e.w;
    }
    return den ? num / den : base.z;
  }

  makeTexture(image, crop) {
    const size = 1024;
    const c = document.createElement('canvas');
    c.width = size;
    c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#090b09';
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(image, crop.x, crop.y, crop.size, crop.size, 0, 0, size, size);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
  }

  computeCrop(front) {
    const img = this.images.front;
    const xs = front.map((p) => p.x * img.width);
    const ys = front.map((p) => p.y * img.height);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const fw = maxX - minX;
    const fh = maxY - minY;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2 + fh * 0.06;
    let size = Math.max(fw * 1.9, fh * 1.70);
    size = Math.min(size, img.width, img.height);
    return {
      x: clamp(cx - size / 2, 0, img.width - size),
      y: clamp(cy - size / 2, 0, img.height - size),
      size
    };
  }

  createAvatar() {
    const front = this.results.front;
    const img = this.images.front;
    const crop = this.computeCrop(front);
    const texture = this.makeTexture(img, crop);

    const positions = [];
    const uvs = [];
    const depths = [];

    for (let i = 0; i < front.length; i++) {
      const px = front[i].x * img.width;
      const py = front[i].y * img.height;
      const x = ((px - crop.x) / crop.size) * 2 - 1;
      const y = 1 - ((py - crop.y) / crop.size) * 2;
      depths.push(this.estimateDepth(i));
      positions.push(x, y, 0);
      uvs.push((px - crop.x) / crop.size, 1 - (py - crop.y) / crop.size);
    }

    // Make the nose point toward the camera, then normalize the relief.
    let sign = depths[1] >= depths[CENTER_INDEX] ? 1 : -1;
    const centered = depths.map((z) => (z - depths[CENTER_INDEX]) * sign);
    const abs = centered.map(Math.abs).sort((a, b) => a - b);
    const scaleRef = abs[Math.floor(abs.length * 0.90)] || 0.15;
    for (let i = 0; i < centered.length; i++) {
      const z = clamp(centered[i] / scaleRef, -1.2, 1.2) * 0.34;
      positions[i * 3 + 2] = z;
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geom.setIndex(FACEMESH_TESSELATION);
    geom.computeVertexNormals();

    this.addExpressionMorphs(geom, positions, crop);

    const mat = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: 0.86,
      metalness: 0,
      side: THREE.DoubleSide
    });

    const face = new THREE.Mesh(geom, mat);
    face.name = 'IkabotFace';
    face.morphTargetDictionary = {
      jawOpen: 0,
      mouthPucker: 1,
      mouthWide: 2,
      eyeBlinkLeft: 3,
      eyeBlinkRight: 4
    };
    face.morphTargetInfluences = [0,0,0,0,0];
    this.faceMesh = face;

    // Flat portrait behind the 3D face keeps hair, beard, ears and clothes exactly from the source.
    const bgGeom = new THREE.PlaneGeometry(2, 2);
    const bgMat = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
    const bg = new THREE.Mesh(bgGeom, bgMat);
    bg.position.z = -0.16;
    bg.name = 'IkabotPortraitBase';
    this.background = bg;

    const group = new THREE.Group();
    group.name = 'IkabotAvatar';
    group.add(bg);
    group.add(face);
    group.scale.set(1.18, 1.18, 1.18);

    return group;
  }

  expressionDelta(targetId, i, crop) {
    const front = this.results.front;
    const target = this.results[targetId];
    const img = this.images.front;

    const f = front[i];
    const t = target[i];
    const fc = front[CENTER_INDEX];
    const tc = target[CENTER_INDEX];
    const hf = this.faceHeight(front);
    const ht = this.faceHeight(target);

    const fx = (f.x - fc.x) / hf;
    const fy = (f.y - fc.y) / hf;
    const tx = (t.x - tc.x) / ht;
    const ty = (t.y - tc.y) / ht;

    // Convert normalized facial displacement to mesh space.
    const pxPerNorm = hf * img.height / crop.size * 2;
    return {
      x: (tx - fx) * pxPerNorm,
      y: -(ty - fy) * pxPerNorm
    };
  }

  addExpressionMorphs(geom, basePositions, crop) {
    const names = ['jawOpen','mouthPucker','mouthWide','eyeBlinkLeft','eyeBlinkRight'];
    const morphs = names.map(() => new Float32Array(basePositions.length));

    const mouthSet = new Set(OUTER_MOUTH);
    const leftEyeSet = new Set(LEFT_EYE);
    const rightEyeSet = new Set(RIGHT_EYE);

    for (let i = 0; i < basePositions.length / 3; i++) {
      const bx = basePositions[i * 3];
      const by = basePositions[i * 3 + 1];

      // A expression captured from dedicated front photo.
      const a = this.expressionDelta('mouthA', i, crop);
      const dxM = bx - (basePositions[13 * 3] + basePositions[14 * 3]) / 2;
      const dyM = by - (basePositions[13 * 3 + 1] + basePositions[14 * 3 + 1]) / 2;
      const mouthInfluence = Math.exp(-(dxM * dxM * 7 + dyM * dyM * 15));
      const jawInfluence = by < -0.10 ? clamp((-by - 0.10) * 1.7, 0, 1) : 0;
      const wA = Math.max(mouthInfluence, jawInfluence * 0.42, LOWER_FACE.has(i) ? 0.12 : 0);
      morphs[0][i * 3] = a.x * wA;
      morphs[0][i * 3 + 1] = a.y * wA;
      morphs[0][i * 3 + 2] = -0.02 * jawInfluence;

      // O expression captured from dedicated front photo.
      const o = this.expressionDelta('mouthO', i, crop);
      const wO = mouthSet.has(i) ? 1 : mouthInfluence * 0.72;
      morphs[1][i * 3] = o.x * wO;
      morphs[1][i * 3 + 1] = o.y * wO;
      morphs[1][i * 3 + 2] = 0.035 * wO;

      // Wide E/I heuristic: move mouth corners outward/up without changing identity.
      const wWide = mouthInfluence;
      morphs[2][i * 3] = Math.sign(dxM) * 0.055 * wWide * Math.min(1, Math.abs(dxM) * 6);
      morphs[2][i * 3 + 1] = 0.018 * wWide;

      // Eyes closed expression captured from dedicated front photo.
      const e = this.expressionDelta('eyesClosed', i, crop);
      const wl = leftEyeSet.has(i) ? 1 : 0;
      const wr = rightEyeSet.has(i) ? 1 : 0;
      morphs[3][i * 3] = e.x * wl;
      morphs[3][i * 3 + 1] = e.y * wl;
      morphs[4][i * 3] = e.x * wr;
      morphs[4][i * 3 + 1] = e.y * wr;
    }

    geom.morphTargetsRelative = true;
    geom.morphAttributes.position = morphs.map((arr) => new THREE.Float32BufferAttribute(arr, 3));
  }

  updateDemo(now) {
    if (!this.faceMesh) return;

    if (this.demo) {
      const t = (now - this.demoStart) / 1000;
      const cycle = t % 2.6;

      if (cycle < 0.35) this.target = { ...this.target, jawOpen: 0.08, mouthPucker: 0, mouthWide: 0.15 };
      else if (cycle < 0.75) this.target = { ...this.target, jawOpen: 0.88, mouthPucker: 0, mouthWide: 0.05 };
      else if (cycle < 1.10) this.target = { ...this.target, jawOpen: 0.30, mouthPucker: 0, mouthWide: 0.68 };
      else if (cycle < 1.45) this.target = { ...this.target, jawOpen: 0.42, mouthPucker: 0.92, mouthWide: 0 };
      else if (cycle < 1.80) this.target = { ...this.target, jawOpen: 0.58, mouthPucker: 0.18, mouthWide: 0.10 };
      else if (cycle < 2.15) this.target = { ...this.target, jawOpen: 0.20, mouthPucker: 0, mouthWide: 0.60 };
      else this.target = { ...this.target, jawOpen: 0.04, mouthPucker: 0, mouthWide: 0.06 };
    } else {
      this.target.jawOpen = 0;
      this.target.mouthPucker = 0;
      this.target.mouthWide = 0;
    }

    // Natural blink.
    if (!this.blinkStart && now > this.nextBlink) this.blinkStart = now;
    let blink = 0;
    if (this.blinkStart) {
      const d = now - this.blinkStart;
      if (d > 170) {
        this.blinkStart = 0;
        this.nextBlink = now + 2400 + Math.random() * 3000;
      } else {
        blink = Math.sin(Math.PI * d / 170);
      }
    }
    this.target.eyeBlinkLeft = blink;
    this.target.eyeBlinkRight = blink;
  }

  loop(now) {
    const dt = Math.min(50, now - this.last);
    this.last = now;
    this.updateDemo(now);

    const a = 1 - Math.exp(-dt / 82);
    for (const key of Object.keys(this.current)) {
      this.current[key] += (this.target[key] - this.current[key]) * a;
    }

    if (this.faceMesh?.morphTargetInfluences) {
      const d = this.faceMesh.morphTargetDictionary;
      for (const [name, index] of Object.entries(d)) {
        this.faceMesh.morphTargetInfluences[index] = this.current[name] || 0;
      }
    }

    // Tiny idle movement in true 3D. Kept deliberately subtle.
    if (this.avatar) {
      const s = now / 1000;
      this.avatar.rotation.y = Math.sin(s * 0.43) * 0.025;
      this.avatar.rotation.x = Math.sin(s * 0.31 + 1.3) * 0.012;
      this.avatar.position.y = Math.sin(s * 0.55) * 0.008;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame((t) => this.loop(t));
  }

  exportGLB() {
    if (!this.avatar) return;
    this.setStatus('Préparation du GLB…', 'loading');
    const exporter = new GLTFExporter();
    exporter.parse(
      this.avatar,
      (result) => {
        const blob = new Blob([result], { type: 'model/gltf-binary' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ikabot-multiview.glb';
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        this.setStatus('GLB téléchargé.', 'success');
      },
      (err) => {
        console.error(err);
        this.setStatus('Export GLB impossible : ' + err.message, 'error');
      },
      { binary: true, embedImages: true, maxTextureSize: 1024 }
    );
  }

  setStatus(text, type = '') {
    const el = document.getElementById('status');
    el.textContent = text;
    el.className = 'status ' + type;
  }
}

new IkabotMultiView();
