import { setStatus, renderTopK, fmt } from '../ui.js'
import { loadTMPoseModel } from '../modelLoader.js'

// NUEVAS etiquetas TM
const DEFAULT_POSE_LABELS = [
  'mirando',
  'no_mirando'
]

// Cómo se muestran en tu UI
const DISPLAY_POSE_LABELS = {
  mirando: 'Mirando 👀',
  no_mirando: 'No mirando ↘️'
}

export function initPoses () {
  const video = document.getElementById('pose-video')
  const canvas = document.getElementById('pose-canvas')
  const overlay = document.getElementById('pose-overlay')
  const st = document.getElementById('pose-status')
  const fpsSlider = document.getElementById('pose-fps')
  const fpsVal = document.getElementById('pose-fps-val')
  const top1 = document.getElementById('pose-top1')
  const top1p = document.getElementById('pose-top1prob')
  const top3 = document.getElementById('pose-top3')
  const lat = document.getElementById('pose-latency')
  const uploadInput = document.getElementById('pose-upload')

  let model = null
  let labels = DEFAULT_POSE_LABELS.slice()
let inputH = 224
let inputW = 224

  let stream = null
  let running = false
  let useCam = true

  let targetFPS = 15
  let lastInferTime = 0

  // ========== FPS SLIDER ==========
  if (fpsSlider && fpsVal) {
    fpsVal.textContent = String(fpsSlider.value)
    targetFPS = Number(fpsSlider.value)
    fpsSlider.addEventListener('input', () => {
      targetFPS = Number(fpsSlider.value)
      fpsVal.textContent = String(targetFPS)
    })
  }

  // ========== DRAW CANVAS ==========
  function clearOverlay () {
    if (!overlay) return
    overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height)
  }

  function resizeCanvases (w, h) {
    canvas.width = w
    canvas.height = h
    overlay.width = w
    overlay.height = h
  }

  // ========== PREPROCESO ==========
// Preprocesar imagen/canvas -> tensor [1, 224, 224, 3]
function preprocessFrame(tf) {
  const ctx = canvas.getContext('2d');

  // SIEMPRE usar 224x224 para modelos de Teachable Machine
  const TARGET_H = 224;
  const TARGET_W = 224;

  // Dibujar el frame original
  const frame = tf.browser.fromPixels(canvas);

  // 🔥 Redimensionar SIEMPRE a 224x224
  const resized = tf.image.resizeBilinear(frame, [TARGET_H, TARGET_W]);

  const floatImg = resized.toFloat().div(255);
  const batched = floatImg.expandDims(0); // [1,224,224,3]

  frame.dispose();
  resized.dispose();

  return batched;
}


  // ========== INFERENCIA ==========
  function inferFrame () {
    if (!model || !running) return
    const tf = window.tf
    if (!tf) return

    const now = performance.now()
    if (now - lastInferTime < (1000 / targetFPS)) return
    lastInferTime = now

    const t0 = performance.now()

    tf.engine().startScope()
    try {
      const input = preprocessFrame(tf)
      let out = model.predict(input)
      if (Array.isArray(out)) out = out[0]

      const probs = out.dataSync()

      const mapped = labels.map((key, i) => ({
        label: DISPLAY_POSE_LABELS[key] ?? key,
        prob: probs[i] ?? 0
      })).sort((a, b) => b.prob - a.prob)

      const best = mapped[0]
      top1.textContent = best.label
      top1p.textContent = fmt(best.prob)
      renderTopK(top3, mapped.slice(0, 3))

      lat.textContent = Math.round(performance.now() - t0)
    } catch (err) {
      console.error('Error en inferencia atención:', err)
      setStatus(st, 'error')
    } finally {
      tf.engine().endScope()
    }
  }

  // ========== LOOP ==========
  function drawFromVideo () {
    if (!running || !useCam || video.readyState < 2) return
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  }

  function loop () {
    if (!running) return
    if (useCam) {
      drawFromVideo()
      inferFrame()
    }
    requestAnimationFrame(loop)
  }

  // ========== CÁMARA ==========
  async function startCamera () {
    const tf = window.tf
    if (!tf) return

    try {
      setStatus(st, 'loading')
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false
      })
      video.srcObject = stream
      await video.play()

      resizeCanvases(video.videoWidth || 640, video.videoHeight || 480)

      running = true
      useCam = true
      setStatus(st, 'ready')
      loop()
    } catch (err) {
      console.error('No se pudo iniciar la cámara:', err)
      setStatus(st, 'denied')
    }
  }

  function stopCamera () {
    if (stream) {
      for (const t of stream.getTracks()) t.stop()
    }
    running = false
  }

  // ========== SUBIR IMAGEN ==========
  if (uploadInput) {
    uploadInput.addEventListener('change', (ev) => {
      const file = ev.target.files[0]
      if (!file) return

      stopCamera()
      useCam = false

      const img = new Image()
      img.onload = () => {
        resizeCanvases(img.width, img.height)
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height)
        clearOverlay()
        inferFrame()
      }
      img.src = URL.createObjectURL(file)
      uploadInput.value = ''
    })
  }

  // ========== INICIALIZACIÓN ==========
  ;(async () => {
    setStatus(st, 'loading')
    try {
      const loaded = await loadTMPoseModel('ModelosAI/Posturas') 
      model = loaded.model
      labels = DEFAULT_POSE_LABELS.slice()  // 🔥 reemplazamos las del TM si vienen diferentes

      if (loaded.inputShape) {
        inputH = loaded.inputShape.imgHeight || 160
        inputW = loaded.inputShape.imgWidth || 160
      }

      console.log('🟢 Modelo mirándo/no_mirando cargado:', labels)
      setStatus(st, 'ready')

      await startCamera()
    } catch (err) {
      console.error('❌ Error cargando modelo:', err)
      setStatus(st, 'error')
    }
  })()

  return () => {
    stopCamera()
    clearOverlay()
  }
}
