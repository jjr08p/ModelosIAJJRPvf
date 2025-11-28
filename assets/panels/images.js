// assets/panels/images.js
import { setStatus, renderTopK, fmt } from '../ui.js'
import { loadTMImageModel } from '../modelLoader.js'

// Etiquetas por defecto, por si el metadata no trae
const CLASSES = ['Futbol', 'Basket', 'Tenis']

export function initImages () {
  const video = document.getElementById('img-video')
  const canvas = document.getElementById('img-canvas')
  const st = document.getElementById('img-status')
  const demoBadge = document.getElementById('img-demo')
  const btnFreeze = document.getElementById('img-freeze')
  const inputUpload = document.getElementById('img-upload')
  const fpsSlider = document.getElementById('img-fps')
  const fpsVal = document.getElementById('img-fps-val')

  const top1 = document.getElementById('img-top1')
  const top1p = document.getElementById('img-top1prob')
  const top3 = document.getElementById('img-top3')
  const lat = document.getElementById('img-latency')

  let model = null
  let labels = CLASSES.slice()
  let imageSize = { width: 224, height: 224 }

  let stream = null
  let running = false
  let frozen = false
  let loopId = null
  let fps = 30

  // ==========================
  // Carga de modelo
  // ==========================
  async function loadModel () {
    setStatus(st, 'loading')
    try {
      const loaded = await loadTMImageModel('ModelosAI/Imagenes')
      model = loaded.model
      if (loaded.labels && loaded.labels.length) {
        labels = loaded.labels
      }
      if (loaded.imageSize && loaded.imageSize.width && loaded.imageSize.height) {
        imageSize = loaded.imageSize
      }
      console.log('✅ Modelo de imágenes listo. Clases:', labels)
      setStatus(st, 'ready')
      demoBadge.classList.add('hidden')
    } catch (e) {
      console.error('❌ Error cargando modelo de imágenes', e)
      setStatus(st, 'demo')
      demoBadge.classList.remove('hidden')
    }
  }

  // ==========================
  // Cámara
  // ==========================
    async function startCam () {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      video.srcObject = stream
      await video.play()
      video.classList.remove('hidden')   // 👈 aseguramos que se vea
      running = true
      startLoop()
    } catch (e) {
      console.error('No se pudo iniciar la cámara', e)
      setStatus(st, 'denied')
    }
  }


  function stopCam () {
    running = false
    if (loopId != null) cancelAnimationFrame(loopId)
    if (stream) {
      for (const tr of stream.getTracks()) tr.stop()
      stream = null
    }
  }

  function drawFromVideo () {
    if (!video || video.readyState < 2) return
    const ctx = canvas.getContext('2d')
    const w = video.videoWidth || 640
    const h = video.videoHeight || 480
    canvas.width = w
    canvas.height = h
    ctx.drawImage(video, 0, 0, w, h)
  }

  // ==========================
  // Inferencia con TF.js (modelo TM por debajo)
  // ==========================
  async function infer () {
    if (!model) return
    const tf = window.tf
    if (!tf) {
      console.warn('tf global no disponible')
      return
    }

    try {
      const t0 = performance.now()
      const { width, height } = imageSize || { width: 224, height: 224 }

      // Creamos el tensor dentro de tidy (no hay await aquí)
      const input = tf.tidy(() => {
        const img = tf.browser.fromPixels(canvas)
        return img
          .resizeBilinear([height, width])
          .toFloat()
          .div(255.0)
          .expandDims(0) // [1,h,w,3]
      })

      // Hacemos la predicción FUERA de tidy
      let out = model.predict(input)
      if (Array.isArray(out)) {
        out = out[0]
      }

      // Leemos datos (esto sí es async)
      const probs = await out.data()

      // Liberar tensores
      input.dispose()
      out.dispose()

      const mapped = labels.map((label, i) => ({
        label,
        prob: probs[i] ?? 0
      })).sort((a, b) => b.prob - a.prob)

      const best = mapped[0]
      top1.textContent = best ? best.label : '—'
      top1p.textContent = best ? fmt(best.prob) : ''
      renderTopK(top3, mapped.slice(0, 3))
      lat.textContent = Math.round(performance.now() - t0)
    } catch (e) {
      console.error('Error en inferencia de imágenes', e)
    }
  }

  // ==========================
  // Bucle principal
  // ==========================
  function startLoop () {
    const intervalMs = 1000 / fps
    let last = 0

    const loop = (now) => {
      if (!running) return
      if (!frozen) {
        drawFromVideo()
      }
      if (now - last >= intervalMs) {
        last = now
        infer()
      }
      loopId = requestAnimationFrame(loop)
    }

    loopId = requestAnimationFrame(loop)
  }

  // ==========================
  // Handlers UI
  // ==========================
    btnFreeze.addEventListener('click', () => {
      frozen = !frozen

      if (frozen) {
        // Modo CONGELADO:
        // - No actualizamos el canvas desde la cámara (ya lo controla el loop con `if (!frozen)`)
        // - Ocultamos el video para que solo se vea el canvas fijo
        btnFreeze.textContent = 'Descongelar'
        video.classList.add('hidden')
      } else {
        // Modo NORMAL:
        btnFreeze.textContent = 'Congelar fotograma'

        // Si no hay stream (veníamos de imagen subida), reactivamos la cámara
        if (!stream) {
          startCam()
        } else {
          // Si ya hay stream, simplemente mostramos el video otra vez
          video.classList.remove('hidden')
        }
      }
    })



  fpsSlider.addEventListener('input', () => {
    fps = Number(fpsSlider.value) || 30
    fpsVal.textContent = String(fps)
  })

    inputUpload.addEventListener('change', (ev) => {
    const file = ev.target.files?.[0]
    if (!file) return
    const img = new Image()
    img.onload = () => {
      // 👇 Apagamos la cámara y ocultamos el video para que no tape el canvas
      stopCam()
      video.classList.add('hidden')

      const ctx = canvas.getContext('2d')
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      frozen = true
      btnFreeze.textContent = 'Descongelar'
      infer() // inferencia directa sobre la imagen subida
    }
    img.src = URL.createObjectURL(file)
  })


  // ==========================
  // Inicialización
  // ==========================
  loadModel()
  startCam()

  // Cleanup para cuando se desmonta el panel
  return () => {
    stopCam()
  }
}
