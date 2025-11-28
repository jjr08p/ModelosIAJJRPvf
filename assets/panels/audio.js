import { setStatus, renderTopK, fmt } from '../ui.js'
import { loadTMAudioModel } from '../modelLoader.js'

// Por defecto (igual que en Colab)
// Etiquetas internas (las que usa el modelo)
// Etiquetas internas (las que usa el modelo)
const DEFAULT_CLASSES = ['espanol', 'ingles', 'aleman']

// Cómo se muestran en pantalla
const DISPLAY_LABELS = {
  espanol: 'Español 🇪🇸',
  ingles:  'Inglés 🇬🇧',
  aleman:  'Alemán 🇩🇪'
}


const DEFAULT_SAMPLE_RATE = 16000
const DEFAULT_DURATION = 5.0

export function initAudio () {
  const canvas = document.getElementById('aud-canvas')
  const st = document.getElementById('aud-status')
  const btnStart = document.getElementById('aud-start')
  const btnStop = document.getElementById('aud-stop')
  const fileInput = document.getElementById('aud-upload')
  const top1 = document.getElementById('aud-top1')
  const top1p = document.getElementById('aud-top1prob')
  const top3 = document.getElementById('aud-top3')
  const lat = document.getElementById('aud-latency')

  let model = null
  let labels = DEFAULT_CLASSES.slice()
  let audioCfg = { sampleRate: DEFAULT_SAMPLE_RATE, duration: DEFAULT_DURATION }

  let mediaStream = null
  let mediaRecorder = null
  let chunks = []
  let recording = false

  // 🔍 Visualización en vivo
  let visCtx = null
  let visAnalyser = null
  let visData = null
  let visAnimId = null

  // ======== Dibujo de onda a partir de float32 (para la señal final) ========
  function drawWaveFloat (wave) {
    if (!wave || !wave.length) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width = 640
    const h = canvas.height = 160
    ctx.clearRect(0, 0, w, h)
    ctx.beginPath()
    const step = wave.length / w
    for (let i = 0; i < w; i++) {
      const idx = Math.floor(i * step)
      const v = wave[idx] || 0
      const y = (0.5 - v / 2) * h
      if (i === 0) ctx.moveTo(i, y); else ctx.lineTo(i, y)
    }
    ctx.lineWidth = 2
    ctx.strokeStyle = getComputedStyle(canvas).color
    ctx.stroke()
  }

  // ======== Dibujo de onda en vivo (byteTime) ========
  function drawOscLive (byteTime) {
    const ctx = canvas.getContext('2d')
    const w = canvas.width = 640
    const h = canvas.height = 160
    ctx.clearRect(0, 0, w, h)
    ctx.beginPath()
    const len = byteTime.length
    const step = w / len
    for (let i = 0; i < len; i++) {
      const x = i * step
      const v = byteTime[i] / 255 // 0..1
      const y = v * h
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
    }
    ctx.lineWidth = 2
    ctx.strokeStyle = getComputedStyle(canvas).color
    ctx.stroke()
  }

  function startLiveVis (stream) {
    const AudioContext = window.AudioContext || window.webkitAudioContext
    visCtx = new AudioContext()
    const src = visCtx.createMediaStreamSource(stream)
    visAnalyser = visCtx.createAnalyser()
    visAnalyser.fftSize = 1024
    src.connect(visAnalyser)
    visData = new Uint8Array(visAnalyser.fftSize)

    const loop = () => {
      if (!visAnalyser) return
      visAnalyser.getByteTimeDomainData(visData)
      drawOscLive(visData)
      visAnimId = requestAnimationFrame(loop)
    }
    loop()
  }

  function stopLiveVis () {
    if (visAnimId != null) cancelAnimationFrame(visAnimId)
    visAnimId = null
    if (visCtx) {
      try { visCtx.close() } catch (e) {}
    }
    visCtx = null
    visAnalyser = null
    visData = null
  }

  // ======== Reescalar a longitud fija ========
  function resampleToFixedLength (data, targetLength) {
    const out = new Float32Array(targetLength)
    const len = data.length
    if (len === 0) return out
    for (let i = 0; i < targetLength; i++) {
      const idx = Math.floor(i * len / targetLength)
      out[i] = data[idx] || 0
    }
    return out
  }

  // ======== Espectrograma (igual que en Colab) ========
  function computeSpectrogramTensor (tf, audioFloats, numSamples) {
    const frameLength = 255
    const frameStep = 128

    return tf.tidy(() => {
      let audio = tf.tensor1d(audioFloats)

      if (audio.shape[0] > numSamples) {
        audio = audio.slice(0, numSamples)
      } else if (audio.shape[0] < numSamples) {
        const pad = numSamples - audio.shape[0]
        audio = audio.pad([[0, pad]])
      }
      audio = audio.reshape([numSamples])

      const stft = tf.signal.stft(audio, frameLength, frameStep)
      let spec = stft.abs()
      spec = spec.add(1e-10).log()

      spec = spec.expandDims(-1) // [time, freq, 1]
      spec = spec.expandDims(0)  // [1, time, freq, 1]
      return spec
    })
  }

  // ======== Inferencia ========
  async function runInferenceOnBuffer (audioBuffer) {
    const tf = window.tf
    if (!tf || !tf.signal || !tf.signal.stft) {
      throw new Error('tf.signal.stft no disponible en tfjs')
    }
    if (!model) {
      throw new Error('Modelo de audio no cargado')
    }

    const numSamples = Math.floor((audioCfg.sampleRate || DEFAULT_SAMPLE_RATE) *
                                 (audioCfg.duration   || DEFAULT_DURATION))

    const channelData = audioBuffer.getChannelData(0)
    const resampled = resampleToFixedLength(channelData, numSamples)

    // Onda final (la que queda dibujada tras la grabación)
    drawWaveFloat(resampled)

    const t0 = performance.now()
    const input = computeSpectrogramTensor(tf, resampled, numSamples)
    let out
    try {
      out = model.predict(input)
      if (Array.isArray(out)) out = out[0]

      const probs = await out.data()
      const mapped = labels.map((labelKey, i) => ({
    label: DISPLAY_LABELS[labelKey] ?? labelKey,
    prob: probs[i] ?? 0
    })).sort((a, b) => b.prob - a.prob)



      const best = mapped[0]
      top1.textContent = best ? best.label : '—'
      top1p.textContent = best ? fmt(best.prob) : ''
      renderTopK(top3, mapped.slice(0, 3))
      lat.textContent = Math.round(performance.now() - t0)
    } finally {
      input.dispose()
      if (out && out.dispose) out.dispose()
    }
  }

  // ======== Grabación (MediaRecorder) ========
  async function startRecording () {
    if (recording) return
    if (!model) {
      console.error('Modelo de audio no listo')
      setStatus(st, 'error')
      return
    }

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    } catch (e) {
      console.error('Permiso de micrófono denegado', e)
      setStatus(st, 'denied')
      return
    }

    mediaStream = stream
    chunks = []

    // 🔊 Empezar visualización en vivo
    startLiveVis(mediaStream)

    const AudioContext = window.AudioContext || window.webkitAudioContext
    const decodeCtx = new AudioContext()

    mediaRecorder = new MediaRecorder(mediaStream)
    mediaRecorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data)
    }

    mediaRecorder.onstop = async () => {
      try {
        stopLiveVis()  // parar oscilo en vivo

        const blob = new Blob(chunks, { type: 'audio/webm' })

        // ▶️ Reproducir lo grabado
        try {
          const playUrl = URL.createObjectURL(blob)
          const player = new Audio(playUrl)
          player.play()
        } catch (e) {
          console.warn('No se pudo reproducir el audio grabado:', e)
        }

        // Inferencia
        const arrayBuffer = await blob.arrayBuffer()
        const buffer = await decodeCtx.decodeAudioData(arrayBuffer)
        await runInferenceOnBuffer(buffer)

        setStatus(st, 'ready')
      } catch (e) {
        console.error('Error procesando grabación:', e)
        setStatus(st, 'error')
      } finally {
        if (mediaStream) {
          mediaStream.getTracks().forEach(t => t.stop())
        }
        mediaStream = null
        recording = false
        decodeCtx.close()
      }
    }

    setStatus(st, 'recording')
    top1.textContent = '—'
    top1p.textContent = ''
    top3.innerHTML = ''
    lat.textContent = '—'

    mediaRecorder.start()
    recording = true

    const maxMs = (audioCfg.duration ? audioCfg.duration * 1000 : 4000)
    setTimeout(() => {
      if (recording && mediaRecorder.state === 'recording') {
        setStatus(st, 'processing')
        mediaRecorder.stop()
      }
    }, maxMs)
  }

  function stopRecording () {
    if (mediaRecorder && recording && mediaRecorder.state === 'recording') {
      setStatus(st, 'processing')
      mediaRecorder.stop()
    }
  }

  // Clips de prueba (solo reproducen sonido)
  document.querySelectorAll('.aud-clip').forEach(btn => {
    btn.addEventListener('click', async () => {
      try { await new Audio(btn.dataset.clip).play() } catch (e) {
        console.warn('Error reproduciendo clip:', e)
      }
    })
  })

  // ⬆️ ya estaba (clips de prueba)

// ⬇️ NUEVO: subir audio desde archivo
if (fileInput) {
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0]
    if (!file || !model) return

    const AudioContext = window.AudioContext || window.webkitAudioContext
    const audioCtx = new AudioContext()

    try {
      setStatus(st, 'processing')
      top1.textContent = '—'
      top1p.textContent = ''
      top3.innerHTML = ''
      lat.textContent = '—'

      const arrayBuffer = await file.arrayBuffer()
      const buffer = await audioCtx.decodeAudioData(arrayBuffer)
      await runInferenceOnBuffer(buffer)
      setStatus(st, 'ready')
    } catch (err) {
      console.error('Error al procesar audio subido:', err)
      setStatus(st, 'error')
    } finally {
      audioCtx.close()
      fileInput.value = ''
    }
  })
}




  btnStart.addEventListener('click', startRecording)
  btnStop.addEventListener('click', stopRecording)

  // ======== Cargar modelo ========
  ;(async () => {
    setStatus(st, 'loading')
    try {
      const loaded = await loadTMAudioModel('ModelosAI/Sonidos')
      model = loaded.model
      if (loaded.labels && loaded.labels.length) labels = loaded.labels
      if (loaded.audioConfig) audioCfg = loaded.audioConfig
      setStatus(st, 'ready')
      console.log('✅ Modelo de audio listo. Clases:', labels, 'cfg:', audioCfg)
    } catch (e) {
      console.error('❌ Error cargando modelo de audio:', e)
      setStatus(st, 'error')
      btnStart.disabled = true
      btnStop.disabled = true
    }
  })()

  // Cleanup
  return () => {
    stopRecording()
    stopLiveVis()
  }
}
