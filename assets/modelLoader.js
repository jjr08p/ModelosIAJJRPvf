// ========================
// IMÁGENES
// ========================
export async function loadTMImageModel(basePath) {
  const modelURL = `${basePath}/model.json`
  const metadataURL = `${basePath}/metadata.json`

  console.log('Cargando modelo de imágenes desde', basePath)

  let model
  let labels = []

  // 1) Intentar usar Teachable Machine si está disponible
  if (window.tmImage && typeof window.tmImage.load === 'function') {
    console.log('Usando tmImage.load(...)')
    model = await window.tmImage.load(modelURL, metadataURL)
    if (typeof model.getClassLabels === 'function') {
      labels = await model.getClassLabels()
    } else if (model.metadata?.labels) {
      labels = model.metadata.labels
    }
  }
  // 2) Fallback: usar directamente TensorFlow.js
  else if (window.tf && typeof window.tf.loadLayersModel === 'function') {
    console.warn('tmImage no disponible, usando tf.loadLayersModel como fallback')
    model = await window.tf.loadLayersModel(modelURL)
    try {
      const res = await fetch(metadataURL)
      if (res.ok) {
        const meta = await res.json()
        if (Array.isArray(meta.labels)) {
          labels = meta.labels
        }
      }
    } catch (e) {
      console.warn('No se pudieron leer las etiquetas desde metadata.json', e)
    }
  } else {
    throw new Error('Ni tmImage ni tf.loadLayersModel están disponibles')
  }

  return { model, labels }
}

// ========================
// AUDIO (tfjs puro, sin tmAudio)
// ========================
export async function loadTMAudioModel(basePath) {
  const tf = window.tf
  if (!tf || !tf.loadLayersModel) {
    throw new Error('tfjs no disponible para audio')
  }

  const modelURL = `${basePath}/model.json`
  const metadataURL = `${basePath}/metadata.json`

  console.log('Cargando modelo de audio TFJS desde:', modelURL)

  const model = await tf.loadLayersModel(modelURL)

  let labels = []
  let sampleRate = 16000
  let duration = 5.0

  try {
    const res = await fetch(metadataURL)
    if (res.ok) {
      const meta = await res.json()
      if (Array.isArray(meta.labels)) labels = meta.labels
      if (typeof meta.sampleRate === 'number') sampleRate = meta.sampleRate
      if (typeof meta.duration === 'number') duration = meta.duration
    }
  } catch (e) {
    console.warn('No se pudo leer metadata del audio, usando defaults.', e)
  }

  return { model, labels, audioConfig: { sampleRate, duration } }
}




export async function loadTMPoseModel(basePath) {
  const tf = window.tf
  if (!tf || !tf.loadLayersModel) {
    throw new Error('tfjs no disponible para posturas')
  }

  const modelURL = `${basePath}/model.json`
  const metadataURL = `${basePath}/metadata.json`

  console.log('Cargando modelo de posturas TFJS desde:', modelURL)

  const model = await tf.loadLayersModel(modelURL)

  let labels = []
  let imgHeight = 160
  let imgWidth = 160

  try {
    const res = await fetch(metadataURL)
    if (res.ok) {
      const meta = await res.json()
      if (Array.isArray(meta.labels)) labels = meta.labels
      if (typeof meta.imgHeight === 'number') imgHeight = meta.imgHeight
      if (typeof meta.imgWidth === 'number') imgWidth = meta.imgWidth
    }
  } catch (e) {
    console.warn('No se pudo leer metadata de posturas, usando defaults.', e)
  }

  if (!labels.length) {
    labels = ['atento', 'no_atento']
  }

  return { model, labels, inputShape: { imgHeight, imgWidth } }
}

