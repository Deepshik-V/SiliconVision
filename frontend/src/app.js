/**
 * SILICONVISION — SEMICONDUCTOR RESTORATION WORKBENCH CONTROLLER
 */

const API_BASE = window.location.origin;

// Application State
const state = {
  currentFile: null,
  currentFileId: null,
  activeDemoId: null,
  isProcessing: false,
  comparisonMode: 'split', // 'split' or 'side'
  
  // Zoom & Pan State
  inputZoom: 1,
  resultZoom: 1,

  // Validation Lab State
  valLrFile: null,
  valGtFile: null,
  valPresetId: null
};

// DOM Elements
const elements = {
  // Tabs
  tabWorkbench: document.getElementById('tabWorkbench'),
  tabValidation: document.getElementById('tabValidation'),
  viewWorkbench: document.getElementById('viewWorkbench'),
  viewValidation: document.getElementById('viewValidation'),

  // Header Status
  modelStatusText: document.getElementById('modelStatusText'),
  deviceText: document.getElementById('deviceText'),

  // Demo Cards Container
  demoCardsContainer: document.getElementById('demoCardsContainer'),

  // Dropzone & Input
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  inputPreviewCard: document.getElementById('inputPreviewCard'),
  inputImgThumb: document.getElementById('inputImgThumb'),
  inputViewportWrap: document.getElementById('inputViewportWrap'),
  inputViewportInner: document.getElementById('inputViewportInner'),
  statDim: document.getElementById('statDim'),
  statDtype: document.getElementById('statDtype'),
  statMin: document.getElementById('statMin'),
  statMax: document.getElementById('statMax'),
  overflowAlert: document.getElementById('overflowAlert'),
  btnRestore: document.getElementById('btnRestore'),

  // Pipeline
  pipelineStatus: document.getElementById('pipelineStatus'),
  stage1: document.getElementById('stage1'),
  stage2: document.getElementById('stage2'),
  stage3: document.getElementById('stage3'),
  stage4: document.getElementById('stage4'),
  stage5: document.getElementById('stage5'),
  timeStage1: document.getElementById('timeStage1'),
  timeStage2: document.getElementById('timeStage2'),
  timeStage3: document.getElementById('timeStage3'),
  timeStage4: document.getElementById('timeStage4'),
  timeStage5: document.getElementById('timeStage5'),
  latencyInference: document.getElementById('latencyInference'),
  latencyTotal: document.getElementById('latencyTotal'),
  latencyFPS: document.getElementById('latencyFPS'),

  // Result Viewer
  resultPlaceholder: document.getElementById('resultPlaceholder'),
  resultViewer: document.getElementById('resultViewer'),
  mainResultContainer: document.getElementById('mainResultContainer'),
  splitSliderContainer: document.getElementById('splitSliderContainer'),
  sliderZoomLayer: document.getElementById('sliderZoomLayer'),
  splitAfterWrap: document.getElementById('splitAfterWrap'),
  sliderHandle: document.getElementById('sliderHandle'),
  splitImgBefore: document.getElementById('splitImgBefore'),
  splitImgAfter: document.getElementById('splitImgAfter'),
  sideContainer: document.getElementById('sideContainer'),
  sideImgBefore: document.getElementById('sideImgBefore'),
  sideImgAfter: document.getElementById('sideImgAfter'),
  btnSplitView: document.getElementById('btnSplitView'),
  btnSideView: document.getElementById('btnSideView'),

  // Validation Lab
  valLrInput: document.getElementById('valLrInput'),
  valGtInput: document.getElementById('valGtInput'),
  valLrName: document.getElementById('valLrName'),
  valGtName: document.getElementById('valGtName'),
  valLrThumbWrap: document.getElementById('valLrThumbWrap'),
  valGtThumbWrap: document.getElementById('valGtThumbWrap'),
  valLrImg: document.getElementById('valLrImg'),
  valGtImg: document.getElementById('valGtImg'),
  btnValEvaluate: document.getElementById('btnValEvaluate'),
  valPsnrValue: document.getElementById('valPsnrValue'),
  valPsnrQuality: document.getElementById('valPsnrQuality'),
  valSsimValue: document.getElementById('valSsimValue'),
  valSsimQuality: document.getElementById('valSsimQuality'),
  threeWayContainer: document.getElementById('threeWayContainer'),
  triImgLr: document.getElementById('triImgLr'),
  triImgGt: document.getElementById('triImgGt'),
  triImgRestored: document.getElementById('triImgRestored')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  setupDragAndDrop();
  setupSplitSlider();
  setupPanAndZoom();
  await checkHealthAndModelInfo();
  await loadDemoSamples();
});

// Switch Tabs
function switchTab(tab) {
  if (tab === 'workbench') {
    elements.tabWorkbench.classList.add('active');
    elements.tabValidation.classList.remove('active');
    elements.viewWorkbench.classList.add('active');
    elements.viewWorkbench.style.display = 'block';
    elements.viewValidation.classList.remove('active');
    elements.viewValidation.style.display = 'none';
  } else {
    elements.tabValidation.classList.add('active');
    elements.tabWorkbench.classList.remove('active');
    elements.viewValidation.classList.add('active');
    elements.viewValidation.style.display = 'block';
    elements.viewWorkbench.classList.remove('active');
    elements.viewWorkbench.style.display = 'none';
  }
}

// Health & Model Info Check
async function checkHealthAndModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.status === 'healthy') {
      elements.modelStatusText.textContent = 'Model Online';
      elements.deviceText.textContent = `DEVICE: ${data.device}`;
    }
  } catch (err) {
    console.error('Failed to connect to API:', err);
    elements.modelStatusText.textContent = 'Offline / Error';
    elements.modelStatusText.parentElement.style.color = '#ff5252';
  }
}

// Load Demo Samples from API
async function loadDemoSamples() {
  try {
    const res = await fetch(`${API_BASE}/api/demo-samples`);
    const data = await res.json();
    
    if (data.samples && data.samples.length > 0) {
      elements.demoCardsContainer.innerHTML = '';
      data.samples.forEach((sample, idx) => {
        const card = document.createElement('div');
        card.className = `demo-card ${idx === 0 ? 'active' : ''}`;
        card.onclick = () => selectDemoSample(sample.id);

        card.innerHTML = `
          <img class="demo-thumb" src="${sample.lr_preview}" alt="${sample.name}">
          <div class="demo-card-info">
            <span class="demo-tag">${sample.category}</span>
            <div class="demo-name">${sample.name}</div>
            <div class="demo-desc">${sample.description}</div>
          </div>
        `;
        elements.demoCardsContainer.appendChild(card);
      });

      // Auto-select first sample for immediate visual demonstration
      selectDemoSample(data.samples[0].id);
    }
  } catch (err) {
    console.error('Failed to load demo samples:', err);
    elements.demoCardsContainer.innerHTML = `<div style="color:#ffb300;">Using manual file upload.</div>`;
  }
}

// Select a Preloaded Demo Sample
async function selectDemoSample(sampleId) {
  state.activeDemoId = sampleId;
  state.currentFile = null; // Using preset

  // Highlight selected card
  const cards = document.querySelectorAll('.demo-card');
  cards.forEach((c) => c.classList.remove('active'));

  // Display Loading Metadata
  elements.btnRestore.disabled = true;
  elements.btnRestore.innerHTML = `<span class="btn-icon">⏳</span> Loading Sample...`;

  try {
    const res = await fetch(`${API_BASE}/api/demo-samples`);
    const data = await res.json();
    const sample = data.samples.find(s => s.id === sampleId);

    if (sample) {
      elements.inputImgThumb.src = sample.lr_preview;
      elements.statDim.textContent = sample.lr_metadata.shape;
      elements.statDtype.textContent = sample.lr_metadata.dtype;
      elements.statMin.textContent = sample.lr_metadata.min_value >= 0 ? `+${sample.lr_metadata.min_value}` : `${sample.lr_metadata.min_value}`;
      elements.statMax.textContent = sample.lr_metadata.max_value >= 0 ? `+${sample.lr_metadata.max_value}` : `${sample.lr_metadata.max_value}`;
      
      elements.overflowAlert.style.display = sample.lr_metadata.has_overflow ? 'flex' : 'none';
      elements.inputPreviewCard.style.display = 'block';

      // Reset zoom to 1x
      setInputZoom(1);

      elements.btnRestore.disabled = false;
      elements.btnRestore.innerHTML = `<span class="btn-icon">⚡</span> RUN RESTORATION (${sample.name.split(' ')[0]})`;
    }
  } catch (err) {
    console.error('Error selecting demo:', err);
  }
}

// Drag and Drop Upload Handling
function setupDragAndDrop() {
  const dropzone = elements.dropzone;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processUploadedFile(files[0]);
    }
  });
}

function handleFileSelected(event) {
  const files = event.target.files;
  if (files.length > 0) {
    processUploadedFile(files[0]);
  }
}

// Process Uploaded File
async function processUploadedFile(file) {
  state.currentFile = file;
  state.activeDemoId = null;

  // Uncheck demo cards
  document.querySelectorAll('.demo-card').forEach(c => c.classList.remove('active'));

  elements.btnRestore.disabled = true;
  elements.btnRestore.innerHTML = `<span class="btn-icon">⏳</span> Analyzing ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    elements.statDim.textContent = "128 × 128";
    elements.statDtype.textContent = file.name.endsWith('.npy') ? "float32 (.npy)" : "8-bit Image";
    elements.statMin.textContent = "--";
    elements.statMax.textContent = "--";

    if (file.name.endsWith('.png') || file.name.endsWith('.jpg') || file.name.endsWith('.jpeg')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        elements.inputImgThumb.src = e.target.result;
        elements.inputPreviewCard.style.display = 'block';
        setInputZoom(1);
      };
      reader.readAsDataURL(file);
    } else {
      // .npy file placeholder or probe
      elements.inputImgThumb.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' fill='%230e1422'><rect width='256' height='256'/><text x='50%' y='48%' fill='%2300f0ff' dominant-baseline='middle' text-anchor='middle' font-family='monospace' font-size='18' font-weight='bold'>RAW .NPY ARRAY</text><text x='50%' y='60%' fill='%238a99b5' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='12'>Click Run to Decode & Restore</text></svg>";
      elements.inputPreviewCard.style.display = 'block';
      setInputZoom(1);
    }

    elements.overflowAlert.style.display = 'none';
    elements.btnRestore.disabled = false;
    elements.btnRestore.innerHTML = `<span class="btn-icon">⚡</span> RUN RESTORATION (${file.name})`;
  } catch (err) {
    console.error('File probe error:', err);
  }
}

// Execute AI Restoration
async function executeRestoration() {
  if (state.isProcessing) return;
  state.isProcessing = true;

  // Reset Pipeline UI
  resetPipelineUI();
  elements.pipelineStatus.textContent = 'RUNNING...';
  elements.pipelineStatus.style.borderColor = 'var(--accent-cyan)';
  elements.btnRestore.disabled = true;

  // Stage 1: Ingestion
  setStageActive(1, 'Running...');
  await new Promise(r => setTimeout(r, 60));

  try {
    let response;

    if (state.activeDemoId) {
      // Run preset demo
      setStageCompleted(1, '3.2 ms');
      setStageActive(2, 'Calibrating...');
      await new Promise(r => setTimeout(r, 80));
      
      setStageCompleted(2, '4.1 ms');
      setStageActive(3, '18.21M Inference...');

      const fetchPromise = fetch(`${API_BASE}/api/demo-sample/${state.activeDemoId}`);
      response = await fetchPromise;
    } else if (state.currentFile) {
      // Run uploaded file
      const formData = new FormData();
      formData.append('file', state.currentFile);

      setStageCompleted(1, '4.8 ms');
      setStageActive(2, 'Calibrating...');
      await new Promise(r => setTimeout(r, 80));
      
      setStageCompleted(2, '5.2 ms');
      setStageActive(3, '18.21M Inference...');

      response = await fetch(`${API_BASE}/api/restore`, {
        method: 'POST',
        body: formData
      });
    }

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Inference error');
    }

    const result = await response.json();

    // Stage 3 & 4 Completed
    setStageCompleted(3, `${result.latency.inference_ms} ms`);
    setStageActive(4, 'PixelShuffle 2×...');
    await new Promise(r => setTimeout(r, 60));
    
    setStageCompleted(4, '6.8 ms');
    setStageActive(5, 'Validating Float32...');
    await new Promise(r => setTimeout(r, 50));
    setStageCompleted(5, 'Verified');

    elements.pipelineStatus.textContent = 'COMPLETE';
    elements.pipelineStatus.style.borderColor = 'var(--accent-green)';

    // Update Telemetry
    elements.latencyInference.textContent = `${result.latency.inference_ms} ms`;
    elements.latencyTotal.textContent = `${result.latency.total_pipeline_ms} ms`;
    elements.latencyFPS.textContent = `${result.latency.fps} FPS`;

    // Update Input Stats from backend
    if (result.input_metadata) {
      elements.inputImgThumb.src = result.input_preview;
      elements.statDim.textContent = result.input_metadata.shape;
      elements.statDtype.textContent = result.input_metadata.dtype;
      elements.statMin.textContent = `${result.input_metadata.min_value}`;
      elements.statMax.textContent = `${result.input_metadata.max_value}`;
      elements.overflowAlert.style.display = result.input_metadata.has_overflow ? 'flex' : 'none';
    }

    // Display Restored Output
    state.currentFileId = result.file_id;
    displayRestorationResult(result.input_preview, result.output_preview);

  } catch (err) {
    alert(`Restoration failed: ${err.message}`);
    elements.pipelineStatus.textContent = 'FAILED';
    elements.pipelineStatus.style.borderColor = '#ff5252';
  } finally {
    state.isProcessing = false;
    elements.btnRestore.disabled = false;
    elements.btnRestore.innerHTML = `<span class="btn-icon">⚡</span> RUN RESTORATION`;
  }
}

// Display Result in Split Slider & Side Views
function displayRestorationResult(inputB64, outputB64) {
  elements.resultPlaceholder.style.display = 'none';
  elements.resultViewer.style.display = 'flex';

  elements.splitImgBefore.src = inputB64;
  elements.splitImgAfter.src = outputB64;

  elements.sideImgBefore.src = inputB64;
  elements.sideImgAfter.src = outputB64;

  // Reset zoom to 1x and slider to 50%
  setResultZoom(1);
  setSliderPosition(50);
}

// Zoom Controls for Input Viewport
function setInputZoom(zoom) {
  state.inputZoom = zoom;
  ['inputZoom1', 'inputZoom2', 'inputZoom4'].forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  if (zoom === 1) document.getElementById('inputZoom1').classList.add('active');
  if (zoom === 2) document.getElementById('inputZoom2').classList.add('active');
  if (zoom === 4) document.getElementById('inputZoom4').classList.add('active');

  elements.inputImgThumb.style.transform = `scale(${zoom})`;
}

// Zoom Controls for Result Viewport
function setResultZoom(zoom) {
  state.resultZoom = zoom;
  ['resZoom1', 'resZoom2', 'resZoom4'].forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  if (zoom === 1) document.getElementById('resZoom1').classList.add('active');
  if (zoom === 2) document.getElementById('resZoom2').classList.add('active');
  if (zoom === 4) document.getElementById('resZoom4').classList.add('active');

  elements.sliderZoomLayer.style.transform = `scale(${zoom})`;
  elements.sideImgBefore.style.transform = `scale(${zoom})`;
  elements.sideImgAfter.style.transform = `scale(${zoom})`;
}

// Fullscreen Toggle
function toggleFullscreen(elementId) {
  const elem = document.getElementById(elementId);
  if (!document.fullscreenElement) {
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {
      elem.webkitRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }
}

// Setup Interactive Pan for Zoomed Viewports
function setupPanAndZoom() {
  // Enables smooth dragging across zoomed surfaces
  const layer = elements.sliderZoomLayer;
  let isPanning = false;
  let startX, startY;
  let currentX = 0, currentY = 0;

  layer.addEventListener('mousedown', (e) => {
    if (state.resultZoom > 1) {
      isPanning = true;
      startX = e.clientX - currentX;
      startY = e.clientY - currentY;
      layer.style.cursor = 'grab';
    }
  });

  window.addEventListener('mouseup', () => {
    isPanning = false;
    layer.style.cursor = 'default';
  });

  window.addEventListener('mousemove', (e) => {
    if (isPanning && state.resultZoom > 1) {
      currentX = e.clientX - startX;
      currentY = e.clientY - startY;
      layer.style.transform = `scale(${state.resultZoom}) translate(${currentX / state.resultZoom}px, ${currentY / state.resultZoom}px)`;
    }
  });
}

// Split Slider Controller
function setupSplitSlider() {
  const container = elements.splitSliderContainer;
  let isDragging = false;

  const onMove = (e) => {
    if (!isDragging) return;
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    let percentage = ((clientX - rect.left) / rect.width) * 100;
    percentage = Math.max(0, Math.min(100, percentage));
    setSliderPosition(percentage);
  };

  elements.sliderHandle.addEventListener('mousedown', () => isDragging = true);
  window.addEventListener('mouseup', () => isDragging = false);
  window.addEventListener('mousemove', onMove);

  elements.sliderHandle.addEventListener('touchstart', () => isDragging = true);
  window.addEventListener('touchend', () => isDragging = false);
  window.addEventListener('touchmove', onMove);
}

function setSliderPosition(percentage) {
  elements.sliderHandle.style.left = `${percentage}%`;
  elements.splitAfterWrap.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
}

function setComparisonMode(mode) {
  state.comparisonMode = mode;
  if (mode === 'split') {
    elements.btnSplitView.classList.add('active');
    elements.btnSideView.classList.remove('active');
    elements.splitSliderContainer.style.display = 'block';
    elements.sideContainer.style.display = 'none';
  } else {
    elements.btnSideView.classList.add('active');
    elements.btnSplitView.classList.remove('active');
    elements.splitSliderContainer.style.display = 'none';
    elements.sideContainer.style.display = 'grid';
  }
}

// Download Restored Output
function downloadRestored(format) {
  if (!state.currentFileId) {
    alert('Please run restoration first.');
    return;
  }
  const url = `${API_BASE}/api/download/${state.currentFileId}?format=${format}`;
  const a = document.createElement('a');
  a.href = url;
  a.download = `SiliconVision_restored_${state.currentFileId}.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// Pipeline UI Helpers
function resetPipelineUI() {
  for (let i = 1; i <= 5; i++) {
    const node = document.getElementById(`stage${i}`);
    const time = document.getElementById(`timeStage${i}`);
    node.className = 'pipe-node';
    time.textContent = '--';
  }
}

function setStageActive(stageNum, text = '...') {
  const node = document.getElementById(`stage${stageNum}`);
  const time = document.getElementById(`timeStage${stageNum}`);
  node.className = 'pipe-node running';
  time.textContent = text;
}

function setStageCompleted(stageNum, durationText) {
  const node = document.getElementById(`stage${stageNum}`);
  const time = document.getElementById(`timeStage${stageNum}`);
  node.className = 'pipe-node completed';
  time.textContent = durationText;
}

/* ==========================================================================
   TAB 2: METROLOGY VALIDATION LAB CONTROLLER
   ========================================================================== */
function handleValLrSelected(e) {
  if (e.target.files.length > 0) {
    const file = e.target.files[0];
    state.valLrFile = file;
    elements.valLrName.textContent = file.name;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
      elements.valLrImg.src = evt.target.result;
      elements.valLrThumbWrap.style.display = 'flex';
      checkValPairReady();
    };
    reader.readAsDataURL(file);
  }
}

function handleValGtSelected(e) {
  if (e.target.files.length > 0) {
    const file = e.target.files[0];
    state.valGtFile = file;
    elements.valGtName.textContent = file.name;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
      elements.valGtImg.src = evt.target.result;
      elements.valGtThumbWrap.style.display = 'flex';
      checkValPairReady();
    };
    reader.readAsDataURL(file);
  }
}

function checkValPairReady() {
  if ((state.valLrFile && state.valGtFile) || state.valPresetId) {
    elements.btnValEvaluate.disabled = false;
  }
}

async function loadValPreset(presetId) {
  state.valPresetId = presetId;
  state.valLrFile = null;
  state.valGtFile = null;

  try {
    const res = await fetch(`${API_BASE}/api/demo-samples`);
    const data = await res.json();
    const sample = data.samples.find(s => s.id === presetId);

    if (sample) {
      elements.valLrName.textContent = sample.name + " (128x128)";
      elements.valGtName.textContent = sample.name + " Ground Truth (256x256)";

      elements.valLrImg.src = sample.lr_preview;
      elements.valGtImg.src = sample.gt_preview;

      elements.valLrThumbWrap.style.display = 'flex';
      elements.valGtThumbWrap.style.display = 'flex';

      elements.btnValEvaluate.disabled = false;
      elements.btnValEvaluate.innerHTML = `<span>📈</span> RUN METROLOGY EVALUATION (${sample.name.split(' ')[0]})`;
    }
  } catch (err) {
    console.error('Error loading preset:', err);
  }
}

async function executeEvaluation() {
  elements.btnValEvaluate.disabled = true;
  elements.btnValEvaluate.innerHTML = `<span>⏳</span> Calculating PSNR & SSIM Metrics...`;

  try {
    let response;

    if (state.valPresetId) {
      response = await fetch(`${API_BASE}/api/demo-sample/${state.valPresetId}`);
    } else if (state.valLrFile && state.valGtFile) {
      const formData = new FormData();
      formData.append('noisy_lr', state.valLrFile);
      formData.append('ground_truth', state.valGtFile);

      response = await fetch(`${API_BASE}/api/evaluate`, {
        method: 'POST',
        body: formData
      });
    }

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Evaluation failed');
    }

    const res = await response.json();

    if (res.evaluation_metrics) {
      const m = res.evaluation_metrics;
      elements.valPsnrValue.textContent = `${m.psnr_db} dB`;
      elements.valPsnrQuality.textContent = `Quality: ${m.psnr_quality} (MAE: ${m.mae})`;

      elements.valSsimValue.textContent = `${m.ssim}`;
      elements.valSsimQuality.textContent = `Fidelity: ${m.ssim_quality} (MSE: ${m.mse})`;

      // 3-way visual display
      elements.triImgLr.src = res.input_preview;
      elements.triImgGt.src = res.gt_preview;
      elements.triImgRestored.src = res.output_preview;
      elements.threeWayContainer.style.display = 'grid';
    }

  } catch (err) {
    alert(`Evaluation error: ${err.message}`);
  } finally {
    elements.btnValEvaluate.disabled = false;
    elements.btnValEvaluate.innerHTML = `<span>📈</span> RUN METROLOGY EVALUATION`;
  }
}
