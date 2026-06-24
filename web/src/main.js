import {
  buildDisplayRows,
  buildPredictionRows,
  csvFilename,
  downsampleRgbForInference,
  echotasticFilename,
  fcFilename,
  formatEchotasticFile,
  formatFcFile,
  formatPredictionCsv,
  framerateFromMetadata,
  headerFramerateFromMetadata,
} from "./detections/exports.js";
import { createSegmentationSession, defaultModelUrl, runYoloSegmentation } from "./inference/yoloSegmentation.js";
import { rgbToImageData } from "./render/imageData.js";
import { makeOverlayImage } from "./render/overlay.js";
import { makeEchogramVisualFromBgr } from "./render/visualizeEchogram.js";
import { bgrToRgbImage, decodeSonarBuffer } from "./sonar/decoder.js";

const elements = {
  sonarFile: document.querySelector("#sonar-file"),
  modelFile: document.querySelector("#model-file"),
  useBundledModel: document.querySelector("#use-bundled-model"),
  modelFileField: document.querySelector("#model-file-field"),
  startFrame: document.querySelector("#start-frame"),
  endFrame: document.querySelector("#end-frame"),
  upstreamDirection: document.querySelector("#upstream-direction"),
  backend: document.querySelector("#backend"),
  confidence: document.querySelector("#confidence"),
  iou: document.querySelector("#iou"),
  nativeFps: document.querySelector("#native-fps"),
  inferenceFps: document.querySelector("#inference-fps"),
  decodeButton: document.querySelector("#decode-button"),
  runButton: document.querySelector("#run-button"),
  statusText: document.querySelector("#status-text"),
  progressBar: document.querySelector("#progress-bar"),
  statusTimeLabel: document.querySelector("#status-time-label"),
  statusTimeValue: document.querySelector("#status-time-value"),
  metaList: document.querySelector("#meta-list"),
  decodedSummary: document.querySelector("#decoded-summary"),
  overlaySummary: document.querySelector("#overlay-summary"),
  countsSummary: document.querySelector("#counts-summary"),
  detectionsBody: document.querySelector("#detections-body"),
  decodedCanvas: document.querySelector("#decoded-canvas"),
  overlayCanvas: document.querySelector("#overlay-canvas"),
  downloadDecodedButton: document.querySelector("#download-decoded-button"),
  downloadOverlayButton: document.querySelector("#download-overlay-button"),
  downloadFcButton: document.querySelector("#download-fc-button"),
  downloadCsvButton: document.querySelector("#download-csv-button"),
  downloadEchotasticButton: document.querySelector("#download-echotastic-button"),
  zoomPopup: document.querySelector("#zoom-popup"),
  zoomCanvas: document.querySelector("#zoom-canvas"),
  zoomTitle: document.querySelector("#zoom-title"),
  zoomCoords: document.querySelector("#zoom-coords"),
};

const state = {
  sonarFile: null,
  decoded: null,
  rgbImage: null,
  visualRgbImage: null,
  overlayImage: null,
  decodedImageData: null,
  overlayImageData: null,
  detections: [],
  predictionRows: [],
  displayRows: [],
  frameIndices: null,
  inferenceFrameRate: null,
  inferenceUsedNativeFps: true,
  exportFiles: {
    csvName: null,
    csvText: null,
    fcName: null,
    fcText: null,
    echotasticName: null,
    echotasticText: null,
  },
  modelKey: null,
  session: null,
  running: false,
  timerStartMs: null,
  timerHandle: null,
  zoom: {
    sourceCanvas: null,
    title: "",
    imageX: 0,
    imageY: 0,
    clientX: 0,
    clientY: 0,
  },
};

const DEFAULT_ZOOM_REGION_SIZE = 128;
const MIN_ZOOM_REGION_SIZE = 32;
const MAX_ZOOM_REGION_SIZE = 256;
const ZOOM_REGION_STEP = 16;
const ZOOM_CANVAS_SIZE = 512;
let zoomRegionSize = DEFAULT_ZOOM_REGION_SIZE;

function setStatus(text, progress = null) {
  elements.statusText.textContent = text;
  if (progress !== null) {
    elements.progressBar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

function formatElapsedTime(ms) {
  const totalSeconds = Math.max(0, ms / 1000);
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(1)} s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}m ${seconds.toFixed(1)}s`;
}

function stopStatusTimer(finalLabel = "Time") {
  if (state.timerHandle !== null) {
    window.clearInterval(state.timerHandle);
    state.timerHandle = null;
  }
  if (state.timerStartMs !== null) {
    elements.statusTimeLabel.textContent = finalLabel;
    elements.statusTimeValue.textContent = formatElapsedTime(
      performance.now() - state.timerStartMs,
    );
    state.timerStartMs = null;
  } else if (finalLabel === "Time") {
    elements.statusTimeLabel.textContent = "Time";
    elements.statusTimeValue.textContent = "--";
  }
}

function startStatusTimer() {
  stopStatusTimer();
  state.timerStartMs = performance.now();
  elements.statusTimeLabel.textContent = "Elapsed time";
  elements.statusTimeValue.textContent = "0.0 s";
  state.timerHandle = window.setInterval(() => {
    if (state.timerStartMs === null) {
      return;
    }
    elements.statusTimeValue.textContent = formatElapsedTime(
      performance.now() - state.timerStartMs,
    );
  }, 100);
}

function syncModelFileState() {
  const disabled = elements.useBundledModel.checked || state.running;
  elements.modelFile.disabled = disabled;
  elements.modelFileField.style.opacity = disabled ? "0.6" : "1";
}

function syncInferenceFpsState() {
  const disabled = elements.nativeFps.checked || state.running;
  elements.inferenceFps.disabled = disabled;
}

function updateDownloadButtons() {
  const busy = state.running;
  elements.downloadDecodedButton.disabled = busy || !state.decodedImageData;
  elements.downloadOverlayButton.disabled = busy || !state.overlayImageData;
  elements.downloadCsvButton.disabled = busy || !state.exportFiles.csvText;
  elements.downloadFcButton.disabled = busy || !state.exportFiles.fcText;
  elements.downloadEchotasticButton.disabled = busy || !state.exportFiles.echotasticText;
}

function setBusy(busy) {
  state.running = busy;
  elements.sonarFile.disabled = busy;
  elements.startFrame.disabled = busy;
  elements.endFrame.disabled = busy;
  elements.upstreamDirection.disabled = busy;
  elements.backend.disabled = busy;
  elements.confidence.disabled = busy;
  elements.iou.disabled = busy;
  elements.nativeFps.disabled = busy;
  elements.useBundledModel.disabled = busy;
  elements.decodeButton.disabled = busy;
  elements.runButton.disabled = busy;
  syncModelFileState();
  syncInferenceFpsState();
  updateDownloadButtons();
}

function fileStem(filename) {
  const dotIndex = filename.lastIndexOf(".");
  return dotIndex > 0 ? filename.slice(0, dotIndex) : filename;
}

function currentArisStem() {
  return fileStem(state.sonarFile?.name ?? "echogram");
}

function currentCsvBaseName() {
  if (!state.decoded) {
    return currentArisStem();
  }
  return `${currentArisStem()}_${state.decoded.frameRange.start}_${state.decoded.frameRange.end}`;
}

function asFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildDecodedSummary(decoded) {
  const parts = [`${decoded.width} frames x ${decoded.height} bins`];
  const metadata = decoded.metadata ?? null;
  const windowStart = asFiniteNumber(metadata?.windowstart);
  const windowLength = asFiniteNumber(metadata?.windowlength);
  const frameRate = headerFramerateFromMetadata(metadata);

  if (windowStart !== null && windowLength !== null) {
    const windowEnd = windowStart + windowLength;
    parts.push(`${windowStart.toFixed(2)}m - ${windowEnd.toFixed(2)}m`);
  }
  if (frameRate !== null) {
    parts.push(`${frameRate.toFixed(2)}fps`);
  }

  return parts.join(", ");
}

function buildZoomCoordsText(imageX, imageY, imageHeight) {
  const frameStart = state.decoded?.frameRange?.start ?? 0;
  const frameNumber = frameStart + imageX;
  const metadata = state.decoded?.metadata ?? null;
  const windowStart = asFiniteNumber(metadata?.windowstart);
  const windowLength = asFiniteNumber(metadata?.windowlength);

  if (
    windowStart !== null &&
    windowLength !== null &&
    imageHeight > 1
  ) {
    const percentUp = (imageHeight - 1 - imageY) / (imageHeight - 1);
    const distance = windowStart + percentUp * windowLength;
    return `frame ${frameNumber} | distance ${distance.toFixed(2)}m`;
  }

  return `frame ${frameNumber}`;
}

function renderMeta(decoded) {
  const info = decoded.metadata;
  const nativeHeaderFrameRate = headerFramerateFromMetadata(decoded.metadata);
  const entries = [
    ["Version", `DDF_${info.version_id}`],
    ["Frames", `${decoded.frameRange.start}..${decoded.frameRange.end - 1}`],
    ["Image", `${decoded.width} x ${decoded.height}`],
    ["Native FPS", nativeHeaderFrameRate === null ? "--" : nativeHeaderFrameRate.toFixed(2)],
    ["Beams", String(info.numbeams)],
    ["Samples", String(info.samplesperchannel)],
    ["Num frames", String(info.numframes)],
  ];

  elements.metaList.innerHTML = entries
    .map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`)
    .join("");
}

function drawImageDataToCanvas(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.classList.remove("canvas-empty");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.putImageData(imageData, 0, 0);
}

function getStoredImageDataForCanvas(canvas) {
  if (canvas === elements.decodedCanvas) {
    return state.decodedImageData;
  }
  if (canvas === elements.overlayCanvas) {
    return state.overlayImageData;
  }
  return null;
}

function restoreCanvasBaseImage(canvas) {
  const imageData = getStoredImageDataForCanvas(canvas);
  if (!imageData) {
    return;
  }
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.putImageData(imageData, 0, 0);
}

function renderDecodedPreview(decoded, rgbImage) {
  const imageData = rgbToImageData(rgbImage, decoded.width, decoded.height);
  state.decodedImageData = imageData;
  drawImageDataToCanvas(elements.decodedCanvas, imageData);
  elements.decodedSummary.textContent = buildDecodedSummary(decoded);
  updateDownloadButtons();
}

function renderOverlayPreview(
  imageData,
  detections,
  achievedFrameRate,
  usedNativeFps,
  nativeHeaderFrameRate,
) {
  state.overlayImageData = imageData;
  drawImageDataToCanvas(elements.overlayCanvas, imageData);
  const nativeDisplayRate =
    nativeHeaderFrameRate ?? achievedFrameRate;
  const fpsSummary = usedNativeFps
    ? `native ${nativeDisplayRate.toFixed(2)} fps`
    : `${achievedFrameRate.toFixed(2)} fps`;
  elements.overlaySummary.textContent =
    `${detections.length} detections rendered | inference fps ${fpsSummary}`;
  updateDownloadButtons();
}

function renderDetectionsTable(displayRows, detections) {
  if (displayRows.length === 0) {
    elements.detectionsBody.innerHTML =
      '<tr><td colspan="5">No detections above the current confidence threshold.</td></tr>';
    elements.countsSummary.textContent = "0 detections";
    return;
  }

  const counts = new Map();
  for (const detection of detections) {
    counts.set(
      detection.className,
      (counts.get(detection.className) ?? 0) + 1,
    );
  }
  elements.countsSummary.textContent = [...counts.entries()]
    .map(([name, count]) => `${name}: ${count}`)
    .join("  |  ");

  elements.detectionsBody.innerHTML = displayRows
    .map((row) => {
      const frameCell = row.frameNumber ?? "--";
      const rangeCell = row.rangeMeters === null ? "--" : row.rangeMeters.toFixed(2);
      return `<tr>
        <td>${row.index}</td>
        <td>${frameCell}</td>
        <td>${row.direction}</td>
        <td>${rangeCell}</td>
        <td>${row.confidence}</td>
      </tr>`;
    })
    .join("");
}

function hideZoomPopup() {
  elements.zoomPopup.hidden = true;
  if (state.zoom.sourceCanvas) {
    restoreCanvasBaseImage(state.zoom.sourceCanvas);
  }
  state.zoom.sourceCanvas = null;
}

function renderZoomPopup(sourceCanvas, title, imageX, imageY, clientX, clientY) {
  if (sourceCanvas.width <= 1 || sourceCanvas.height <= 1) {
    hideZoomPopup();
    return;
  }

  const halfRegion = Math.floor(zoomRegionSize / 2);
  let sourceX = imageX - halfRegion;
  let sourceY = imageY - halfRegion;
  sourceX = Math.max(
    0,
    Math.min(sourceCanvas.width - zoomRegionSize, sourceX),
  );
  sourceY = Math.max(
    0,
    Math.min(sourceCanvas.height - zoomRegionSize, sourceY),
  );
  if (sourceCanvas.width < zoomRegionSize) {
    sourceX = 0;
  }
  if (sourceCanvas.height < zoomRegionSize) {
    sourceY = 0;
  }

  elements.zoomCanvas.width = ZOOM_CANVAS_SIZE;
  elements.zoomCanvas.height = ZOOM_CANVAS_SIZE;
  const zoomCtx = elements.zoomCanvas.getContext("2d");
  zoomCtx.clearRect(0, 0, ZOOM_CANVAS_SIZE, ZOOM_CANVAS_SIZE);
  zoomCtx.imageSmoothingEnabled = false;
  const sampleWidth = Math.min(zoomRegionSize, sourceCanvas.width);
  const sampleHeight = Math.min(zoomRegionSize, sourceCanvas.height);
  zoomCtx.drawImage(
    sourceCanvas,
    sourceX,
    sourceY,
    sampleWidth,
    sampleHeight,
    0,
    0,
    ZOOM_CANVAS_SIZE,
    ZOOM_CANVAS_SIZE,
  );

  const centerX = Math.round(
    ((imageX - sourceX) / Math.max(1, sampleWidth)) * ZOOM_CANVAS_SIZE,
  );
  const centerY = Math.round(
    ((imageY - sourceY) / Math.max(1, sampleHeight)) * ZOOM_CANVAS_SIZE,
  );
  zoomCtx.strokeStyle = "rgba(255,255,255,0.9)";
  zoomCtx.lineWidth = 1;
  zoomCtx.beginPath();
  zoomCtx.moveTo(centerX, 0);
  zoomCtx.lineTo(centerX, ZOOM_CANVAS_SIZE);
  zoomCtx.moveTo(0, centerY);
  zoomCtx.lineTo(ZOOM_CANVAS_SIZE, centerY);
  zoomCtx.stroke();

  elements.zoomTitle.textContent = title;
  elements.zoomCoords.textContent =
    `${buildZoomCoordsText(imageX, imageY, sourceCanvas.height)} | ${sampleWidth} px | +/-`;
  elements.zoomPopup.hidden = false;

  const popupWidth = elements.zoomPopup.offsetWidth || 624;
  const popupHeight = elements.zoomPopup.offsetHeight || 260;
  const preferLeftSide = clientX > window.innerWidth / 2;
  const preferredLeft = preferLeftSide
    ? clientX - popupWidth - 20
    : clientX + 20;
  const left = Math.max(
    12,
    Math.min(window.innerWidth - popupWidth - 12, preferredLeft),
  );
  const top = Math.min(
    window.innerHeight - popupHeight - 12,
    Math.max(12, clientY + 20),
  );
  elements.zoomPopup.style.left = `${left}px`;
  elements.zoomPopup.style.top = `${top}px`;
}

function rerenderZoomPopup() {
  if (!state.zoom.sourceCanvas) {
    return;
  }
  renderZoomPopup(
    state.zoom.sourceCanvas,
    state.zoom.title,
    state.zoom.imageX,
    state.zoom.imageY,
    state.zoom.clientX,
    state.zoom.clientY,
  );
}

function updateZoomPopup(sourceCanvas, title, event) {
  if (sourceCanvas.width <= 1 || sourceCanvas.height <= 1) {
    hideZoomPopup();
    return;
  }

  const rect = sourceCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    hideZoomPopup();
    return;
  }

  const xRatio = sourceCanvas.width / rect.width;
  const yRatio = sourceCanvas.height / rect.height;
  const imageX = Math.max(
    0,
    Math.min(sourceCanvas.width - 1, Math.floor((event.clientX - rect.left) * xRatio)),
  );
  const imageY = Math.max(
    0,
    Math.min(
      sourceCanvas.height - 1,
      Math.floor((event.clientY - rect.top) * yRatio),
    ),
  );

  state.zoom.sourceCanvas = sourceCanvas;
  state.zoom.title = title;
  state.zoom.imageX = imageX;
  state.zoom.imageY = imageY;
  state.zoom.clientX = event.clientX;
  state.zoom.clientY = event.clientY;
  rerenderZoomPopup();
}

function attachZoomHandlers(canvas, title) {
  canvas.addEventListener("mouseenter", (event) => {
    updateZoomPopup(canvas, title, event);
  });
  canvas.addEventListener("mousemove", (event) => {
    updateZoomPopup(canvas, title, event);
  });
  canvas.addEventListener("mouseleave", () => {
    hideZoomPopup();
  });
}

function currentModelSource() {
  if (elements.useBundledModel.checked) {
    return defaultModelUrl();
  }
  const [file] = elements.modelFile.files ?? [];
  return file ?? null;
}

function currentModelKey() {
  const source = currentModelSource();
  if (typeof source === "string") {
    return `url:${source}`;
  }
  if (source) {
    return `file:${source.name}:${source.size}:${source.lastModified}`;
  }
  return null;
}

async function ensureSession() {
  const backend = elements.backend.value;
  const modelSource = currentModelSource();
  const modelKey = `${backend}:${currentModelKey()}`;
  if (!modelSource) {
    throw new Error("Select an ONNX model or use the bundled weights");
  }
  if (state.session && state.modelKey === modelKey) {
    return state.session;
  }

  setStatus(`Loading ONNX model with ${backend}...`, 10);
  state.session = await createSegmentationSession(modelSource, backend);
  state.modelKey = modelKey;
  return state.session;
}

function buildExportFiles() {
  const upstreamDirection = elements.upstreamDirection.value;
  const arisStem = currentArisStem();
  state.displayRows = buildDisplayRows(state.predictionRows, upstreamDirection);
  state.exportFiles = {
    csvName: csvFilename(currentCsvBaseName()),
    csvText: formatPredictionCsv(state.predictionRows),
    fcName: fcFilename(arisStem),
    fcText: formatFcFile(state.predictionRows, state.sonarFile?.name ?? arisStem, upstreamDirection),
    echotasticName: echotasticFilename(arisStem),
    echotasticText: formatEchotasticFile(
      state.predictionRows,
      state.sonarFile?.name ?? arisStem,
      arisStem,
      state.decoded?.metadata ?? null,
      upstreamDirection,
    ),
  };
  renderDetectionsTable(state.displayRows, state.detections);
  updateDownloadButtons();
}

async function decodeSelectedFile() {
  const [file] = elements.sonarFile.files ?? [];
  if (!file) {
    throw new Error("Select an ARIS or DDF file first");
  }
  state.sonarFile = file;

  const buffer = await file.arrayBuffer();
  const startFrame = Number(elements.startFrame.value);
  const endFrame = Number(elements.endFrame.value);

  setStatus("Decoding sonar file in JavaScript...", 0);
  const decoded = await decodeSonarBuffer(buffer, {
    startFrame,
    endFrame,
    bgs: true,
    rawThirdChannel: true,
    returnAsBgr: true,
    onProgress(done, total) {
      const pct = total > 0 ? (done / total) * 100 : 0;
      setStatus(`Decoding echogram... ${done}/${total} frames`, pct);
    },
  });

  const rgbImage = bgrToRgbImage(decoded.imageBgr);
  const visualRgbImage = makeEchogramVisualFromBgr(
    decoded.imageBgr,
    decoded.width,
    decoded.height,
  );
  state.decoded = decoded;
  state.rgbImage = rgbImage;
  state.visualRgbImage = visualRgbImage;
  state.overlayImage = null;
  state.overlayImageData = null;
  state.detections = [];
  state.predictionRows = [];
  state.displayRows = [];
  state.frameIndices = null;
  state.inferenceFrameRate = null;
  state.inferenceUsedNativeFps = true;
  state.exportFiles = {
    csvName: null,
    csvText: null,
    fcName: null,
    fcText: null,
    echotasticName: null,
    echotasticText: null,
  };
  renderMeta(decoded);
  renderDecodedPreview(decoded, visualRgbImage);
  elements.overlaySummary.textContent = "Run inference to populate this panel.";
  elements.countsSummary.textContent = "No detections yet.";
  elements.detectionsBody.innerHTML =
    '<tr><td colspan="5">No detections yet.</td></tr>';
  elements.overlayCanvas.width = 1;
  elements.overlayCanvas.height = 1;
  elements.overlayCanvas.classList.add("canvas-empty");
  updateDownloadButtons();
  setStatus(`Decoded ${file.name}`, 100);
  return decoded;
}

async function runSegmentation() {
  const decoded = state.decoded ?? (await decodeSelectedFile());
  const nativeHeaderFrameRate = headerFramerateFromMetadata(decoded.metadata);
  const session = await ensureSession();
  const nativeFrameRate =
    nativeHeaderFrameRate ?? framerateFromMetadata(decoded.metadata);

  let inferenceRgbImage = state.rgbImage;
  let inferenceWidth = decoded.width;
  let frameIndices = Uint32Array.from({ length: decoded.width }, (_, index) => index);
  let achievedFrameRate = nativeFrameRate;
  let usedNativeFps = true;

  if (!elements.nativeFps.checked) {
    const goalFrameRate = Number(elements.inferenceFps.value);
    if (!Number.isFinite(goalFrameRate) || goalFrameRate <= 0) {
      throw new Error("Inference FPS must be a positive number");
    }
    const downsampled = downsampleRgbForInference(
      state.rgbImage,
      decoded.width,
      decoded.height,
      goalFrameRate,
      nativeFrameRate,
    );
    inferenceRgbImage = downsampled.rgbImage;
    inferenceWidth = downsampled.width;
    frameIndices = downsampled.frameIndices;
    achievedFrameRate = downsampled.achievedFrameRate;
    usedNativeFps = inferenceWidth === decoded.width;
  }

  state.frameIndices = frameIndices;
  state.inferenceFrameRate = achievedFrameRate;
  state.inferenceUsedNativeFps = usedNativeFps;

  setStatus("Running ONNX segmentation...", 20);
  const result = await runYoloSegmentation({
    session,
    rgbImage: inferenceRgbImage,
    width: inferenceWidth,
    height: decoded.height,
    outputWidth: decoded.width,
    outputHeight: decoded.height,
    confidence: Number(elements.confidence.value),
    iouThreshold: Number(elements.iou.value),
  });

  const overlayImage = makeOverlayImage(
    decoded.imageBgr,
    decoded.width,
    decoded.height,
    result.detections,
  );
  state.overlayImage = overlayImage;
  state.detections = result.detections;
  state.predictionRows = buildPredictionRows(
    result.detections,
    decoded.metadata,
    frameIndices,
  );
  buildExportFiles();
  renderOverlayPreview(
    overlayImage,
    result.detections,
    achievedFrameRate,
    usedNativeFps,
    nativeHeaderFrameRate,
  );
  setStatus(`Inference finished with ${result.detections.length} detections`, 100);
}

function resetVisuals() {
  if (state.zoom.sourceCanvas) {
    restoreCanvasBaseImage(state.zoom.sourceCanvas);
  }
  state.sonarFile = null;
  state.decoded = null;
  state.rgbImage = null;
  state.visualRgbImage = null;
  state.overlayImage = null;
  state.decodedImageData = null;
  state.overlayImageData = null;
  state.detections = [];
  state.predictionRows = [];
  state.displayRows = [];
  state.frameIndices = null;
  state.inferenceFrameRate = null;
  state.inferenceUsedNativeFps = true;
  state.exportFiles = {
    csvName: null,
    csvText: null,
    fcName: null,
    fcText: null,
    echotasticName: null,
    echotasticText: null,
  };
  elements.metaList.innerHTML = "";
  elements.decodedSummary.textContent = "No file decoded yet.";
  elements.overlaySummary.textContent = "Run inference to populate this panel.";
  elements.countsSummary.textContent = "No detections yet.";
  elements.detectionsBody.innerHTML =
    '<tr><td colspan="5">No detections yet.</td></tr>';
  elements.decodedCanvas.width = 1;
  elements.decodedCanvas.height = 1;
  elements.decodedCanvas.classList.add("canvas-empty");
  elements.overlayCanvas.width = 1;
  elements.overlayCanvas.height = 1;
  elements.overlayCanvas.classList.add("canvas-empty");
  hideZoomPopup();
  stopStatusTimer("Time");
  setStatus("Idle.", 0);
  updateDownloadButtons();
}

async function runWithBusyState(task) {
  setBusy(true);
  startStatusTimer();
  try {
    await task();
    stopStatusTimer("Time");
  } catch (error) {
    stopStatusTimer("Time");
    setStatus(`Error: ${error.message}`, 0);
    throw error;
  } finally {
    setBusy(false);
  }
}

async function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Failed to create image download"));
        return;
      }
      resolve(blob);
    }, "image/png");
  });
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadCanvasPng(canvas, filename) {
  const blob = await canvasToBlob(canvas);
  downloadBlob(filename, blob);
}

function downloadTextFile(filename, text) {
  downloadBlob(filename, new Blob([text], { type: "text/plain;charset=utf-8" }));
}

elements.useBundledModel.addEventListener("change", () => {
  syncModelFileState();
});

elements.nativeFps.addEventListener("change", () => {
  syncInferenceFpsState();
});

elements.upstreamDirection.addEventListener("change", () => {
  if (!state.predictionRows.length) {
    return;
  }
  buildExportFiles();
});

elements.sonarFile.addEventListener("change", resetVisuals);

elements.decodeButton.addEventListener("click", async () => {
  try {
    await runWithBusyState(async () => {
      await decodeSelectedFile();
    });
  } catch (error) {
    console.error(error);
  }
});

elements.runButton.addEventListener("click", async () => {
  try {
    await runWithBusyState(async () => {
      await runSegmentation();
    });
  } catch (error) {
    console.error(error);
  }
});

elements.downloadDecodedButton.addEventListener("click", async () => {
  try {
    await downloadCanvasPng(
      elements.decodedCanvas,
      `${currentCsvBaseName()}_decoded.png`,
    );
  } catch (error) {
    setStatus(`Error: ${error.message}`, 0);
  }
});

elements.downloadOverlayButton.addEventListener("click", async () => {
  try {
    await downloadCanvasPng(
      elements.overlayCanvas,
      `${currentCsvBaseName()}_overlay.png`,
    );
  } catch (error) {
    setStatus(`Error: ${error.message}`, 0);
  }
});

elements.downloadCsvButton.addEventListener("click", () => {
  if (state.exportFiles.csvText && state.exportFiles.csvName) {
    downloadTextFile(state.exportFiles.csvName, state.exportFiles.csvText);
  }
});

elements.downloadFcButton.addEventListener("click", () => {
  if (state.exportFiles.fcText && state.exportFiles.fcName) {
    downloadTextFile(state.exportFiles.fcName, state.exportFiles.fcText);
  }
});

elements.downloadEchotasticButton.addEventListener("click", () => {
  if (state.exportFiles.echotasticText && state.exportFiles.echotasticName) {
    downloadTextFile(
      state.exportFiles.echotasticName,
      state.exportFiles.echotasticText,
    );
  }
});

attachZoomHandlers(elements.decodedCanvas, "Decoded echogram");
attachZoomHandlers(elements.overlayCanvas, "Detections overlay");

window.addEventListener("keydown", (event) => {
  if (!state.zoom.sourceCanvas) {
    return;
  }
  if (
    event.target instanceof HTMLElement &&
    (event.target.tagName === "INPUT" ||
      event.target.tagName === "SELECT" ||
      event.target.tagName === "TEXTAREA")
  ) {
    return;
  }

  if (event.key === "+" || event.key === "=") {
    const nextSize = Math.max(MIN_ZOOM_REGION_SIZE, zoomRegionSize - ZOOM_REGION_STEP);
    if (nextSize !== zoomRegionSize) {
      zoomRegionSize = nextSize;
      rerenderZoomPopup();
    }
    event.preventDefault();
  } else if (event.key === "-") {
    const nextSize = Math.min(MAX_ZOOM_REGION_SIZE, zoomRegionSize + ZOOM_REGION_STEP);
    if (nextSize !== zoomRegionSize) {
      zoomRegionSize = nextSize;
      rerenderZoomPopup();
    }
    event.preventDefault();
  } else if (event.key === "Escape") {
    hideZoomPopup();
  }
});

syncModelFileState();
syncInferenceFpsState();
updateDownloadButtons();
setStatus("Idle.", 0);
