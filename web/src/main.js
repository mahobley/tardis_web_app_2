import {
  buildDisplayRows,
  buildPredictionRows,
  csvFilename,
  fcDirectionForCrossing,
  downsampleRgbForInference,
  echotasticFilename,
  fcFilename,
  formatEchotasticFile,
  formatFcFile,
  formatPredictionCsv,
  framerateFromMetadata,
  headerFramerateFromMetadata,
  isNoCrossDetection,
  resizeRgbHeightForInference,
} from "./detections/exports.js";
import { createSegmentationSession, defaultModelUrl, runYoloSegmentation } from "./inference/yoloSegmentation.js";
import { rgbToImageData } from "./render/imageData.js";
import {
  makeOverlayBaseImage,
  makeOverlayBaseImageFromRgb,
  makeOverlayImageFromBase,
} from "./render/overlay.js";
import { makeEchogramVisualFromBgr } from "./render/visualizeEchogram.js";
import { bgrToRgbImage, decodeSonarBuffer } from "./sonar/decoder.js";

const EXAMPLE_ARIS_FILENAME = "2018-08-17-JD229_Channel_Stratum1_Set1_CH_2018-08-17_230006.aris";
const EXAMPLE_ARIS_URL = new URL(
  `../../example_aris/${EXAMPLE_ARIS_FILENAME}`,
  import.meta.url,
).href;

const elements = {
  exampleArisButton: document.querySelector("#example-aris-button"),
  sonarFile: document.querySelector("#sonar-file"),
  modelFile: document.querySelector("#model-file"),
  useBundledModel: document.querySelector("#use-bundled-model"),
  modelFileField: document.querySelector("#model-file-field"),
  runAllFrames: document.querySelector("#run-all-frames"),
  startFrame: document.querySelector("#start-frame"),
  startFrameField: document.querySelector("#start-frame-field"),
  endFrame: document.querySelector("#end-frame"),
  endFrameField: document.querySelector("#end-frame-field"),
  upstreamDirectionInputs: [...document.querySelectorAll('input[name="upstream-direction"]')],
  backend: document.querySelector("#backend"),
  confidence: document.querySelector("#confidence"),
  iou: document.querySelector("#iou"),
  advancedReset: document.querySelector("#advanced-reset"),
  advancedToggle: document.querySelector("#advanced-toggle"),
  advancedContent: document.querySelector("#advanced-content"),
  nativeFps: document.querySelector("#native-fps"),
  inferenceFps: document.querySelector("#inference-fps"),
  inferenceFpsField: document.querySelector("#inference-fps-field"),
  nativeBins: document.querySelector("#native-bins"),
  inferenceBins: document.querySelector("#inference-bins"),
  inferenceBinsField: document.querySelector("#inference-bins-field"),
  decodeButton: document.querySelector("#decode-button"),
  runButton: document.querySelector("#run-button"),
  statusText: document.querySelector("#status-text"),
  statusErrorBox: document.querySelector("#status-error-box"),
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
  toggleNoCrossButton: document.querySelector("#toggle-no-cross-button"),
  downloadDecodedButton: document.querySelector("#download-decoded-button"),
  downloadDecodedJpgButton: document.querySelector("#download-decoded-jpg-button"),
  downloadOverlayButton: document.querySelector("#download-overlay-button"),
  downloadOverlayJpgButton: document.querySelector("#download-overlay-jpg-button"),
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
  overlayBaseImageData: null,
  decodedImageData: null,
  overlayImageData: null,
  detections: [],
  predictionRows: [],
  displayRows: [],
  frameIndices: null,
  decodeRequest: null,
  inferenceFrameRate: null,
  inferenceUsedNativeFps: true,
  inferenceNumBins: null,
  inferenceUsedNativeBins: true,
  hideNoCrossTracks: false,
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
const MAX_BROWSER_INFERENCE_PIXELS = 4_194_304;
let zoomRegionSize = DEFAULT_ZOOM_REGION_SIZE;
const ADVANCED_DEFAULTS = {
  backend: "webgpu",
  runAllFrames: true,
  startFrame: "0",
  endFrame: "-1",
  nativeFps: true,
  inferenceFps: "15",
  nativeBins: true,
  inferenceBins: "128",
  confidence: "0.10",
  iou: "0.50",
  useBundledModel: true,
};
const SETTINGS_STORAGE_KEY = "fisheye-echogram-settings-v1";
const DEFAULT_SETTINGS = {
  upstreamDirection: "left",
  advancedExpanded: false,
  ...ADVANCED_DEFAULTS,
};

function setAdvancedExpanded(expanded, persist = false) {
  elements.advancedContent.hidden = !expanded;
  elements.advancedReset.hidden = !expanded;
  elements.advancedToggle.setAttribute("aria-expanded", String(expanded));
  elements.advancedToggle.textContent = expanded ? "Hide" : "Show";
  if (persist) {
    persistSettings();
  }
}

function resetAdvancedSettings() {
  elements.backend.value = ADVANCED_DEFAULTS.backend;
  elements.runAllFrames.checked = ADVANCED_DEFAULTS.runAllFrames;
  elements.startFrame.value = ADVANCED_DEFAULTS.startFrame;
  elements.endFrame.value = ADVANCED_DEFAULTS.endFrame;
  elements.nativeFps.checked = ADVANCED_DEFAULTS.nativeFps;
  elements.inferenceFps.value = ADVANCED_DEFAULTS.inferenceFps;
  elements.nativeBins.checked = ADVANCED_DEFAULTS.nativeBins;
  elements.inferenceBins.value = ADVANCED_DEFAULTS.inferenceBins;
  elements.confidence.value = ADVANCED_DEFAULTS.confidence;
  elements.iou.value = ADVANCED_DEFAULTS.iou;
  elements.useBundledModel.checked = ADVANCED_DEFAULTS.useBundledModel;
  elements.modelFile.value = "";
  syncFrameRangeState();
  syncInferenceFpsState();
  syncInferenceBinsState();
  syncModelFileState();
  persistSettings();
}

function loadStoredSettings() {
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function persistSettings() {
  try {
    window.localStorage.setItem(
      SETTINGS_STORAGE_KEY,
      JSON.stringify({
        upstreamDirection: currentUpstreamDirection(),
        advancedExpanded: !elements.advancedContent.hidden,
        backend: elements.backend.value,
        runAllFrames: elements.runAllFrames.checked,
        startFrame: elements.startFrame.value,
        endFrame: elements.endFrame.value,
        nativeFps: elements.nativeFps.checked,
        inferenceFps: elements.inferenceFps.value,
        nativeBins: elements.nativeBins.checked,
        inferenceBins: elements.inferenceBins.value,
        confidence: elements.confidence.value,
        iou: elements.iou.value,
        useBundledModel: elements.useBundledModel.checked,
      }),
    );
  } catch {
    // Ignore storage failures and continue with in-memory settings.
  }
}

function applyStoredSettings() {
  const stored = loadStoredSettings();
  const settings = {
    ...DEFAULT_SETTINGS,
    ...(stored ?? {}),
  };

  const upstreamDirection =
    settings.upstreamDirection === "right" ? "right" : DEFAULT_SETTINGS.upstreamDirection;
  for (const input of elements.upstreamDirectionInputs) {
    input.checked = input.value === upstreamDirection;
  }

  elements.backend.value =
    settings.backend === "wasm" || settings.backend === "webgpu"
      ? settings.backend
      : DEFAULT_SETTINGS.backend;
  elements.runAllFrames.checked =
    typeof settings.runAllFrames === "boolean"
      ? settings.runAllFrames
      : DEFAULT_SETTINGS.runAllFrames;
  elements.startFrame.value =
    typeof settings.startFrame === "string" ? settings.startFrame : DEFAULT_SETTINGS.startFrame;
  elements.endFrame.value =
    typeof settings.endFrame === "string" ? settings.endFrame : DEFAULT_SETTINGS.endFrame;
  elements.nativeFps.checked =
    typeof settings.nativeFps === "boolean" ? settings.nativeFps : DEFAULT_SETTINGS.nativeFps;
  elements.inferenceFps.value =
    typeof settings.inferenceFps === "string"
      ? settings.inferenceFps
      : DEFAULT_SETTINGS.inferenceFps;
  elements.nativeBins.checked =
    typeof settings.nativeBins === "boolean" ? settings.nativeBins : DEFAULT_SETTINGS.nativeBins;
  elements.inferenceBins.value =
    typeof settings.inferenceBins === "string"
      ? settings.inferenceBins
      : DEFAULT_SETTINGS.inferenceBins;
  elements.confidence.value =
    typeof settings.confidence === "string"
      ? settings.confidence
      : DEFAULT_SETTINGS.confidence;
  elements.iou.value = typeof settings.iou === "string" ? settings.iou : DEFAULT_SETTINGS.iou;
  elements.useBundledModel.checked =
    typeof settings.useBundledModel === "boolean"
      ? settings.useBundledModel
      : DEFAULT_SETTINGS.useBundledModel;

  syncFrameRangeState();
  syncInferenceFpsState();
  syncInferenceBinsState();
  syncModelFileState();
  setAdvancedExpanded(Boolean(settings.advancedExpanded));
}

function setStatus(text, progress = null) {
  elements.statusText.textContent = text;
  if (progress !== null) {
    elements.progressBar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

function clearStatusError() {
  elements.statusErrorBox.hidden = true;
  elements.statusErrorBox.textContent = "";
}

function setStatusError(message) {
  elements.statusErrorBox.textContent = message;
  elements.statusErrorBox.hidden = !message;
}

function userFacingErrorMessage(error) {
  const message = error instanceof Error ? error.message : String(error);
  if (
    message.includes("ran out of memory") ||
    message.includes("too large for the browser") ||
    message.includes("std::bad_alloc") ||
    message.includes("ERROR_CODE: 6")
  ) {
    return "The selected data is too large for the web version with the current settings. Decrease the frame range, frame rate, and/or num bins, then try again.";
  }
  return message;
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
    elements.statusTimeValue.textContent = formatElapsedTime(performance.now() - state.timerStartMs);
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
    elements.statusTimeValue.textContent = formatElapsedTime(performance.now() - state.timerStartMs);
  }, 100);
}

async function waitForNextPaint() {
  await new Promise((resolve) => {
    window.requestAnimationFrame(() => {
      window.setTimeout(resolve, 0);
    });
  });
}

function syncModelFileState() {
  const disabled = elements.useBundledModel.checked || state.running;
  elements.modelFile.disabled = disabled;
  elements.modelFileField.classList.toggle("disabled-field", disabled);
}

function syncFrameRangeState() {
  const disabled = elements.runAllFrames.checked || state.running;
  elements.startFrame.disabled = disabled;
  elements.endFrame.disabled = disabled;
  elements.startFrameField.classList.toggle("disabled-field", disabled);
  elements.endFrameField.classList.toggle("disabled-field", disabled);
}

function syncInferenceFpsState() {
  const disabled = elements.nativeFps.checked || state.running;
  elements.inferenceFps.disabled = disabled;
  elements.inferenceFpsField.classList.toggle("disabled-field", disabled);
}

function syncInferenceBinsState() {
  const disabled = elements.nativeBins.checked || state.running;
  elements.inferenceBins.disabled = disabled;
  elements.inferenceBinsField.classList.toggle("disabled-field", disabled);
}

function syncRunButtonLabel() {
  elements.runButton.textContent = state.decoded ? "Run TaRDIS" : "Generate Echogram + Run TaRDIS";
}

function currentInputSonarFile() {
  const [file] = elements.sonarFile.files ?? [];
  return file ?? null;
}

function currentSelectedSonarFile() {
  return state.sonarFile ?? currentInputSonarFile();
}

function currentUpstreamDirection() {
  return elements.upstreamDirectionInputs.find((input) => input.checked)?.value ?? "left";
}

function currentDecodeRequest() {
  return {
    startFrame: elements.runAllFrames.checked ? 0 : Number(elements.startFrame.value),
    endFrame: elements.runAllFrames.checked ? -1 : Number(elements.endFrame.value),
  };
}

function decodeRequestMatchesCurrent() {
  if (!state.decoded || !state.decodeRequest) {
    return false;
  }
  const requested = currentDecodeRequest();
  return (
    state.decodeRequest.startFrame === requested.startFrame &&
    state.decodeRequest.endFrame === requested.endFrame
  );
}

function computeAutoInferenceTargets(width, height) {
  const pixelCount = width * height;
  if (pixelCount <= MAX_BROWSER_INFERENCE_PIXELS) {
    return null;
  }
  const scale = Math.sqrt(MAX_BROWSER_INFERENCE_PIXELS / pixelCount);
  return {
    width: Math.max(1, Math.floor(width * scale)),
    height: Math.max(1, Math.floor(height * scale)),
  };
}

function updateDownloadButtons() {
  const busy = state.running;
  elements.toggleNoCrossButton.disabled = busy || !state.overlayBaseImageData;
  elements.downloadDecodedButton.disabled = busy || !state.decodedImageData;
  elements.downloadDecodedJpgButton.disabled = busy || !state.decodedImageData;
  elements.downloadOverlayButton.disabled = busy || !state.overlayImageData;
  elements.downloadOverlayJpgButton.disabled = busy || !state.overlayImageData;
  elements.downloadCsvButton.disabled = busy || !state.exportFiles.csvText;
  elements.downloadFcButton.disabled = busy || !state.exportFiles.fcText;
  elements.downloadEchotasticButton.disabled = busy || !state.exportFiles.echotasticText;
}

function setBusy(busy) {
  state.running = busy;
  elements.exampleArisButton.disabled = busy;
  elements.sonarFile.disabled = busy;
  elements.runAllFrames.disabled = busy;
  for (const input of elements.upstreamDirectionInputs) {
    input.disabled = busy;
  }
  elements.backend.disabled = busy;
  elements.confidence.disabled = busy;
  elements.iou.disabled = busy;
  elements.nativeFps.disabled = busy;
  elements.nativeBins.disabled = busy;
  elements.useBundledModel.disabled = busy;
  elements.advancedReset.disabled = busy;
  elements.advancedToggle.disabled = busy;
  elements.decodeButton.disabled = busy;
  elements.runButton.disabled = busy;
  syncFrameRangeState();
  syncModelFileState();
  syncInferenceFpsState();
  syncInferenceBinsState();
  updateDownloadButtons();
}

function fileStem(filename) {
  const dotIndex = filename.lastIndexOf(".");
  return dotIndex > 0 ? filename.slice(0, dotIndex) : filename;
}

function currentArisStem() {
  return fileStem(currentSelectedSonarFile()?.name ?? "echogram");
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

function clearFieldValidation(field) {
  field.setCustomValidity("");
}

function failFieldValidation(field, message) {
  field.setCustomValidity(message);
  return {
    ok: false,
    field,
    message,
  };
}

function validateFrameRangeInputs() {
  clearFieldValidation(elements.startFrame);
  clearFieldValidation(elements.endFrame);

  if (elements.runAllFrames.checked) {
    return { ok: true };
  }

  const startFrame = Number(elements.startFrame.value);
  if (!Number.isInteger(startFrame) || startFrame < 0) {
    return failFieldValidation(
      elements.startFrame,
      "Start frame must be a whole number greater than or equal to 0.",
    );
  }

  const endFrame = Number(elements.endFrame.value);
  if (!Number.isInteger(endFrame) || (endFrame < 0 && endFrame !== -1)) {
    return failFieldValidation(
      elements.endFrame,
      "End frame must be -1 or a whole number greater than or equal to 0.",
    );
  }

  if (endFrame !== -1 && endFrame <= startFrame) {
    return failFieldValidation(
      elements.endFrame,
      "End frame must be greater than start frame.",
    );
  }

  if (endFrame !== -1 && endFrame - startFrame < 2) {
    return failFieldValidation(
      elements.endFrame,
      "Frame range must include at least 2 frames to generate an echogram.",
    );
  }

  return { ok: true };
}

function validateConfidenceInput() {
  clearFieldValidation(elements.confidence);
  const confidence = Number(elements.confidence.value);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    return failFieldValidation(
      elements.confidence,
      "Confidence must be a number between 0 and 1.",
    );
  }
  return { ok: true };
}

function validateIouInput() {
  clearFieldValidation(elements.iou);
  const iou = Number(elements.iou.value);
  if (!Number.isFinite(iou) || iou < 0 || iou > 1) {
    return failFieldValidation(
      elements.iou,
      "IoU must be a number between 0 and 1.",
    );
  }
  return { ok: true };
}

function validateControls({ report = false, updateStatus = true } = {}) {
  const firstFailure =
    [validateFrameRangeInputs(), validateConfidenceInput(), validateIouInput()].find(
      (result) => !result.ok,
    ) ?? null;

  if (!firstFailure) {
    clearStatusError();
    return true;
  }

  if (updateStatus) {
    setStatus(`Error: ${firstFailure.message}`, 0);
  }
  setStatusError(firstFailure.message);
  if (report) {
    firstFailure.field.reportValidity();
  }
  return false;
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

function buildZoomCoordsText(sourceCanvas, imageX, imageY, imageHeight) {
  const frameStart = state.decoded?.frameRange?.start ?? 0;
  let frameNumber = frameStart + imageX;
  if (sourceCanvas === elements.overlayCanvas) {
    const mappedFrameNumber = state.frameIndices?.[imageX];
    if (Number.isFinite(mappedFrameNumber)) {
      frameNumber = Math.round(mappedFrameNumber);
    }
  }
  const metadata = state.decoded?.metadata ?? null;
  const windowStart = asFiniteNumber(metadata?.windowstart);
  const windowLength = asFiniteNumber(metadata?.windowlength);

  if (windowStart !== null && windowLength !== null && imageHeight > 1) {
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

  elements.metaList.innerHTML = entries.map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`).join("");
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
  inferenceNumBins,
  usedNativeBins,
) {
  state.overlayImageData = imageData;
  drawImageDataToCanvas(elements.overlayCanvas, imageData);
  renderOverlaySummary({
    achievedFrameRate,
    usedNativeFps,
    nativeHeaderFrameRate,
    inferenceNumBins,
    usedNativeBins,
  });
  updateDownloadButtons();
}

function visibleOverlayDetections() {
  if (!state.hideNoCrossTracks) {
    return state.detections;
  }
  return state.detections.filter(
    (detection) => !isNoCrossDetection(detection.className, Number(detection.classId)),
  );
}

function countOverlayDirections(detections, upstreamDirection) {
  const counts = {
    upstream: 0,
    downstream: 0,
    noCrossing: 0,
  };

  for (const detection of detections) {
    const classId = Number(detection.classId);
    const direction = fcDirectionForCrossing(detection.className, classId, upstreamDirection);
    if (direction === "Up") {
      counts.upstream += 1;
      continue;
    }
    if (direction === "Down") {
      counts.downstream += 1;
      continue;
    }
    if (isNoCrossDetection(detection.className, classId)) {
      counts.noCrossing += 1;
      continue;
    }
    counts.noCrossing += 1;
  }

  return counts;
}

function renderOverlaySummary({
  detections = state.detections,
  achievedFrameRate = state.inferenceFrameRate,
  usedNativeFps = state.inferenceUsedNativeFps,
  nativeHeaderFrameRate = headerFramerateFromMetadata(state.decoded?.metadata ?? null),
  inferenceNumBins = state.inferenceNumBins,
  usedNativeBins = state.inferenceUsedNativeBins,
} = {}) {
  if (achievedFrameRate === null || inferenceNumBins === null) {
    elements.overlaySummary.textContent = "Run inference to populate this panel.";
    return;
  }

  const counts = countOverlayDirections(detections, currentUpstreamDirection());
  const upstreamDirection = currentUpstreamDirection();
  const upstreamSummaryClass =
    upstreamDirection === "left"
      ? "overlay-summary-count-left"
      : "overlay-summary-count-right";
  const downstreamSummaryClass =
    upstreamDirection === "left"
      ? "overlay-summary-count-right"
      : "overlay-summary-count-left";
  const nativeDisplayRate = nativeHeaderFrameRate ?? achievedFrameRate;
  const fpsSummary = usedNativeFps
    ? `native ${nativeDisplayRate.toFixed(2)} fps`
    : `${achievedFrameRate.toFixed(2)} fps`;
  const binsSummary = usedNativeBins
    ? `native ${inferenceNumBins} bins`
    : `${inferenceNumBins} bins`;

  elements.overlaySummary.innerHTML =
    `${detections.length} detections rendered (` +
    `<span class="overlay-summary-count ${upstreamSummaryClass}">${counts.upstream} upstream</span>, ` +
    `<span class="overlay-summary-count ${downstreamSummaryClass}">${counts.downstream} downstream</span>, ` +
    `<span class="overlay-summary-count overlay-summary-count-no-crossing">${counts.noCrossing} no crossing</span>)` +
    ` | inference: ${fpsSummary} | ${binsSummary}`;
}

function syncNoCrossButton() {
  elements.toggleNoCrossButton.textContent = state.hideNoCrossTracks
    ? "Show no-cross"
    : "Hide no-cross";
  elements.toggleNoCrossButton.setAttribute("aria-pressed", String(state.hideNoCrossTracks));
}

function rerenderOverlayFromState() {
  if (!state.overlayBaseImageData) {
    elements.overlaySummary.textContent = "Run inference to populate this panel.";
    state.overlayImageData = null;
    updateDownloadButtons();
    return;
  }

  const detections = visibleOverlayDetections();
  const overlayImage = makeOverlayImageFromBase(state.overlayBaseImageData, detections);
  state.overlayImage = overlayImage;
  renderOverlayPreview(
    overlayImage,
    detections,
    state.inferenceFrameRate,
    state.inferenceUsedNativeFps,
    headerFramerateFromMetadata(state.decoded?.metadata ?? null),
    state.inferenceNumBins,
    state.inferenceUsedNativeBins,
  );
}

function renderDetectionsTable(displayRows, detections) {
  if (displayRows.length === 0) {
    elements.detectionsBody.innerHTML =
      '<tr><td colspan="5">No detections above the current confidence threshold.</td></tr>';
    elements.countsSummary.textContent = "0 detections";
    return;
  }

  const counts = countOverlayDirections(detections, currentUpstreamDirection());
  elements.countsSummary.textContent =
    `${detections.length} detections (` +
    `${counts.upstream} upstream | ` +
    `${counts.downstream} downstream | ` +
    `${counts.noCrossing} no crossing)`;

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
  sourceX = Math.max(0, Math.min(sourceCanvas.width - zoomRegionSize, sourceX));
  sourceY = Math.max(0, Math.min(sourceCanvas.height - zoomRegionSize, sourceY));
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

  const centerX = Math.round(((imageX - sourceX) / Math.max(1, sampleWidth)) * ZOOM_CANVAS_SIZE);
  const centerY = Math.round(((imageY - sourceY) / Math.max(1, sampleHeight)) * ZOOM_CANVAS_SIZE);
  zoomCtx.strokeStyle = "rgba(255,255,255,0.9)";
  zoomCtx.lineWidth = 1;
  zoomCtx.beginPath();
  zoomCtx.moveTo(centerX, 0);
  zoomCtx.lineTo(centerX, ZOOM_CANVAS_SIZE);
  zoomCtx.moveTo(0, centerY);
  zoomCtx.lineTo(ZOOM_CANVAS_SIZE, centerY);
  zoomCtx.stroke();

  elements.zoomTitle.textContent = title;
  elements.zoomCoords.textContent = `${buildZoomCoordsText(sourceCanvas, imageX, imageY, sourceCanvas.height)} | ${sampleWidth} px | +/-`;
  elements.zoomPopup.hidden = false;

  const popupWidth = elements.zoomPopup.offsetWidth || 624;
  const popupHeight = elements.zoomPopup.offsetHeight || 260;
  const preferLeftSide = clientX > window.innerWidth / 2;
  const preferredLeft = preferLeftSide ? clientX - popupWidth - 20 : clientX + 20;
  const left = Math.max(12, Math.min(window.innerWidth - popupWidth - 12, preferredLeft));
  const top = Math.min(window.innerHeight - popupHeight - 12, Math.max(12, clientY + 20));
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
  const imageX = Math.max(0, Math.min(sourceCanvas.width - 1, Math.floor((event.clientX - rect.left) * xRatio)));
  const imageY = Math.max(0, Math.min(sourceCanvas.height - 1, Math.floor((event.clientY - rect.top) * yRatio)));

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
  const upstreamDirection = currentUpstreamDirection();
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
  if (state.overlayImageData) {
    renderOverlaySummary();
  }
  updateDownloadButtons();
}

async function decodeSelectedFile() {
  const file = currentSelectedSonarFile();
  if (!file) {
    throw new Error("Select an ARIS or DDF file first");
  }
  state.sonarFile = file;
  const decodeRequest = currentDecodeRequest();

  setStatus("Generating echogram in JavaScript...", 0);
  await waitForNextPaint();

  const buffer = await file.arrayBuffer();
  const decoded = await decodeSonarBuffer(buffer, {
    startFrame: decodeRequest.startFrame,
    endFrame: decodeRequest.endFrame,
    bgs: true,
    rawThirdChannel: true,
    returnAsBgr: true,
    onProgress(done, total) {
      const pct = total > 0 ? (done / total) * 100 : 0;
      setStatus(`Generating echogram... ${done}/${total} frames`, pct);
    },
  });

  const rgbImage = bgrToRgbImage(decoded.imageBgr);
  const visualRgbImage = makeEchogramVisualFromBgr(decoded.imageBgr, decoded.width, decoded.height);
  state.decoded = decoded;
  state.rgbImage = rgbImage;
  state.visualRgbImage = visualRgbImage;
  state.overlayImage = null;
  state.overlayBaseImageData = null;
  state.overlayImageData = null;
  state.detections = [];
  state.predictionRows = [];
  state.displayRows = [];
  state.frameIndices = null;
  state.decodeRequest = decodeRequest;
  state.inferenceFrameRate = null;
  state.inferenceUsedNativeFps = true;
  state.inferenceNumBins = null;
  state.inferenceUsedNativeBins = true;
  state.hideNoCrossTracks = false;
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
  syncRunButtonLabel();
  syncNoCrossButton();
  elements.overlaySummary.textContent = "Run inference to populate this panel.";
  elements.countsSummary.textContent = "No detections yet.";
  elements.detectionsBody.innerHTML = '<tr><td colspan="5">No detections yet.</td></tr>';
  elements.overlayCanvas.width = 1;
  elements.overlayCanvas.height = 1;
  elements.overlayCanvas.classList.add("canvas-empty");
  updateDownloadButtons();
  setStatus(`Generated echogram for ${file.name}`, 100);
  return decoded;
}

async function selectExampleArisFile() {
  setStatus("Loading example ARIS file...", 0);
  await waitForNextPaint();

  const response = await fetch(EXAMPLE_ARIS_URL);
  if (!response.ok) {
    throw new Error(`Unable to load bundled example ARIS (${response.status})`);
  }

  const blob = await response.blob();
  const file = new File([blob], EXAMPLE_ARIS_FILENAME, {
    type: blob.type || "application/octet-stream",
    lastModified: 0,
  });

  try {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    elements.sonarFile.files = transfer.files;
  } catch {
    elements.sonarFile.value = "";
  }

  resetVisuals();
  state.sonarFile = file;
  setStatus(`Selected example file ${file.name}`, 0);
}

async function runSegmentation() {
  const decoded = decodeRequestMatchesCurrent() ? state.decoded : await decodeSelectedFile();
  const nativeHeaderFrameRate = headerFramerateFromMetadata(decoded.metadata);
  const session = await ensureSession();
  const nativeFrameRate = nativeHeaderFrameRate ?? framerateFromMetadata(decoded.metadata);

  let inferenceRgbImage = state.rgbImage;
  let inferenceWidth = decoded.width;
  let inferenceHeight = decoded.height;
  let frameIndices = Uint32Array.from(
    { length: decoded.width },
    (_, index) => decoded.frameRange.start + index,
  );
  let achievedFrameRate = nativeFrameRate;
  let usedNativeFps = true;
  let usedNativeBins = true;
  let autoDownsampledForBrowser = false;
  let preAutoFrameRate = achievedFrameRate;
  let preAutoNumBins = inferenceHeight;

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

  if (!elements.nativeBins.checked) {
    const goalNumBins = Number(elements.inferenceBins.value);
    if (!Number.isFinite(goalNumBins) || goalNumBins <= 0 || !Number.isInteger(goalNumBins)) {
      throw new Error("Num bins must be a positive integer");
    }
    const resized = resizeRgbHeightForInference(
      inferenceRgbImage,
      inferenceWidth,
      inferenceHeight,
      goalNumBins,
    );
    inferenceRgbImage = resized.rgbImage;
    inferenceHeight = resized.height;
    usedNativeBins = inferenceHeight === decoded.height;
  }

  const autoTargets = computeAutoInferenceTargets(inferenceWidth, inferenceHeight);
  if (autoTargets) {
    preAutoFrameRate = achievedFrameRate;
    preAutoNumBins = inferenceHeight;
    setStatus(
      `Large echogram detected; reducing browser inference size to ${autoTargets.width} frames x ${autoTargets.height} bins...`,
      15,
    );

    if (autoTargets.width < inferenceWidth) {
      const priorFrameIndices = frameIndices;
      const autoGoalFrameRate = achievedFrameRate * (autoTargets.width / inferenceWidth);
      const autoDownsampled = downsampleRgbForInference(
        inferenceRgbImage,
        inferenceWidth,
        inferenceHeight,
        autoGoalFrameRate,
        achievedFrameRate,
      );
      inferenceRgbImage = autoDownsampled.rgbImage;
      inferenceWidth = autoDownsampled.width;
      frameIndices = Uint32Array.from(
        autoDownsampled.frameIndices,
        (index) => priorFrameIndices[index],
      );
      achievedFrameRate = autoDownsampled.achievedFrameRate;
      usedNativeFps = false;
      autoDownsampledForBrowser = true;
    }

    if (autoTargets.height < inferenceHeight) {
      const autoResized = resizeRgbHeightForInference(
        inferenceRgbImage,
        inferenceWidth,
        inferenceHeight,
        autoTargets.height,
      );
      inferenceRgbImage = autoResized.rgbImage;
      inferenceHeight = autoResized.height;
      usedNativeBins = false;
      autoDownsampledForBrowser = true;
    }
  }

  if (inferenceWidth * inferenceHeight > MAX_BROWSER_INFERENCE_PIXELS) {
    throw new Error(
      "Inference image is too large for the browser. Reduce Frame rate and/or Num bins, or use the desktop tool.",
    );
  }

  state.frameIndices = frameIndices;
  state.inferenceFrameRate = achievedFrameRate;
  state.inferenceUsedNativeFps = usedNativeFps;
  state.inferenceNumBins = inferenceHeight;
  state.inferenceUsedNativeBins = usedNativeBins;

  if (autoDownsampledForBrowser) {
    setStatusError(
      `The selected data is too large for the web version at these settings. The browser automatically reduced the inference resolution from ${preAutoFrameRate.toFixed(2)} to ${achievedFrameRate.toFixed(2)} FPS and from ${preAutoNumBins} to ${inferenceHeight} distance bins. For better control, decrease the frame range, frame rate, and/or num bins yourself.`,
    );
  }

  setStatus("Running ONNX segmentation...", 20);
  let result;
  try {
    result = await runYoloSegmentation({
      session,
      rgbImage: inferenceRgbImage,
      width: inferenceWidth,
      height: inferenceHeight,
      outputWidth: inferenceWidth,
      outputHeight: inferenceHeight,
      confidence: Number(elements.confidence.value),
      iouThreshold: Number(elements.iou.value),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message.includes("std::bad_alloc") || message.includes("ERROR_CODE: 6")) {
      throw new Error(
        "Browser inference ran out of memory. Reduce Frame rate and/or Num bins, or use the desktop tool.",
      );
    }
    throw error;
  }

  state.detections = result.detections;
  state.hideNoCrossTracks = false;
  syncNoCrossButton();
  state.overlayBaseImageData =
    usedNativeFps && usedNativeBins
      ? makeOverlayBaseImage(decoded.imageBgr, decoded.width, decoded.height)
      : makeOverlayBaseImageFromRgb(
          inferenceRgbImage,
          inferenceWidth,
          inferenceHeight,
        );
  const overlayImage = makeOverlayImageFromBase(state.overlayBaseImageData, visibleOverlayDetections());
  state.overlayImage = overlayImage;
  state.predictionRows = buildPredictionRows(
    result.detections,
    decoded.metadata,
    frameIndices,
    decoded.height,
  );
  buildExportFiles();
  renderOverlayPreview(
    overlayImage,
    result.detections,
    achievedFrameRate,
    usedNativeFps,
    nativeHeaderFrameRate,
    inferenceHeight,
    usedNativeBins,
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
  state.overlayBaseImageData = null;
  state.decodedImageData = null;
  state.overlayImageData = null;
  state.detections = [];
  state.predictionRows = [];
  state.displayRows = [];
  state.frameIndices = null;
  state.decodeRequest = null;
  state.inferenceFrameRate = null;
  state.inferenceUsedNativeFps = true;
  state.inferenceNumBins = null;
  state.inferenceUsedNativeBins = true;
  state.hideNoCrossTracks = false;
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
  elements.detectionsBody.innerHTML = '<tr><td colspan="5">No detections yet.</td></tr>';
  elements.decodedCanvas.width = 1;
  elements.decodedCanvas.height = 1;
  elements.decodedCanvas.classList.add("canvas-empty");
  elements.overlayCanvas.width = 1;
  elements.overlayCanvas.height = 1;
  elements.overlayCanvas.classList.add("canvas-empty");
  hideZoomPopup();
  stopStatusTimer("Time");
  setStatus("Idle.", 0);
  clearStatusError();
  syncRunButtonLabel();
  syncNoCrossButton();
  updateDownloadButtons();
}

async function runWithBusyState(task) {
  setBusy(true);
  startStatusTimer();
  clearStatusError();
  try {
    await task();
    stopStatusTimer("Time");
  } catch (error) {
    stopStatusTimer("Time");
    setStatus(`Error: ${error.message}`, 0);
    setStatusError(userFacingErrorMessage(error));
    throw error;
  } finally {
    setBusy(false);
  }
}

async function canvasToBlob(canvas, type = "image/png", quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Failed to create image download"));
          return;
        }
        resolve(blob);
      },
      type,
      quality,
    );
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
  const blob = await canvasToBlob(canvas, "image/png");
  downloadBlob(filename, blob);
}

async function downloadCanvasJpg(canvas, filename) {
  const blob = await canvasToBlob(canvas, "image/jpeg", 0.92);
  downloadBlob(filename, blob);
}

function downloadTextFile(filename, text) {
  downloadBlob(filename, new Blob([text], { type: "text/plain;charset=utf-8" }));
}

elements.useBundledModel.addEventListener("change", () => {
  syncModelFileState();
  persistSettings();
});

elements.runAllFrames.addEventListener("change", () => {
  syncFrameRangeState();
  persistSettings();
  validateControls({ updateStatus: false });
});

elements.nativeFps.addEventListener("change", () => {
  syncInferenceFpsState();
  persistSettings();
});

elements.nativeBins.addEventListener("change", () => {
  syncInferenceBinsState();
  persistSettings();
});

elements.advancedReset.addEventListener("click", () => {
  resetAdvancedSettings();
});

elements.advancedToggle.addEventListener("click", () => {
  setAdvancedExpanded(elements.advancedContent.hidden, true);
});

for (const input of elements.upstreamDirectionInputs) {
  input.addEventListener("change", () => {
    persistSettings();
    if (!state.predictionRows.length) {
      return;
    }
    buildExportFiles();
  });
}

elements.toggleNoCrossButton.addEventListener("click", () => {
  if (!state.overlayBaseImageData) {
    return;
  }
  state.hideNoCrossTracks = !state.hideNoCrossTracks;
  syncNoCrossButton();
  rerenderOverlayFromState();
});

for (const input of [
  elements.backend,
  elements.startFrame,
  elements.endFrame,
  elements.confidence,
  elements.iou,
  elements.inferenceFps,
  elements.inferenceBins,
]) {
  input.addEventListener("change", () => {
    persistSettings();
    validateControls({ updateStatus: false });
  });
}

elements.sonarFile.addEventListener("change", resetVisuals);

elements.exampleArisButton.addEventListener("click", async () => {
  try {
    await runWithBusyState(async () => {
      await selectExampleArisFile();
    });
  } catch (error) {
    console.error(error);
  }
});

elements.decodeButton.addEventListener("click", async () => {
  if (!validateControls({ report: true })) {
    return;
  }
  try {
    await runWithBusyState(async () => {
      await decodeSelectedFile();
    });
  } catch (error) {
    console.error(error);
  }
});

elements.runButton.addEventListener("click", async () => {
  if (!validateControls({ report: true })) {
    return;
  }
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
    await downloadCanvasPng(elements.decodedCanvas, `${currentCsvBaseName()}_decoded.png`);
  } catch (error) {
    setStatus(`Error: ${error.message}`, 0);
  }
});

elements.downloadOverlayButton.addEventListener("click", async () => {
  try {
    await downloadCanvasPng(elements.overlayCanvas, `${currentCsvBaseName()}_overlay.png`);
  } catch (error) {
    setStatus(`Error: ${error.message}`, 0);
  }
});

elements.downloadDecodedJpgButton.addEventListener("click", async () => {
  try {
    await downloadCanvasJpg(elements.decodedCanvas, `${currentCsvBaseName()}_decoded.jpg`);
  } catch (error) {
    setStatus(`Error: ${error.message}`, 0);
  }
});

elements.downloadOverlayJpgButton.addEventListener("click", async () => {
  try {
    await downloadCanvasJpg(elements.overlayCanvas, `${currentCsvBaseName()}_overlay.jpg`);
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
    downloadTextFile(state.exportFiles.echotasticName, state.exportFiles.echotasticText);
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
    (event.target.tagName === "INPUT" || event.target.tagName === "SELECT" || event.target.tagName === "TEXTAREA")
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

applyStoredSettings();
syncRunButtonLabel();
syncNoCrossButton();
updateDownloadButtons();
setStatus("Idle.", 0);
clearStatusError();
