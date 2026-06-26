const FC_TITLE = "*** Manual Marking (Manual Sizing: Q = Quality, N = Repeat Count) ***";
const FC_COMMENT_PREFIX = "TaRDIS-conf=";
const ECHOTASTIC_VERSION = "2.0";
const ECHOTASTIC_OPERATOR = "AUT";

const FC_SCHEMA = {
  File: { width: 4, default: 1 },
  Total: { width: 7, default: 0 },
  "Frame#": { width: 8, default: 0 },
  Dir: { width: 5, default: "" },
  "R (m)": { width: 8, default: 0.0 },
  Theta: { width: 8, default: 99.0 },
  "L(cm)": { width: 8, default: 0.0 },
  "dR(cm)": { width: 8, default: 0.0 },
  "L/dR": { width: 8, default: 0.0 },
  Aspect: { width: 8, default: 0.0 },
  Time: { width: 10, default: "00:00:00" },
  Date: { width: 12, default: "" },
  Latitude: { width: 19, default: "N 00 d  0.00000 m" },
  Longitude: { width: 20, default: "E 000 d  0.00000 m" },
  Pan: { width: 9, default: 0.0 },
  Tilt: { width: 9, default: 0.0 },
  Roll: { width: 9, default: 0.0 },
  Species: { width: 10, default: "Unknown" },
  Motion: { width: 39, default: "Running <-->" },
  Q: { width: 7, default: 5 },
  N: { width: 8, default: 1 },
  Comment: { width: 17, default: "" },
};

const FC_HEADERS = Object.keys(FC_SCHEMA);

export const PREDICTION_CSV_FIELDNAMES = [
  "instance_index",
  "confidence",
  "class_id",
  "class_name",
  "enter_frame",
  "exit_frame",
  "center_frame",
  "center_frame_bin",
  "center_frame_distance",
  "duration",
  "minimum_bin_y",
  "maximum_bin_y",
  "average_bin_y",
  "start_bin_y",
  "end_bin_y",
  "minimum_distance",
  "maximum_distance",
  "average_distance",
  "start_distance",
  "end_distance",
];

const ECHOTASTIC_COLUMNS = [
  "Sample",
  "Ping",
  "Time",
  "Range",
  "Amplitude",
  "XAngle",
  "YAngle",
  "Direction",
  "Length",
  "Area",
  "Operator",
];

function formatFcComment(confidence) {
  const value = asFloat(confidence);
  if (Number.isNaN(value)) {
    return FC_COMMENT_PREFIX.slice(0, FC_SCHEMA.Comment.width);
  }
  return `${FC_COMMENT_PREFIX}${value.toFixed(2)}`.slice(0, FC_SCHEMA.Comment.width);
}

function asFloat(value) {
  if (value === null || value === undefined) {
    return Number.NaN;
  }
  if (typeof value === "string") {
    const lowered = value.trim().toLowerCase();
    if (lowered === "nan" || lowered === "") {
      return Number.NaN;
    }
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function roundInt(value) {
  const x = asFloat(value);
  if (Number.isNaN(x)) {
    return Number.NaN;
  }
  return Math.round(x);
}

function roundDecimal(value, decimalPlaces = 2) {
  const x = asFloat(value);
  if (Number.isNaN(x)) {
    return Number.NaN;
  }
  return Number(x.toFixed(decimalPlaces));
}

function confidenceCell(value) {
  const x = asFloat(value);
  if (Number.isNaN(x)) {
    return "nan";
  }
  return x.toFixed(4);
}

function normalizeClassLabel(className, classId) {
  const name = String(className ?? "")
    .toLowerCase()
    .trim()
    .replaceAll("_", "-");
  if (name === "-1" || name === "nan" || name === "unknown" || name === "") {
    if (classId === 0) {
      return "right";
    }
    if (classId === 1) {
      return "left";
    }
    if (classId === 2) {
      return "no-cross";
    }
  }
  return name;
}

export function isNoCrossDetection(className, classId) {
  const label = normalizeClassLabel(className, classId);
  if (classId === 2) {
    return true;
  }
  return label === "no-cross" || label === "no cross" || label === "nocross" || label.includes("no-cross");
}

function crossingSide(className, classId) {
  if (isNoCrossDetection(className, classId)) {
    return null;
  }
  const label = normalizeClassLabel(className, classId);
  if (label.includes("left")) {
    return "left";
  }
  if (label.includes("right")) {
    return "right";
  }
  if (label.includes("pos")) {
    return "left";
  }
  if (label.includes("neg")) {
    return "right";
  }
  if (label === "0" || classId === 0) {
    return "right";
  }
  if (label === "1" || classId === 1) {
    return "left";
  }
  return null;
}

export function fcDirectionForCrossing(className, classId, upstreamDirection) {
  const side = crossingSide(className, classId);
  if (side === null) {
    return "";
  }
  const upstream = String(upstreamDirection ?? "").toLowerCase().trim();
  if (upstream === "left") {
    return side === "left" ? "Up" : "Down";
  }
  if (upstream === "right") {
    return side === "right" ? "Up" : "Down";
  }
  throw new Error(`upstream_direction must be 'left' or 'right', got ${upstreamDirection}`);
}

function echotasticDirection(className, classId, upstreamDirection) {
  const side = crossingSide(className, classId);
  if (side === null) {
    return null;
  }
  const upstream = String(upstreamDirection ?? "").toLowerCase().trim();
  if (upstream === "left") {
    return side === "left" ? 1 : -1;
  }
  if (upstream === "right") {
    return side === "right" ? 1 : -1;
  }
  throw new Error(`upstream_direction must be 'left' or 'right', got ${upstreamDirection}`);
}

export function headerFramerateFromMetadata(metadata) {
  if (!metadata) {
    return null;
  }
  for (const key of ["framerate", "FrameRate"]) {
    if (!(key in metadata)) {
      continue;
    }
    const rate = asFloat(metadata[key]);
    if (!Number.isNaN(rate) && rate > 0) {
      return rate;
    }
  }
  return null;
}

export function framerateFromMetadata(metadata) {
  const headerRate = headerFramerateFromMetadata(metadata);
  if (headerRate !== null) {
    return headerRate;
  }
  if (!metadata) {
    return 15.0;
  }
  for (const key of ["frame_rate", "fps"]) {
    if (!(key in metadata)) {
      continue;
    }
    const rate = asFloat(metadata[key]);
    if (!Number.isNaN(rate) && rate > 0) {
      return rate;
    }
  }
  return 15.0;
}

function chooseFrameKeepRatio(originalFrameRate, goalFrameRate, maxCycleLength = 60) {
  const targetKeepRatio = goalFrameRate / originalFrameRate;
  let bestKeepCount = 1;
  let bestCycleLength = 1;
  let bestError = Math.abs(originalFrameRate - goalFrameRate);

  for (let cycleLength = 1; cycleLength <= maxCycleLength; cycleLength += 1) {
    for (let keepCount = 1; keepCount <= cycleLength; keepCount += 1) {
      const achievedFrameRate = (originalFrameRate * keepCount) / cycleLength;
      const error = Math.abs(achievedFrameRate - goalFrameRate);
      const currentRatio = keepCount / cycleLength;
      const bestRatio = bestKeepCount / bestCycleLength;
      if (
        error < bestError ||
        (error === bestError &&
          Math.abs(currentRatio - targetKeepRatio) <
            Math.abs(bestRatio - targetKeepRatio))
      ) {
        bestKeepCount = keepCount;
        bestCycleLength = cycleLength;
        bestError = error;
      }
    }
  }

  return [bestKeepCount, bestCycleLength];
}

function buildKeepIndices(frameCount, keepCount, cycleLength) {
  const indices = [];
  for (let position = 0; position < frameCount; position += 1) {
    const cyclePosition = position % cycleLength;
    const keep =
      Math.ceil(((cyclePosition + 1) * keepCount) / cycleLength) >
      Math.ceil((cyclePosition * keepCount) / cycleLength);
    if (keep) {
      indices.push(position);
    }
  }
  return Uint32Array.from(indices);
}

export function downsampleRgbForInference(
  rgbImage,
  width,
  height,
  goalFrameRate,
  originalFrameRate,
) {
  if (
    !Number.isFinite(goalFrameRate) ||
    goalFrameRate <= 0 ||
    !Number.isFinite(originalFrameRate) ||
    originalFrameRate <= 0 ||
    originalFrameRate <= goalFrameRate
  ) {
    return {
      rgbImage,
      width,
      height,
      frameIndices: Uint32Array.from({ length: width }, (_, index) => index),
      achievedFrameRate: originalFrameRate,
    };
  }

  const [keepCount, cycleLength] = chooseFrameKeepRatio(
    originalFrameRate,
    goalFrameRate,
  );
  const frameIndices = buildKeepIndices(width, keepCount, cycleLength);
  const downsampled = new Uint8ClampedArray(frameIndices.length * height * 3);

  for (let y = 0; y < height; y += 1) {
    const sourceRowBase = y * width * 3;
    const targetRowBase = y * frameIndices.length * 3;
    for (let x = 0; x < frameIndices.length; x += 1) {
      const sourceBase = sourceRowBase + frameIndices[x] * 3;
      const targetBase = targetRowBase + x * 3;
      downsampled[targetBase] = rgbImage[sourceBase];
      downsampled[targetBase + 1] = rgbImage[sourceBase + 1];
      downsampled[targetBase + 2] = rgbImage[sourceBase + 2];
    }
  }

  return {
    rgbImage: downsampled,
    width: frameIndices.length,
    height,
    frameIndices,
    achievedFrameRate: (originalFrameRate * keepCount) / cycleLength,
  };
}

export function resizeRgbHeightForInference(rgbImage, width, height, goalHeight) {
  if (
    !Number.isFinite(goalHeight) ||
    goalHeight <= 0 ||
    !Number.isInteger(goalHeight) ||
    goalHeight >= height
  ) {
    return {
      rgbImage,
      width,
      height,
    };
  }

  const resized = new Uint8ClampedArray(width * goalHeight * 3);
  for (let targetY = 0; targetY < goalHeight; targetY += 1) {
    const sourceY = Math.max(
      0,
      Math.min(height - 1, Math.round((((targetY + 0.5) * height) / goalHeight) - 0.5)),
    );
    const sourceRowBase = sourceY * width * 3;
    const targetRowBase = targetY * width * 3;
    resized.set(rgbImage.subarray(sourceRowBase, sourceRowBase + width * 3), targetRowBase);
  }

  return {
    rgbImage: resized,
    width,
    height: goalHeight,
  };
}

function mapFrameCoordinateToOriginal(coord, frameIndices) {
  if (!frameIndices || frameIndices.length === 0) {
    return Number.NaN;
  }
  if (coord <= 0) {
    return Number(frameIndices[0]);
  }
  if (coord >= frameIndices.length - 1) {
    return Number(frameIndices[frameIndices.length - 1]);
  }
  const lo = Math.floor(coord);
  const hi = Math.ceil(coord);
  if (lo === hi) {
    return Number(frameIndices[lo]);
  }
  const frac = coord - lo;
  return Number(frameIndices[lo]) * (1 - frac) + Number(frameIndices[hi]) * frac;
}

function mapAxisCoordinateToOriginal(coord, sourceSize, targetSize) {
  if (
    !Number.isFinite(coord) ||
    !Number.isFinite(sourceSize) ||
    !Number.isFinite(targetSize) ||
    sourceSize <= 0 ||
    targetSize <= 0
  ) {
    return Number.NaN;
  }
  if (sourceSize === targetSize) {
    return coord;
  }
  const mapped = (((coord + 0.5) * targetSize) / sourceSize) - 0.5;
  return Math.max(0, Math.min(targetSize - 1, mapped));
}

function maskHorizontalExtentAndCentroid(mask, width, height) {
  if (!mask || width <= 0 || height <= 0 || mask.length !== width * height) {
    return Array(10).fill(Number.NaN);
  }

  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;
  let sumX = 0;
  let sumY = 0;
  let pixelCount = 0;
  const colCounts = new Uint32Array(width);
  const colSumY = new Float64Array(width);

  for (let y = 0; y < height; y += 1) {
    const rowBase = y * width;
    for (let x = 0; x < width; x += 1) {
      if (!mask[rowBase + x]) {
        continue;
      }
      pixelCount += 1;
      sumX += x;
      sumY += y;
      colCounts[x] += 1;
      colSumY[x] += y;
      if (x < minX) {
        minX = x;
      }
      if (x > maxX) {
        maxX = x;
      }
      if (y < minY) {
        minY = y;
      }
      if (y > maxY) {
        maxY = y;
      }
    }
  }

  if (pixelCount === 0) {
    return Array(10).fill(Number.NaN);
  }

  const averageBinY = sumY / pixelCount;
  const duration = maxX - minX + 1;
  const centerTarget = Math.round(sumX / pixelCount);
  let centerX = centerTarget;

  if (colCounts[centerX] === 0) {
    const maxDelta = Math.max(centerTarget - minX, maxX - centerTarget);
    for (let delta = 0; delta <= maxDelta; delta += 1) {
      const candidates = delta === 0 ? [centerTarget] : [centerTarget - delta, centerTarget + delta];
      let chosen = false;
      for (const candidate of candidates) {
        if (candidate < minX || candidate > maxX) {
          continue;
        }
        if (colCounts[candidate] > 0) {
          centerX = candidate;
          chosen = true;
          break;
        }
      }
      if (chosen) {
        break;
      }
    }
  }

  const centerFrameBin = colSumY[centerX] / colCounts[centerX];
  const startBinY = colSumY[minX] / colCounts[minX];
  const endBinY = colSumY[maxX] / colCounts[maxX];
  return [
    minX,
    maxX,
    centerX,
    centerFrameBin,
    minY,
    maxY,
    averageBinY,
    duration,
    startBinY,
    endBinY,
  ];
}

export function convertBinToM(binValue, metadata) {
  if (!metadata) {
    return Number.NaN;
  }
  const x = asFloat(binValue);
  if (Number.isNaN(x)) {
    return Number.NaN;
  }

  if ("windowstart" in metadata && "windowlength" in metadata) {
    const numBins = metadata.samplesperbeam || metadata.samplesperchannel;
    if (!numBins) {
      return Number.NaN;
    }
    const binSize = Number(metadata.windowlength) / Number(numBins);
    return (
      Number(metadata.windowlength) -
      x * binSize +
      Number(metadata.windowstart)
    );
  }

  const sampleLength = metadata.sample_length;
  if (sampleLength !== undefined && sampleLength !== null) {
    let winStart = metadata.WinStart;
    if (winStart === undefined || winStart === null) {
      const delay = metadata.samplestartdelay;
      const speed = metadata.soundspeed;
      if (delay !== undefined && delay !== null && speed !== undefined && speed !== null) {
        winStart = Number(delay) * 1e-6 * Number(speed) / 2.0;
      }
    }
    if (winStart !== undefined && winStart !== null) {
      return Number(winStart) + x * Number(sampleLength);
    }
  }

  return Number.NaN;
}

export function buildPredictionRows(
  detections,
  echogramMetadata,
  frameIndices = null,
  originalHeight = null,
) {
  const rows = [];
  const mappedFrameIndices = frameIndices && frameIndices.length > 0 ? frameIndices : null;

  for (let index = 0; index < detections.length; index += 1) {
    const detection = detections[index];
    const width = detection.analysisWidth ?? detection.maskWidth;
    const height = detection.analysisHeight ?? detection.maskHeight;
    const mask = detection.analysisMask ?? detection.mask;
    const [
      enterFrame,
      exitFrame,
      centerFrame,
      centerFrameBin,
      minimumBinY,
      maximumBinY,
      averageBinY,
      duration,
      startBinY,
      endBinY,
    ] = maskHorizontalExtentAndCentroid(mask, width, height);

    let mappedCenterFrameBin = centerFrameBin;
    let mappedMinimumBinY = minimumBinY;
    let mappedMaximumBinY = maximumBinY;
    let mappedAverageBinY = averageBinY;
    let mappedStartBinY = startBinY;
    let mappedEndBinY = endBinY;

    let mappedEnterFrame = enterFrame;
    let mappedExitFrame = exitFrame;
    let mappedCenterFrame = centerFrame;
    let mappedDuration = duration;
    if (mappedFrameIndices) {
      mappedEnterFrame = mapFrameCoordinateToOriginal(mappedEnterFrame, mappedFrameIndices);
      mappedExitFrame = mapFrameCoordinateToOriginal(mappedExitFrame, mappedFrameIndices);
      mappedCenterFrame = mapFrameCoordinateToOriginal(mappedCenterFrame, mappedFrameIndices);
      mappedDuration = Math.round(mappedExitFrame) - Math.round(mappedEnterFrame) + 1;
    }

    if (originalHeight !== null && originalHeight !== height) {
      mappedCenterFrameBin = mapAxisCoordinateToOriginal(centerFrameBin, height, originalHeight);
      mappedMinimumBinY = mapAxisCoordinateToOriginal(minimumBinY, height, originalHeight);
      mappedMaximumBinY = mapAxisCoordinateToOriginal(maximumBinY, height, originalHeight);
      mappedAverageBinY = mapAxisCoordinateToOriginal(averageBinY, height, originalHeight);
      mappedStartBinY = mapAxisCoordinateToOriginal(startBinY, height, originalHeight);
      mappedEndBinY = mapAxisCoordinateToOriginal(endBinY, height, originalHeight);
    }

    const centerFrameDistance = convertBinToM(mappedCenterFrameBin, echogramMetadata);
    const startDistance = convertBinToM(mappedStartBinY, echogramMetadata);
    const endDistance = convertBinToM(mappedEndBinY, echogramMetadata);
    const minimumDistance = convertBinToM(mappedMinimumBinY, echogramMetadata);
    const maximumDistance = convertBinToM(mappedMaximumBinY, echogramMetadata);
    const averageDistance = convertBinToM(mappedAverageBinY, echogramMetadata);

    rows.push({
      instance_index: roundInt(index),
      confidence: confidenceCell(detection.score),
      class_id: roundInt(detection.classId),
      class_name: detection.className ?? String(detection.classId),
      enter_frame: roundInt(mappedEnterFrame),
      exit_frame: roundInt(mappedExitFrame),
      center_frame: roundInt(mappedCenterFrame),
      center_frame_bin: roundInt(mappedCenterFrameBin),
      center_frame_distance: roundDecimal(centerFrameDistance, 2),
      duration: roundInt(mappedDuration),
      minimum_bin_y: roundInt(mappedMinimumBinY),
      maximum_bin_y: roundInt(mappedMaximumBinY),
      average_bin_y: roundInt(mappedAverageBinY),
      start_bin_y: roundInt(mappedStartBinY),
      end_bin_y: roundInt(mappedEndBinY),
      minimum_distance: roundDecimal(minimumDistance, 2),
      maximum_distance: roundDecimal(maximumDistance, 2),
      average_distance: roundDecimal(averageDistance, 2),
      start_distance: roundDecimal(startDistance, 2),
      end_distance: roundDecimal(endDistance, 2),
    });
  }

  return rows;
}

export function buildDisplayRows(predictionRows, upstreamDirection) {
  return predictionRows
    .map((row) => {
      const frameNumber = asFloat(row.center_frame);
      const rangeMeters = asFloat(row.center_frame_distance);
      return {
        frameNumber: Number.isNaN(frameNumber) ? null : Math.round(frameNumber),
        direction:
          fcDirectionForCrossing(row.class_name, asFloat(row.class_id), upstreamDirection) || "--",
        rangeMeters: Number.isNaN(rangeMeters) ? null : Number(rangeMeters.toFixed(2)),
        confidence: String(row.confidence ?? "nan"),
      };
    })
    .sort((a, b) => {
      if (a.frameNumber === null && b.frameNumber === null) {
        return 0;
      }
      if (a.frameNumber === null) {
        return 1;
      }
      if (b.frameNumber === null) {
        return -1;
      }
      return a.frameNumber - b.frameNumber;
    })
    .map((row, index) => ({ ...row, index: index + 1 }));
}

function csvEscape(value) {
  const text =
    value === null || value === undefined || (typeof value === "number" && Number.isNaN(value))
      ? "nan"
      : String(value);
  if (text.includes(",") || text.includes("\"") || text.includes("\n")) {
    return `"${text.replaceAll("\"", "\"\"")}"`;
  }
  return text;
}

export function formatPredictionCsv(rows) {
  const header = PREDICTION_CSV_FIELDNAMES.join(",");
  const body = rows.map((row) =>
    PREDICTION_CSV_FIELDNAMES.map((field) => csvEscape(row[field])).join(","),
  );
  return [header, ...body].join("\n") + "\n";
}

function dateFromFilename(filename) {
  const match = String(filename).match(/(\d{4}-\d{2}-\d{2})/);
  if (match) {
    return match[1];
  }
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function buildFcRecords(rows, filename, upstreamDirection) {
  const defaults = Object.fromEntries(
    Object.entries(FC_SCHEMA).map(([name, spec]) => [name, spec.default]),
  );
  const date = dateFromFilename(filename);
  const fcRows = [];

  for (const row of rows) {
    const className = String(row.class_name ?? "");
    const classId = roundInt(row.class_id);
    if (isNoCrossDetection(className, classId)) {
      continue;
    }

    const rangeMeters = asFloat(row.center_frame_distance);
    if (Number.isNaN(rangeMeters)) {
      continue;
    }

    const minimumDistance = asFloat(row.minimum_distance);
    const maximumDistance = asFloat(row.maximum_distance);
    const startDistance = asFloat(row.start_distance);
    const endDistance = asFloat(row.end_distance);
    const dRCm =
      Number.isNaN(minimumDistance) || Number.isNaN(maximumDistance)
        ? 0
        : Math.max((maximumDistance - minimumDistance) * 100.0, 0.0);
    const lengthCm =
      Number.isNaN(startDistance) || Number.isNaN(endDistance)
        ? 0
        : Math.abs(endDistance - startDistance) * 100.0;
    const lengthOverRange = dRCm > 0 ? lengthCm / dRCm : 0;
    const frameNumber = Number.isNaN(asFloat(row.center_frame))
      ? 0
      : Math.round(asFloat(row.center_frame));
    const direction = fcDirectionForCrossing(className, classId, upstreamDirection);
    if (!direction) {
      continue;
    }

    fcRows.push({
      ...defaults,
      File: 1,
      "Frame#": frameNumber,
      Dir: direction.slice(0, FC_SCHEMA.Dir.width),
      "R (m)": roundDecimal(rangeMeters, 2),
      Theta: defaults.Theta,
      "L(cm)": roundDecimal(lengthCm, 2),
      "dR(cm)": roundDecimal(dRCm, 2),
      "L/dR": roundDecimal(lengthOverRange, 2),
      Aspect: roundDecimal(lengthOverRange, 2),
      Date: date,
      Species: String(defaults.Species).slice(0, FC_SCHEMA.Species.width),
      Comment: formatFcComment(row.confidence),
    });
  }

  fcRows.sort((a, b) => Number(a["Frame#"]) - Number(b["Frame#"]));
  for (let index = 0; index < fcRows.length; index += 1) {
    fcRows[index].Total = index + 1;
  }
  return fcRows;
}

export function formatFcFile(rows, filename, upstreamDirection) {
  const fcRows = buildFcRecords(rows, filename, upstreamDirection);
  const headerLine = FC_HEADERS.map((header) =>
    String(header).padStart(FC_SCHEMA[header].width, " "),
  ).join("");
  const separatorLine = "-".repeat(headerLine.length);
  const lines = [FC_TITLE, "", headerLine, separatorLine];

  for (const record of fcRows) {
    lines.push(
      FC_HEADERS.map((header) => {
        const value = record[header] ?? FC_SCHEMA[header].default;
        return String(value).padStart(FC_SCHEMA[header].width, " ");
      }).join(""),
    );
  }

  return lines.join("\n") + "\n";
}

function arisDurationMinutes(metadata) {
  if (!metadata) {
    return null;
  }
  const numFrames = asFloat(metadata.numframes);
  const cyclePeriod = asFloat(metadata.cycleperiod);
  const samplePeriod = asFloat(metadata.sampleperiod);
  if (Number.isNaN(numFrames) || Number.isNaN(cyclePeriod) || Number.isNaN(samplePeriod) || numFrames < 1) {
    return null;
  }
  const durationSeconds = (numFrames - 1) * cyclePeriod * samplePeriod / 1_000_000;
  return durationSeconds / 60.0;
}

function echotasticTotalTimeHeader(metadata) {
  const minutes = arisDurationMinutes(metadata);
  if (minutes === null) {
    return "";
  }
  return `${minutes.toFixed(3)} minutes`;
}

function numBinsFromMetadata(metadata) {
  if (!metadata) {
    return null;
  }
  for (const key of ["samplesperbeam", "samplesperchannel", "ydim"]) {
    if (!(key in metadata)) {
      continue;
    }
    const count = roundInt(metadata[key]);
    if (Number.isFinite(count) && count > 0) {
      return count;
    }
  }
  return null;
}

function echotasticSample(binIndex, metadata) {
  const bin = Number.isNaN(asFloat(binIndex)) ? 0 : Math.round(asFloat(binIndex));
  const numBins = numBinsFromMetadata(metadata);
  if (numBins !== null) {
    return numBins - bin;
  }
  return bin;
}

function echotasticTimeSeconds(ping, frameRate) {
  if (!Number.isFinite(frameRate) || frameRate <= 0) {
    return 0.0;
  }
  return ping / frameRate / 60.0;
}

function echotasticHeaderDateTime(stem) {
  const match = String(stem).match(/_(\d{4})-(\d{2})-(\d{2})_(\d{6})$/);
  if (!match) {
    return ["", ""];
  }
  const [, year, month, day, hhmmss] = match;
  const date = `${month}/${day}/${year}`;
  if (hhmmss.length !== 6) {
    return [date, ""];
  }
  return [date, `${hhmmss.slice(0, 2)}:${hhmmss.slice(2, 4)}:${hhmmss.slice(4, 6)}`];
}

function buildEchotasticRows(rows, upstreamDirection, echogramMetadata) {
  const frameRate = framerateFromMetadata(echogramMetadata);
  const echotasticRows = [];

  for (const row of rows) {
    const className = String(row.class_name ?? "");
    const classId = roundInt(row.class_id);
    if (isNoCrossDetection(className, classId)) {
      continue;
    }

    const direction = echotasticDirection(className, classId, upstreamDirection);
    const rangeMeters = asFloat(row.center_frame_distance);
    if (direction === null || Number.isNaN(rangeMeters)) {
      continue;
    }

    const ping = Number.isNaN(asFloat(row.center_frame)) ? 0 : Math.round(asFloat(row.center_frame));
    const sample = echotasticSample(row.center_frame_bin, echogramMetadata);
    echotasticRows.push({
      Sample: sample,
      Ping: ping,
      Time: echotasticTimeSeconds(ping, frameRate),
      Range: rangeMeters,
      Amplitude: 0.0,
      XAngle: 0.0,
      YAngle: 0.0,
      Direction: direction,
      Length: 0.0,
      Area: 0.0,
      Operator: ECHOTASTIC_OPERATOR,
    });
  }

  echotasticRows.sort((a, b) => {
    if (a.Ping !== b.Ping) {
      return a.Ping - b.Ping;
    }
    return a.Sample - b.Sample;
  });
  return echotasticRows;
}

export function formatEchotasticFile(
  rows,
  arisName,
  arisStem,
  echogramMetadata,
  upstreamDirection,
) {
  const dataRows = buildEchotasticRows(rows, upstreamDirection, echogramMetadata);
  const [dateString, startString] = echotasticHeaderDateTime(arisStem);
  const headerLines = [
    `Version = ${ECHOTASTIC_VERSION}`,
    `File Name = ${arisName}`,
    `Total Number Of Marks = ${dataRows.length}`,
    `Total Time = ${echotasticTotalTimeHeader(echogramMetadata)}`,
    `Date = ${dateString}`,
    `Start Time = ${startString}`,
    "",
    ECHOTASTIC_COLUMNS.join("\t"),
  ];
  const bodyLines = dataRows.map((record) =>
    [
      record.Sample,
      record.Ping,
      record.Time.toFixed(2),
      record.Range.toFixed(2),
      record.Amplitude.toFixed(2),
      record.XAngle.toFixed(2),
      record.YAngle.toFixed(2),
      record.Direction,
      record.Length.toFixed(2),
      record.Area.toFixed(2),
      record.Operator,
    ].join("\t"),
  );
  return [...headerLines, ...bodyLines].join("\n") + "\n";
}

export function csvFilename(baseName) {
  return `${baseName}_predictions.csv`;
}

export function fcFilename(arisStem) {
  return `FCe_${arisStem}_ID_.txt`;
}

export function echotasticFilename(arisStem) {
  return `${arisStem}.aris.txt`;
}
