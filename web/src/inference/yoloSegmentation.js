import * as ort from "onnxruntime-web";
import { clampClassId } from "../render/classColours.js";
import { rgbToImageData } from "../render/imageData.js";

const CLASS_NAMES = ["pos", "neg", "no-cross"];

function createCanvas(width, height) {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function letterboxShape(height, width, stride = 32) {
  return [
    Math.ceil(height / stride) * stride,
    Math.ceil(width / stride) * stride,
  ];
}

function prepareLetterboxCanvas(rgbImage, width, height, targetHeight, targetWidth) {
  const sourceCanvas = createCanvas(width, height);
  const sourceCtx = sourceCanvas.getContext("2d");
  sourceCtx.putImageData(rgbToImageData(rgbImage, width, height), 0, 0);

  const scale = Math.min(targetWidth / width, targetHeight / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const padLeft = Math.round((targetWidth - resizedWidth) / 2 - 0.1);
  const padTop = Math.round((targetHeight - resizedHeight) / 2 - 0.1);

  const targetCanvas = createCanvas(targetWidth, targetHeight);
  const targetCtx = targetCanvas.getContext("2d", { willReadFrequently: true });
  targetCtx.fillStyle = "rgb(114 114 114)";
  targetCtx.fillRect(0, 0, targetWidth, targetHeight);
  targetCtx.drawImage(
    sourceCanvas,
    0,
    0,
    width,
    height,
    padLeft,
    padTop,
    resizedWidth,
    resizedHeight,
  );

  return {
    canvas: targetCanvas,
    ctx: targetCtx,
    scale,
    resizedWidth,
    resizedHeight,
    padLeft,
    padTop,
    targetWidth,
    targetHeight,
  };
}

function canvasToChwTensor(ctx, width, height) {
  const { data } = ctx.getImageData(0, 0, width, height);
  const chw = new Float32Array(3 * width * height);
  const channelSize = width * height;

  for (let pixel = 0, dataIndex = 0; pixel < channelSize; pixel += 1, dataIndex += 4) {
    chw[pixel] = data[dataIndex] / 255;
    chw[channelSize + pixel] = data[dataIndex + 1] / 255;
    chw[2 * channelSize + pixel] = data[dataIndex + 2] / 255;
  }

  return chw;
}

function boxIou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const intersection = interW * interH;
  if (intersection <= 0) {
    return 0;
  }
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

function nonMaxSuppression(detections, iouThreshold) {
  const byScore = [...detections].sort((a, b) => b.score - a.score);
  const kept = [];

  while (byScore.length > 0) {
    const candidate = byScore.shift();
    kept.push(candidate);
    for (let index = byScore.length - 1; index >= 0; index -= 1) {
      const other = byScore[index];
      if (candidate.classId !== other.classId) {
        continue;
      }
      if (boxIou(candidate.boxInput, other.boxInput) > iouThreshold) {
        byScore.splice(index, 1);
      }
    }
  }

  return kept;
}

function clipBox(box, width, height) {
  return [
    Math.max(0, Math.min(width, box[0])),
    Math.max(0, Math.min(height, box[1])),
    Math.max(0, Math.min(width, box[2])),
    Math.max(0, Math.min(height, box[3])),
  ];
}

function scaleBoxToImage(boxInput, letterbox, sourceWidth, sourceHeight) {
  return clipBox(
    [
      (boxInput[0] - letterbox.padLeft) / letterbox.scale,
      (boxInput[1] - letterbox.padTop) / letterbox.scale,
      (boxInput[2] - letterbox.padLeft) / letterbox.scale,
      (boxInput[3] - letterbox.padTop) / letterbox.scale,
    ],
    sourceWidth,
    sourceHeight,
  );
}

function rescaleBox(box, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) {
    return [...box];
  }
  const scaleX = targetWidth / sourceWidth;
  const scaleY = targetHeight / sourceHeight;
  return clipBox(
    [
      box[0] * scaleX,
      box[1] * scaleY,
      box[2] * scaleX,
      box[3] * scaleY,
    ],
    targetWidth,
    targetHeight,
  );
}

function resizeMaskToDimensions(
  mask,
  protoWidth,
  protoHeight,
  letterbox,
  sourceWidth,
  sourceHeight,
  targetWidth,
  targetHeight,
) {
  const binaryMask = new Uint8Array(targetWidth * targetHeight);

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = ((y + 0.5) * sourceHeight) / targetHeight - 0.5;
    const letterboxY =
      letterbox.padTop +
      ((sourceY + 0.5) * letterbox.resizedHeight) / sourceHeight -
      0.5;
    const protoY =
      ((letterboxY + 0.5) * protoHeight) / letterbox.targetHeight - 0.5;

    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = ((x + 0.5) * sourceWidth) / targetWidth - 0.5;
      const letterboxX =
        letterbox.padLeft +
        ((sourceX + 0.5) * letterbox.resizedWidth) / sourceWidth -
        0.5;
      const protoX =
        ((letterboxX + 0.5) * protoWidth) / letterbox.targetWidth - 0.5;
      const value = sampleBilinear(mask, protoWidth, protoHeight, protoX, protoY);
      binaryMask[y * targetWidth + x] = value > 0 ? 1 : 0;
    }
  }

  return binaryMask;
}

function resizeMaskToOriginal(mask, protoWidth, protoHeight, letterbox, originalWidth, originalHeight) {
  return resizeMaskToDimensions(
    mask,
    protoWidth,
    protoHeight,
    letterbox,
    originalWidth,
    originalHeight,
    originalWidth,
    originalHeight,
  );
}

function scaleBoxToOriginal(boxInput, letterbox, originalWidth, originalHeight) {
  const scaled = [
    (boxInput[0] - letterbox.padLeft) / letterbox.scale,
    (boxInput[1] - letterbox.padTop) / letterbox.scale,
    (boxInput[2] - letterbox.padLeft) / letterbox.scale,
    (boxInput[3] - letterbox.padTop) / letterbox.scale,
  ];
  return clipBox(scaled, originalWidth, originalHeight);
}

function cropMask(mask, width, height, box) {
  const [x1, y1, x2, y2] = box.map((value) => Math.max(0, Math.round(value)));
  const out = new Float32Array(mask);

  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * width;
    const keepRow = y >= y1 && y < y2;
    for (let x = 0; x < width; x += 1) {
      if (!keepRow || x < x1 || x >= x2) {
        out[rowOffset + x] = 0;
      }
    }
  }

  return out;
}

function decodeMask(protoData, protoWidth, protoHeight, coeffs, boxInput, inputWidth, inputHeight) {
  const maskPixels = protoWidth * protoHeight;
  const mask = new Float32Array(maskPixels);

  for (let channel = 0; channel < coeffs.length; channel += 1) {
    const coeff = coeffs[channel];
    const protoOffset = channel * maskPixels;
    for (let pixel = 0; pixel < maskPixels; pixel += 1) {
      mask[pixel] += coeff * protoData[protoOffset + pixel];
    }
  }

  const widthRatio = protoWidth / inputWidth;
  const heightRatio = protoHeight / inputHeight;
  const cropped = cropMask(mask, protoWidth, protoHeight, [
    boxInput[0] * widthRatio,
    boxInput[1] * heightRatio,
    boxInput[2] * widthRatio,
    boxInput[3] * heightRatio,
  ]);

  return cropped;
}

function sampleBilinear(mask, width, height, x, y) {
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const x1 = Math.max(0, Math.min(width - 1, x0 + 1));
  const y1 = Math.max(0, Math.min(height - 1, y0 + 1));
  const dx = x - x0;
  const dy = y - y0;

  const topLeft = mask[y0 * width + x0];
  const topRight = mask[y0 * width + x1];
  const bottomLeft = mask[y1 * width + x0];
  const bottomRight = mask[y1 * width + x1];

  const top = topLeft * (1 - dx) + topRight * dx;
  const bottom = bottomLeft * (1 - dx) + bottomRight * dx;
  return top * (1 - dy) + bottom * dy;
}

async function loadModelSource(modelSource) {
  if (modelSource instanceof File || modelSource instanceof Blob) {
    return modelSource.arrayBuffer();
  }
  if (typeof modelSource === "string") {
    const response = await fetch(modelSource);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }
    return response.arrayBuffer();
  }
  if (modelSource instanceof ArrayBuffer) {
    return modelSource;
  }
  throw new TypeError("Unsupported model source");
}

export function defaultModelUrl() {
  return new URL("../../../weights/noklamath.onnx", import.meta.url).href;
}

export async function createSegmentationSession(modelSource, backend = "webgpu") {
  const executionProviders =
    backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  const modelData = await loadModelSource(modelSource);
  return ort.InferenceSession.create(modelData, { executionProviders });
}

export async function runYoloSegmentation({
  session,
  rgbImage,
  width,
  height,
  outputWidth = width,
  outputHeight = height,
  confidence = 0.1,
  iouThreshold = 0.5,
}) {
  const [inputHeight, inputWidth] = letterboxShape(height, width, 32);
  const letterbox = prepareLetterboxCanvas(
    rgbImage,
    width,
    height,
    inputHeight,
    inputWidth,
  );
  const inputTensor = new ort.Tensor(
    "float32",
    canvasToChwTensor(letterbox.ctx, inputWidth, inputHeight),
    [1, 3, inputHeight, inputWidth],
  );

  const { output0, output1 } = await session.run({ images: inputTensor });
  const detections = [];
  const protoData = output1.data;
  const protoHeight = output1.dims[2];
  const protoWidth = output1.dims[3];
  const rowWidth = output0.dims[2];
  const coeffOffset = 6;

  for (let row = 0; row < output0.dims[1]; row += 1) {
    const base = row * rowWidth;
    const score = output0.data[base + 4];
    if (score < confidence) {
      continue;
    }

    const classId = clampClassId(Math.round(output0.data[base + 5]));
    const boxInput = [
      output0.data[base],
      output0.data[base + 1],
      output0.data[base + 2],
      output0.data[base + 3],
    ];
    const coeffs = output0.data.slice(base + coeffOffset, base + rowWidth);

    detections.push({
      score,
      classId,
      className: CLASS_NAMES[classId] ?? String(classId),
      boxInput,
      coeffs,
    });
  }

  const suppressed = nonMaxSuppression(detections, iouThreshold);
  const finalized = suppressed.map((detection) => {
    const sourceBox = scaleBoxToImage(
      detection.boxInput,
      letterbox,
      width,
      height,
    );
    const lowResMask = decodeMask(
      protoData,
      protoWidth,
      protoHeight,
      detection.coeffs,
      detection.boxInput,
      inputWidth,
      inputHeight,
    );
    const analysisMask = resizeMaskToDimensions(
      lowResMask,
      protoWidth,
      protoHeight,
      letterbox,
      width,
      height,
      width,
      height,
    );
    const outputMask =
      outputWidth === width && outputHeight === height
        ? analysisMask
        : resizeMaskToDimensions(
            lowResMask,
            protoWidth,
            protoHeight,
            letterbox,
            width,
            height,
            outputWidth,
            outputHeight,
          );
    return {
      ...detection,
      box: rescaleBox(sourceBox, width, height, outputWidth, outputHeight),
      analysisMask,
      analysisWidth: width,
      analysisHeight: height,
      mask: outputMask,
      maskWidth: outputWidth,
      maskHeight: outputHeight,
    };
  });

  return {
    inputWidth,
    inputHeight,
    letterbox,
    detections: finalized,
  };
}
