import { buildInstanceColours, clampClassId } from "./classColours.js";

function clampChannel(value) {
  if (value <= 0) {
    return 0;
  }
  if (value >= 255) {
    return 255;
  }
  return Math.round(value);
}

function drawMaskOverlay(image, mask, color, alpha = 0.75) {
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) {
      continue;
    }
    const pixelBase = index * 4;
    image[pixelBase] = clampChannel(
      image[pixelBase] * (1 - alpha) + color[0] * alpha,
    );
    image[pixelBase + 1] = clampChannel(
      image[pixelBase + 1] * (1 - alpha) + color[1] * alpha,
    );
    image[pixelBase + 2] = clampChannel(
      image[pixelBase + 2] * (1 - alpha) + color[2] * alpha,
    );
  }
}

function grayscaleBaseFromBgr(imageBgr, width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (
    let pixel = 0, src = 0, dst = 0;
    pixel < width * height;
    pixel += 1, src += 3, dst += 4
  ) {
    const gray = imageBgr[src + 2];
    rgba[dst] = gray;
    rgba[dst + 1] = gray;
    rgba[dst + 2] = gray;
    rgba[dst + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}

function grayscaleBaseFromRgb(rgbImage, width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (
    let pixel = 0, src = 0, dst = 0;
    pixel < width * height;
    pixel += 1, src += 3, dst += 4
  ) {
    const gray = rgbImage[src];
    rgba[dst] = gray;
    rgba[dst + 1] = gray;
    rgba[dst + 2] = gray;
    rgba[dst + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}

function assignDetectionColours(detections) {
  const grouped = new Map();

  for (const detection of detections) {
    const classId = clampClassId(detection.classId);
    detection.classId = classId;
    const bucket = grouped.get(classId) ?? [];
    bucket.push(detection);
    grouped.set(classId, bucket);
  }

  for (const [classId, bucket] of grouped.entries()) {
    const colours = buildInstanceColours(classId, bucket.length);
    for (let index = 0; index < bucket.length; index += 1) {
      bucket[index].color = colours[index];
    }
  }
}

export function makeOverlayImageFromBase(baseImageData, detections) {
  const rgba = new Uint8ClampedArray(baseImageData.data);
  assignDetectionColours(detections);
  for (const detection of detections) {
    drawMaskOverlay(rgba, detection.mask, detection.color);
  }

  return new ImageData(rgba, baseImageData.width, baseImageData.height);
}

export function makeOverlayBaseImage(imageBgr, width, height) {
  return grayscaleBaseFromBgr(imageBgr, width, height);
}

export function makeOverlayImage(imageBgr, width, height, detections) {
  return makeOverlayImageFromBase(grayscaleBaseFromBgr(imageBgr, width, height), detections);
}

export function makeOverlayBaseImageFromRgb(rgbImage, width, height) {
  return grayscaleBaseFromRgb(rgbImage, width, height);
}

export function makeOverlayImageFromRgb(rgbImage, width, height, detections) {
  return makeOverlayImageFromBase(grayscaleBaseFromRgb(rgbImage, width, height), detections);
}
