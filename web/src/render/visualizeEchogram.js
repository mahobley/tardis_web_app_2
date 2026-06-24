function clamp01(value) {
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function numpyToRedBlue(values) {
  const out = new Float32Array(values.length * 3);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const base = index * 3;
    if (value <= 0) {
      out[base] = 1;
      out[base + 1] = clamp01(1 + value);
      out[base + 2] = clamp01(1 + value);
    } else {
      out[base] = clamp01(1 - value);
      out[base + 1] = clamp01((1 - value) * 0.5 + 128 / 255);
      out[base + 2] = 1;
    }
  }
  return out;
}

export function makeEchogramVisualFromBgr(
  imageBgr,
  width,
  height,
  { colourPower = 2, colourMaskPower = 2 } = {},
) {
  const pixelCount = width * height;
  const ecMag = new Float32Array(pixelCount);
  const ecAngle = new Float32Array(pixelCount);

  let minMag = Number.POSITIVE_INFINITY;
  let maxMag = Number.NEGATIVE_INFINITY;
  for (let pixel = 0, src = 0; pixel < pixelCount; pixel += 1, src += 3) {
    const mag = imageBgr[src + 2];
    ecMag[pixel] = mag;
    ecAngle[pixel] = imageBgr[src + 1] / 255 - 0.5;
    if (mag < minMag) {
      minMag = mag;
    }
    if (mag > maxMag) {
      maxMag = mag;
    }
  }

  const magRange = maxMag - minMag;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    ecMag[pixel] = magRange > 0 ? (ecMag[pixel] - minMag) / magRange : 0;
  }

  const angleScaled = new Float32Array(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    angleScaled[pixel] = ecAngle[pixel] * colourPower;
  }

  const colmapped = numpyToRedBlue(angleScaled);
  const out = new Uint8ClampedArray(pixelCount * 3);

  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const mask = ecMag[pixel] ** colourMaskPower;
    const gray = ecMag[pixel];
    const base = pixel * 3;

    for (let channel = 0; channel < 3; channel += 1) {
      let value = gray * (1 - mask) + colmapped[base + channel] * mask;
      value = 0.5 + 1.5 * (value - 0.5);
      value = clamp01(value);
      out[base + channel] = Math.round(value * 255);
    }
  }

  return out;
}
