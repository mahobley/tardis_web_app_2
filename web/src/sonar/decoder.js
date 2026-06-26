const FILE_SCHEMAS = {
  3: [
    ["version", "4s"],
    ["numframes", "i"],
    ["framerate", "i"],
    ["resolution", "i"],
    ["numbeams", "i"],
    ["samplerate", "f"],
    ["samplesperchannel", "i"],
    ["receivergain", "i"],
    ["windowstart", "i"],
    ["windowlength", "i"],
    ["reverse", "i"],
    ["serialnumber", "i"],
    ["date", "32s"],
    ["idstring", "256s"],
    ["id1", "i"],
    ["id2", "i"],
    ["id3", "i"],
    ["id4", "i"],
    ["startframe", "i"],
    ["endframe", "i"],
    ["timelapse", "i"],
    ["recordInterval", "i"],
    ["radioseconds", "i"],
    ["frameinterval", "i"],
    ["userassigned", "136s"],
  ],
  4: [
    ["version", "4s"],
    ["numframes", "i"],
    ["framerate", "i"],
    ["resolution", "i"],
    ["numbeams", "i"],
    ["samplerate", "f"],
    ["samplesperchannel", "i"],
    ["receivergain", "i"],
    ["windowstart", "i"],
    ["windowlength", "i"],
    ["reverse", "i"],
    ["serialnumber", "i"],
    ["date", "32s"],
    ["idstring", "256s"],
    ["id1", "i"],
    ["id2", "i"],
    ["id3", "i"],
    ["id4", "i"],
    ["startframe", "i"],
    ["endframe", "i"],
    ["timelapse", "i"],
    ["recordInterval", "i"],
    ["radioseconds", "i"],
    ["frameinterval", "i"],
    ["userassigned", "136s"],
  ],
  5: [
    ["version", "4s"],
    ["numframes", "I"],
    ["framerate", "I"],
    ["resolution", "I"],
    ["numbeams", "I"],
    ["samplerate", "f"],
    ["samplesperchannel", "I"],
    ["receivergain", "I"],
    ["windowstart", "f"],
    ["windowlength", "f"],
    ["reverse", "I"],
    ["serialnumber", "I"],
    ["strdate", "32s"],
    ["idstring", "256s"],
    ["id1", "i"],
    ["id2", "i"],
    ["id3", "i"],
    ["id4", "i"],
    ["startframe", "I"],
    ["endframe", "I"],
    ["timelapse", "I"],
    ["recordInterval", "I"],
    ["radioseconds", "I"],
    ["frameinterval", "I"],
    ["flags", "I"],
    ["auxflags", "I"],
    ["sspd", "I"],
    ["flags3d", "I"],
    ["softwareversion", "I"],
    ["watertemperature", "I"],
    ["salinity", "I"],
    ["pulselength", "I"],
    ["txmode", "I"],
    ["versionfgpa", "I"],
    ["versionpsuc", "I"],
    ["thumbnailfi", "I"],
    ["filesize", "Q"],
    ["optionalheadersize", "Q"],
    ["optionaltailsize", "Q"],
    ["versionminor", "I"],
    ["largelens", "I"],
    ["userassigned", "568s"],
  ],
};

const FRAME_SCHEMAS = {
  3: [
    ["framenumber", "i"],
    ["frametime", "i"],
    ["frametime2", "i"],
    ["version", "4s"],
    ["status", "i"],
    ["year", "i"],
    ["month", "i"],
    ["day", "i"],
    ["hour", "i"],
    ["minute", "i"],
    ["second", "i"],
    ["hsecond", "i"],
    ["transmit", "i"],
    ["windowstart", "i"],
    ["windowlength", "i"],
    ["threshold", "i"],
    ["intensity", "i"],
    ["receivergain", "i"],
    ["degc1", "i"],
    ["degc2", "i"],
    ["humidity", "i"],
    ["focus", "i"],
    ["battery", "i"],
    ["status1", "16s"],
    ["status2", "8s"],
    ["panwcom", "f"],
    ["tiltwcom", "f"],
    ["velocity", "f"],
    ["depth", "f"],
    ["altitude", "f"],
    ["pitch", "f"],
    ["pitchrate", "f"],
    ["roll", "f"],
    ["rollrate", "f"],
    ["heading", "f"],
    ["headingrate", "f"],
    ["sonarpan", "f"],
    ["sonartilt", "f"],
    ["sonarroll", "f"],
    ["latitude", "d"],
    ["longitude", "d"],
    ["sonarposition", "f"],
    ["configflags", "i"],
    ["userassigned", "60s"],
  ],
  4: [
    ["framenumber", "i"],
    ["frametime", "i"],
    ["frametime2", "i"],
    ["version", "4s"],
    ["status", "i"],
    ["year", "i"],
    ["month", "i"],
    ["day", "i"],
    ["hour", "i"],
    ["minute", "i"],
    ["second", "i"],
    ["hsecond", "i"],
    ["transmit", "i"],
    ["windowstart", "i"],
    ["windowlength", "i"],
    ["threshold", "i"],
    ["intensity", "i"],
    ["receivergain", "i"],
    ["degc1", "i"],
    ["degc2", "i"],
    ["humidity", "i"],
    ["focus", "i"],
    ["battery", "i"],
    ["status1", "16s"],
    ["status2", "8s"],
    ["panwcom", "f"],
    ["tiltwcom", "f"],
    ["velocity", "f"],
    ["depth", "f"],
    ["altitude", "f"],
    ["pitch", "f"],
    ["pitchrate", "f"],
    ["roll", "f"],
    ["rollrate", "f"],
    ["heading", "f"],
    ["headingrate", "f"],
    ["sonarpan", "f"],
    ["sonartilt", "f"],
    ["sonarroll", "f"],
    ["latitude", "d"],
    ["longitude", "d"],
    ["sonarposition", "f"],
    ["configflags", "i"],
    ["userassigned", "828s"],
  ],
  5: [
    ["framenumber", "I"],
    ["frametime", "Q"],
    ["version", "4s"],
    ["status", "I"],
    ["sonartimestamp", "Q"],
    ["tsday", "I"],
    ["tshour", "I"],
    ["tsminute", "I"],
    ["tssecond", "I"],
    ["tshsecond", "I"],
    ["transmitmode", "I"],
    ["windowstart", "f"],
    ["windowlength", "f"],
    ["threshold", "I"],
    ["intensity", "i"],
    ["receivergain", "I"],
    ["degc1", "I"],
    ["degc2", "I"],
    ["humidity", "I"],
    ["focus", "I"],
    ["battery", "I"],
    ["uservalue1", "f"],
    ["uservalue2", "f"],
    ["uservalue3", "f"],
    ["uservalue4", "f"],
    ["uservalue5", "f"],
    ["uservalue6", "f"],
    ["uservalue7", "f"],
    ["uservalue8", "f"],
    ["velocity", "f"],
    ["depth", "f"],
    ["altitude", "f"],
    ["pitch", "f"],
    ["pitchrate", "f"],
    ["roll", "f"],
    ["rollrate", "f"],
    ["heading", "f"],
    ["headingrate", "f"],
    ["compassheading", "f"],
    ["compasspitch", "f"],
    ["compassroll", "f"],
    ["latitude", "d"],
    ["longitude", "d"],
    ["sonarposition", "f"],
    ["configflags", "I"],
    ["beamtilt", "f"],
    ["targetrange", "f"],
    ["targetbearing", "f"],
    ["targetpresent", "I"],
    ["firmwarerevision", "I"],
    ["flags", "I"],
    ["sourceframe", "I"],
    ["watertemp", "f"],
    ["timerperiod", "I"],
    ["sonarx", "f"],
    ["sonary", "f"],
    ["sonarz", "f"],
    ["sonarpan", "f"],
    ["sonartilt", "f"],
    ["sonarroll", "f"],
    ["panpnnl", "f"],
    ["tiltpnnl", "f"],
    ["rollpnnl", "f"],
    ["vehicletime", "d"],
    ["timeggk", "f"],
    ["dateggk", "I"],
    ["qualityggk", "I"],
    ["numsatsggk", "I"],
    ["dopggk", "f"],
    ["ehtggk", "f"],
    ["heavetss", "f"],
    ["yeargps", "I"],
    ["monthgps", "I"],
    ["daygps", "I"],
    ["hourgps", "I"],
    ["minutegps", "I"],
    ["secondgps", "I"],
    ["hsecondgps", "I"],
    ["sonarpanoffset", "f"],
    ["sonartiltoffset", "f"],
    ["sonarrolloffset", "f"],
    ["sonarxoffset", "f"],
    ["sonaryoffset", "f"],
    ["sonarzoffset", "f"],
    ["tmatrix", "64s"],
    ["samplerate", "f"],
    ["accellx", "f"],
    ["accelly", "f"],
    ["accellz", "f"],
    ["pingmode", "I"],
    ["frequencyhilow", "I"],
    ["pulsewidth", "I"],
    ["cycleperiod", "I"],
    ["sampleperiod", "I"],
    ["transmitenable", "I"],
    ["framerate", "f"],
    ["soundspeed", "f"],
    ["samplesperbeam", "I"],
    ["enable150v", "I"],
    ["samplestartdelay", "I"],
    ["largelens", "I"],
    ["thesystemtype", "I"],
    ["sonarserialnumber", "I"],
    ["encryptedkey", "Q"],
    ["ariserrorflagsuint", "I"],
    ["missedpackets", "I"],
    ["arisappversion", "I"],
    ["available2", "I"],
    ["reorderedsamples", "I"],
    ["salinity", "I"],
    ["pressure", "f"],
    ["batteryvoltage", "f"],
    ["mainvoltage", "f"],
    ["switchvoltage", "f"],
    ["focusmotormoving", "I"],
    ["voltagechanging", "I"],
    ["focustimeoutfault", "I"],
    ["focusovercurrentfault", "I"],
    ["focusnotfoundfault", "I"],
    ["focusstalledfault", "I"],
    ["fpgatimeoutfault", "I"],
    ["fpgabusyfault", "I"],
    ["fpgastuckfault", "I"],
    ["cputempfault", "I"],
    ["psutempfault", "I"],
    ["watertempfault", "I"],
    ["humidityfault", "I"],
    ["pressurefault", "I"],
    ["voltagereadfault", "I"],
    ["voltagewritefault", "I"],
    ["focuscurrentposition", "I"],
    ["targetpan", "f"],
    ["targettilt", "f"],
    ["targetroll", "f"],
    ["panmotorerrorcode", "I"],
    ["tiltmotorerrorcode", "I"],
    ["rollmotorerrorcode", "I"],
    ["panabsposition", "f"],
    ["tiltabsposition", "f"],
    ["rollabsposition", "f"],
    ["panaccelx", "f"],
    ["panaccely", "f"],
    ["panaccelz", "f"],
    ["tiltaccelx", "f"],
    ["tiltaccely", "f"],
    ["tiltaccelz", "f"],
    ["rollaccelx", "f"],
    ["rollaccely", "f"],
    ["rollaccelz", "f"],
    ["appliedsettings", "I"],
    ["constrainedsettings", "I"],
    ["invalidsettings", "I"],
    ["enableinterpacketdelay", "I"],
    ["interpacketdelayperiod", "I"],
    ["uptime", "I"],
    ["arisappversionmajor", "H"],
    ["arisappversionminor", "H"],
    ["gotime", "Q"],
  ],
};

const FILE_HEADER_SIZES = { 3: 512, 4: 512, 5: 1024 };
const FRAME_HEADER_SIZES = { 3: 256, 4: 1024, 5: 1024 };

const TYPE_READERS = {
  i: {
    size: 4,
    read(view, offset) {
      return view.getInt32(offset, true);
    },
  },
  I: {
    size: 4,
    read(view, offset) {
      return view.getUint32(offset, true);
    },
  },
  f: {
    size: 4,
    read(view, offset) {
      return view.getFloat32(offset, true);
    },
  },
  Q: {
    size: 8,
    read(view, offset) {
      return Number(view.getBigUint64(offset, true));
    },
  },
  d: {
    size: 8,
    read(view, offset) {
      return view.getFloat64(offset, true);
    },
  },
  H: {
    size: 2,
    read(view, offset) {
      return view.getUint16(offset, true);
    },
  },
};

function parseStringType(type) {
  if (!type.endsWith("s")) {
    return null;
  }
  const size = Number.parseInt(type.slice(0, -1), 10);
  if (!Number.isInteger(size) || size <= 0) {
    throw new Error(`Unsupported struct string type: ${type}`);
  }
  return size;
}

function decodeAscii(bytes) {
  let end = bytes.length;
  while (end > 0 && bytes[end - 1] === 0) {
    end -= 1;
  }
  return new TextDecoder("ascii").decode(bytes.subarray(0, end));
}

function readStruct(view, schema) {
  const out = {};
  let offset = 0;

  for (const [name, type] of schema) {
    const stringSize = parseStringType(type);
    if (stringSize !== null) {
      out[name] = decodeAscii(
        new Uint8Array(view.buffer, view.byteOffset + offset, stringSize),
      );
      offset += stringSize;
      continue;
    }

    const reader = TYPE_READERS[type];
    if (!reader) {
      throw new Error(`Unsupported struct type: ${type}`);
    }
    out[name] = reader.read(view, offset);
    offset += reader.size;
  }

  return out;
}

function clipUint8(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= 255) {
    return 255;
  }
  return Math.trunc(value);
}

function toUint8Array(input) {
  if (input instanceof Uint8Array) {
    return input;
  }
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError("Expected ArrayBuffer or Uint8Array input");
}

function getFrameData(bytes, info, frameIndex) {
  const frameSizeWithHeader = info.frameheadersize + info.framesize;
  const frameStart = info.fileheadersize + frameIndex * frameSizeWithHeader;
  const payloadStart = frameStart + info.frameheadersize;
  const payloadEnd = payloadStart + info.framesize;
  if (payloadEnd > bytes.byteLength) {
    return null;
  }
  return bytes.subarray(payloadStart, payloadEnd);
}

function framePayloadRange(info, frameIndex) {
  const frameSizeWithHeader = info.frameheadersize + info.framesize;
  const frameStart = info.fileheadersize + frameIndex * frameSizeWithHeader;
  return {
    payloadStart: frameStart + info.frameheadersize,
    payloadEnd: frameStart + info.frameheadersize + info.framesize,
  };
}

function allZero(bytes) {
  for (let index = 0; index < bytes.length; index += 1) {
    if (bytes[index] !== 0) {
      return false;
    }
  }
  return true;
}

async function maybeYield(frameIndex, totalFrames, onProgress) {
  if (onProgress) {
    onProgress(frameIndex, totalFrames);
  }
  if (frameIndex % 32 === 0) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

export function parseSonarHeader(input) {
  const bytes = toUint8Array(input);
  if (bytes.byteLength < 4) {
    throw new Error("Input is too small to be a DIDSON/ARIS file");
  }

  const magic = decodeAscii(bytes.subarray(0, 3));
  if (magic !== "DDF") {
    throw new Error(`Expected DIDSON/ARIS header to start with DDF, got ${magic}`);
  }

  const versionId = bytes[3];
  const schema = FILE_SCHEMAS[versionId];
  const frameSchema = FRAME_SCHEMAS[versionId];
  if (!schema) {
    throw new Error(`Unsupported DIDSON/ARIS version: ${versionId}`);
  }

  const fileheadersize = FILE_HEADER_SIZES[versionId];
  const frameheadersize = FRAME_HEADER_SIZES[versionId];
  const view = new DataView(bytes.buffer, bytes.byteOffset, fileheadersize);
  const rawHeader = readStruct(view, schema);
  if (
    frameSchema &&
    bytes.byteLength >= fileheadersize + frameheadersize
  ) {
    const frameView = new DataView(
      bytes.buffer,
      bytes.byteOffset + fileheadersize,
      frameheadersize,
    );
    Object.assign(rawHeader, readStruct(frameView, frameSchema));
  }
  const framesize = rawHeader.samplesperchannel * rawHeader.numbeams;
  const numframes = Math.floor(
    (bytes.byteLength - fileheadersize) / (frameheadersize + framesize),
  );

  return {
    ...rawHeader,
    version_id: versionId,
    fileheadersize,
    frameheadersize,
    framesize,
    numframes,
    filename: null,
  };
}

export async function decodeSonarBuffer(
  input,
  {
    startFrame = 0,
    endFrame = -1,
    bgs = true,
    rawThirdChannel = true,
    returnAsBgr = true,
    numFramesBgSubtract = 100000,
    onProgress = null,
  } = {},
) {
  const bytes = toUint8Array(input);
  const info = parseSonarHeader(bytes);
  const planeSize = info.samplesperchannel * info.numbeams;

  let resolvedStartFrame =
    startFrame === -1 ? (info.startframe || 0) : Number(startFrame);
  let resolvedEndFrame =
    endFrame === -1 ? (info.numframes || info.endframe) : Number(endFrame);
  resolvedEndFrame = Math.min(resolvedEndFrame, info.numframes);

  if (resolvedStartFrame < 0 || resolvedEndFrame < 0) {
    throw new Error("Frame range must be non-negative after resolution");
  }
  if (resolvedEndFrame <= resolvedStartFrame) {
    throw new Error(
      `Invalid frame range [${resolvedStartFrame}, ${resolvedEndFrame})`,
    );
  }

  const firstFrame = getFrameData(bytes, info, resolvedStartFrame);
  if (firstFrame && allZero(firstFrame)) {
    resolvedEndFrame = resolvedEndFrame - resolvedStartFrame + 1;
    resolvedStartFrame = 0;
  }

  resolvedEndFrame = Math.min(resolvedEndFrame, info.numframes);
  const loadedFrameCount = resolvedEndFrame - resolvedStartFrame;
  const outputFrameCount = Math.max(loadedFrameCount - 1, 0);
  const width = outputFrameCount;
  const height = info.samplesperchannel;

  let meanFrame = null;
  let meanNormalizationValue = null;
  if (bgs) {
    const bgFrameCount = Math.min(loadedFrameCount, numFramesBgSubtract);
    const sumFrame = new Float64Array(planeSize);
    const maxFrame = new Float64Array(planeSize);

    for (let frameOffset = 0; frameOffset < bgFrameCount; frameOffset += 1) {
      const frame = getFrameData(bytes, info, resolvedStartFrame + frameOffset);
      if (!frame || frame.length !== planeSize) {
        break;
      }
      for (
        let revIndex = 0, srcIndex = planeSize - 1;
        revIndex < planeSize;
        revIndex += 1, srcIndex -= 1
      ) {
        const value = frame[srcIndex];
        sumFrame[revIndex] += value;
        if (value > maxFrame[revIndex]) {
          maxFrame[revIndex] = value;
        }
      }
      await maybeYield(frameOffset + 1, bgFrameCount, null);
    }

    meanFrame = new Float64Array(planeSize);
    let normalization = 0;
    const safeDivisor = Math.max(1, Math.min(bgFrameCount, loadedFrameCount));
    for (let index = 0; index < planeSize; index += 1) {
      const meanValue = sumFrame[index] / safeDivisor;
      meanFrame[index] = meanValue;
      const centeredMax = maxFrame[index] - meanValue;
      if (centeredMax > normalization) {
        normalization = centeredMax;
      }
    }
    meanNormalizationValue = normalization;
  }

  const bgrImage = new Uint8Array(height * width * 3);
  const greenScale = 255 / info.numbeams;

  for (let column = 0; column < outputFrameCount; column += 1) {
    const frame = getFrameData(bytes, info, resolvedStartFrame + column);
    if (!frame || frame.length !== planeSize) {
      break;
    }

    for (let row = 0; row < height; row += 1) {
      let maxProc = Number.NEGATIVE_INFINITY;
      let argmaxBeam = 0;
      let rawMax = 0;
      const reversedRowOffset = row * info.numbeams;

      for (let beam = 0; beam < info.numbeams; beam += 1) {
        const reversedIndex = reversedRowOffset + beam;
        const srcIndex = planeSize - 1 - reversedIndex;
        const rawValue = frame[srcIndex];
        if (rawValue > rawMax) {
          rawMax = rawValue;
        }

        let procValue = rawValue;
        if (bgs) {
          procValue =
            (rawValue - meanFrame[reversedIndex]) / meanNormalizationValue;
        }
        if (procValue > maxProc) {
          maxProc = procValue;
          argmaxBeam = beam;
        }
      }

      if (!Number.isFinite(maxProc)) {
        maxProc = 0;
      }

      const pixelBase = (row * width + column) * 3;
      bgrImage[pixelBase + 1] = clipUint8(argmaxBeam * greenScale);

      const magnitudeByte = clipUint8(maxProc * 255);
      const rawThirdByte = rawThirdChannel ? clipUint8(rawMax) : 0;
      if (returnAsBgr) {
        bgrImage[pixelBase] = rawThirdByte;
        bgrImage[pixelBase + 2] = magnitudeByte;
      } else {
        bgrImage[pixelBase] = magnitudeByte;
        bgrImage[pixelBase + 2] = rawThirdByte;
      }
    }

    await maybeYield(column + 1, outputFrameCount, onProgress);
  }

  if (width > 0 && height > 0) {
    const borderGreen = clipUint8(0.5 * 255);
    for (let x = 0; x < width; x += 1) {
      const top = x * 3;
      const bottom = ((height - 1) * width + x) * 3;
      bgrImage[top] = 0;
      bgrImage[top + 1] = borderGreen;
      bgrImage[top + 2] = 0;
      bgrImage[bottom] = 0;
      bgrImage[bottom + 1] = borderGreen;
      bgrImage[bottom + 2] = 0;
    }
    for (let y = 0; y < height; y += 1) {
      const left = (y * width) * 3;
      const right = (y * width + (width - 1)) * 3;
      bgrImage[left] = 0;
      bgrImage[left + 1] = borderGreen;
      bgrImage[left + 2] = 0;
      bgrImage[right] = 0;
      bgrImage[right + 1] = borderGreen;
      bgrImage[right + 2] = 0;
    }
  }

  return {
    imageBgr: bgrImage,
    width,
    height,
    channels: 3,
    frameRange: {
      start: resolvedStartFrame,
      end: resolvedEndFrame,
    },
    metadata: info,
    bgs,
    rawThirdChannel,
    returnAsBgr,
    meanNormalizationValue,
  };
}

function decodeFramePayloadRgb(frame, info, frameIndex) {
  if (!frame || frame.length !== info.samplesperchannel * info.numbeams) {
    return null;
  }

  const width = info.numbeams;
  const height = info.samplesperchannel;
  const planeSize = width * height;
  let maxValue = 0;
  for (let index = 0; index < planeSize; index += 1) {
    if (frame[index] > maxValue) {
      maxValue = frame[index];
    }
  }

  const rgbImage = new Uint8ClampedArray(width * height * 3);
  const safeMax = Math.max(1, maxValue);
  for (let row = 0; row < height; row += 1) {
    const rowOffset = row * width;
    for (let beam = 0; beam < width; beam += 1) {
      const srcIndex = planeSize - 1 - (rowOffset + beam);
      const normalized = frame[srcIndex] / safeMax;
      const intensity = clipUint8(Math.pow(normalized, 0.7) * 255);
      const pixelBase = (rowOffset + beam) * 3;
      rgbImage[pixelBase] = intensity;
      rgbImage[pixelBase + 1] = intensity;
      rgbImage[pixelBase + 2] = intensity;
    }
  }

  return {
    rgbImage,
    width,
    height,
    frameIndex,
  };
}

export function decodeSonarFrameRgb(input, frameIndex) {
  const bytes = toUint8Array(input);
  const info = parseSonarHeader(bytes);
  const resolvedFrameIndex = Math.max(0, Math.min(info.numframes - 1, Math.round(Number(frameIndex))));
  const frame = getFrameData(bytes, info, resolvedFrameIndex);
  return decodeFramePayloadRgb(frame, info, resolvedFrameIndex);
}

export async function decodeSonarFrameRgbFromFile(file, info, frameIndex) {
  if (!file || !info) {
    return null;
  }
  const resolvedFrameIndex = Math.max(0, Math.min(info.numframes - 1, Math.round(Number(frameIndex))));
  const { payloadStart, payloadEnd } = framePayloadRange(info, resolvedFrameIndex);
  const frameBytes = new Uint8Array(await file.slice(payloadStart, payloadEnd).arrayBuffer());
  return decodeFramePayloadRgb(frameBytes, info, resolvedFrameIndex);
}

export function bgrToRgbImage(bgrImage) {
  const rgbImage = new Uint8ClampedArray(bgrImage.length);
  for (let index = 0; index < bgrImage.length; index += 3) {
    rgbImage[index] = bgrImage[index + 2];
    rgbImage[index + 1] = bgrImage[index + 1];
    rgbImage[index + 2] = bgrImage[index];
  }
  return rgbImage;
}
