const CLASS_COLOUR_ANCHORS = {
  0: [
    [178, 235, 242],
    [0, 188, 212],
    [3, 169, 244],
    [30, 136, 229],
    [13, 71, 161],
  ],
  1: [
    [255, 224, 130],
    [255, 193, 7],
    [255, 167, 38],
    [245, 124, 0],
    [230, 81, 0],
  ],
  2: [
    [248, 187, 208],
    [236, 64, 122],
    [171, 71, 188],
    [126, 87, 194],
    [74, 20, 140],
  ],
};

export function clampClassId(classId) {
  return classId === 0 || classId === 1 ? classId : 2;
}

function interpolateColour(anchors, position) {
  if (anchors.length === 1) {
    return anchors[0].slice();
  }

  const clamped = Math.max(0, Math.min(1, position));
  const scaled = clamped * (anchors.length - 1);
  const leftIndex = Math.floor(scaled);
  const rightIndex = Math.min(leftIndex + 1, anchors.length - 1);
  const mix = scaled - leftIndex;
  const left = anchors[leftIndex];
  const right = anchors[rightIndex];

  return [
    left[0] * (1 - mix) + right[0] * mix,
    left[1] * (1 - mix) + right[1] * mix,
    left[2] * (1 - mix) + right[2] * mix,
  ];
}

export function buildInstanceColours(classId, count) {
  const anchors = CLASS_COLOUR_ANCHORS[clampClassId(classId)] ?? CLASS_COLOUR_ANCHORS[2];
  if (count <= 0) {
    return [];
  }
  if (count === 1) {
    return [interpolateColour(anchors, 0.5)];
  }

  const colours = [];
  for (let index = 0; index < count; index += 1) {
    const position = count === 1 ? 0.5 : index / (count - 1);
    colours.push(interpolateColour(anchors, position));
  }
  return colours;
}
