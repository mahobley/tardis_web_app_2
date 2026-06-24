export function rgbToImageData(rgbImage, width, height) {
  const expectedLength = width * height * 3;
  if (rgbImage.length !== expectedLength) {
    throw new Error(
      `Expected RGB buffer length ${expectedLength}, got ${rgbImage.length}`,
    );
  }

  const rgba = new Uint8ClampedArray(width * height * 4);
  for (
    let src = 0, dst = 0;
    src < rgbImage.length;
    src += 3, dst += 4
  ) {
    rgba[dst] = rgbImage[src];
    rgba[dst + 1] = rgbImage[src + 1];
    rgba[dst + 2] = rgbImage[src + 2];
    rgba[dst + 3] = 255;
  }

  return new ImageData(rgba, width, height);
}
