import rawConfig from "../../artifacts/preprocessor/preprocess_config.json" with { type: "json" };

type PreprocessConfig = {
  spec_version: string;
  uint8_max: number;
  image_size: { width: number; height: number };
  foreground_threshold: number;
  grayscale: { coefficients: [number, number, number]; divisor: number };
  mean: [number, number, number];
  std: [number, number, number];
  output_dtype: "float32";
  output_layout: "CHW";
  channels: number;
  alpha: { background_rgb: [number, number, number]; divisor: number };
  polarity: { invert_when_mean_below: number };
  resize: { interpolation: "bilinear" };
  padding: { value: number };
};

export const preprocessConfig = rawConfig as unknown as PreprocessConfig;

export type PixelImage = {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
};

export type BBox = { x0: number; y0: number; x1: number; y1: number };
export type Padding = { left: number; top: number; right: number; bottom: number };
export type PreprocessMetadata = {
  bbox: BBox;
  input_size: [number, number];
  crop_size: [number, number];
  resize_size: [number, number];
  padding: Padding;
  inverted: boolean;
  empty: boolean;
};

export type PreprocessResult = {
  pixelValues: Float32Array;
  imageUint8: Uint8Array;
  metadata: PreprocessMetadata;
};

function roundHalfUpDivision(numerator: number, denominator: number): number {
  return Math.floor((numerator + Math.floor(denominator / 2)) / denominator);
}

function alphaComposite(image: PixelImage): Uint8Array {
  const output = new Uint8Array(image.width * image.height * preprocessConfig.channels);
  const background = preprocessConfig.alpha.background_rgb;
  const divisor = preprocessConfig.alpha.divisor;
  for (let pixel = 0; pixel < image.width * image.height; pixel += 1) {
    const sourceOffset = pixel * 4;
    const targetOffset = pixel * preprocessConfig.channels;
    const alpha = image.data[sourceOffset + 3];
    for (let channel = 0; channel < preprocessConfig.channels; channel += 1) {
      const numerator =
        image.data[sourceOffset + channel] * alpha + background[channel] * (divisor - alpha);
      output[targetOffset + channel] = roundHalfUpDivision(numerator, divisor);
    }
  }
  return output;
}

function grayscale(rgb: Uint8Array, width: number, height: number): Uint8Array {
  const output = new Uint8Array(width * height);
  const coefficients = preprocessConfig.grayscale.coefficients;
  const divisor = preprocessConfig.grayscale.divisor;
  for (let pixel = 0; pixel < output.length; pixel += 1) {
    const offset = pixel * preprocessConfig.channels;
    const numerator =
      rgb[offset] * coefficients[0] +
      rgb[offset + 1] * coefficients[1] +
      rgb[offset + 2] * coefficients[2];
    output[pixel] = roundHalfUpDivision(numerator, divisor);
  }
  return output;
}

function normalizePolarity(
  gray: Uint8Array,
  width: number,
  height: number,
): { image: Uint8Array; inverted: boolean } {
  let total = 0;
  let count = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (y === 0 || y === height - 1 || x === 0 || x === width - 1) {
        total += gray[y * width + x];
        count += 1;
      }
    }
  }
  const inverted = total / count < preprocessConfig.polarity.invert_when_mean_below;
  if (!inverted) return { image: gray, inverted };
  const output = new Uint8Array(gray.length);
  for (let index = 0; index < gray.length; index += 1) {
    output[index] = preprocessConfig.uint8_max - gray[index];
  }
  return { image: output, inverted };
}

function normalizeDynamicRange(gray: Uint8Array): Uint8Array {
  let minimum = preprocessConfig.uint8_max;
  let maximum = 0;
  for (const value of gray) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  if (minimum === maximum) return gray.slice();
  const range = maximum - minimum;
  const output = new Uint8Array(gray.length);
  for (let index = 0; index < gray.length; index += 1) {
    output[index] = roundHalfUpDivision(
      (gray[index] - minimum) * preprocessConfig.uint8_max,
      range,
    );
  }
  return output;
}

function foregroundBBox(gray: Uint8Array, width: number, height: number): { bbox: BBox; empty: boolean } {
  let x0 = width;
  let y0 = height;
  let x1 = 0;
  let y1 = 0;
  let empty = true;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (gray[y * width + x] < preprocessConfig.foreground_threshold) {
        empty = false;
        x0 = Math.min(x0, x);
        y0 = Math.min(y0, y);
        x1 = Math.max(x1, x + 1);
        y1 = Math.max(y1, y + 1);
      }
    }
  }
  return empty
    ? { bbox: { x0: 0, y0: 0, x1: width, y1: height }, empty }
    : { bbox: { x0, y0, x1, y1 }, empty };
}

function crop(gray: Uint8Array, width: number, bbox: BBox): Uint8Array {
  const cropWidth = bbox.x1 - bbox.x0;
  const cropHeight = bbox.y1 - bbox.y0;
  const output = new Uint8Array(cropWidth * cropHeight);
  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      output[y * cropWidth + x] = gray[(bbox.y0 + y) * width + bbox.x0 + x];
    }
  }
  return output;
}

function resizeDimensions(width: number, height: number): [number, number] {
  const target = preprocessConfig.image_size;
  const scale = Math.min(target.width / width, target.height / height);
  const resizedWidth = Math.min(target.width, Math.max(1, Math.floor(width * scale + 0.5)));
  const resizedHeight = Math.min(target.height, Math.max(1, Math.floor(height * scale + 0.5)));
  return [resizedWidth, resizedHeight];
}

function resizeBilinear(
  source: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): Uint8Array {
  if (sourceWidth === width && sourceHeight === height) return source.slice();
  const output = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const sourceY = ((y + 0.5) * sourceHeight) / height - 0.5;
    const sourceYFloor = Math.floor(sourceY);
    const yWeight = sourceY - sourceYFloor;
    const y0 = Math.max(0, Math.min(sourceHeight - 1, sourceYFloor));
    const y1 = Math.max(0, Math.min(sourceHeight - 1, sourceYFloor + 1));
    for (let x = 0; x < width; x += 1) {
      const sourceX = ((x + 0.5) * sourceWidth) / width - 0.5;
      const sourceXFloor = Math.floor(sourceX);
      const xWeight = sourceX - sourceXFloor;
      const x0 = Math.max(0, Math.min(sourceWidth - 1, sourceXFloor));
      const x1 = Math.max(0, Math.min(sourceWidth - 1, sourceXFloor + 1));
      const top =
        source[y0 * sourceWidth + x0] * (1 - xWeight) +
        source[y0 * sourceWidth + x1] * xWeight;
      const bottom =
        source[y1 * sourceWidth + x0] * (1 - xWeight) +
        source[y1 * sourceWidth + x1] * xWeight;
      output[y * width + x] = Math.floor(top * (1 - yWeight) + bottom * yWeight + 0.5);
    }
  }
  return output;
}

function centerPad(
  image: Uint8Array,
  width: number,
  height: number,
): { image: Uint8Array; padding: Padding } {
  const target = preprocessConfig.image_size;
  const deltaWidth = target.width - width;
  const deltaHeight = target.height - height;
  const left = Math.floor(deltaWidth / 2);
  const top = Math.floor(deltaHeight / 2);
  const padding = {
    left,
    top,
    right: deltaWidth - left,
    bottom: deltaHeight - top,
  };
  const output = new Uint8Array(target.width * target.height);
  output.fill(preprocessConfig.padding.value);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      output[(top + y) * target.width + left + x] = image[y * width + x];
    }
  }
  return { image: output, padding };
}

function normalizeTensor(image: Uint8Array): Float32Array {
  const plane = image.length;
  const output = new Float32Array(preprocessConfig.channels * plane);
  for (let channel = 0; channel < preprocessConfig.channels; channel += 1) {
    for (let pixel = 0; pixel < plane; pixel += 1) {
      output[channel * plane + pixel] =
        (image[pixel] / preprocessConfig.uint8_max - preprocessConfig.mean[channel]) /
        preprocessConfig.std[channel];
    }
  }
  return output;
}

export function preprocessPixels(image: PixelImage): PreprocessResult {
  if (image.width <= 0 || image.height <= 0 || image.data.length !== image.width * image.height * 4) {
    throw new Error("decoded input must be a non-empty RGBA raster");
  }
  const rgb = alphaComposite(image);
  const gray = grayscale(rgb, image.width, image.height);
  const polarity = normalizePolarity(gray, image.width, image.height);
  const ranged = normalizeDynamicRange(polarity.image);
  const foreground = foregroundBBox(ranged, image.width, image.height);
  const cropWidth = foreground.bbox.x1 - foreground.bbox.x0;
  const cropHeight = foreground.bbox.y1 - foreground.bbox.y0;
  const cropped = crop(ranged, image.width, foreground.bbox);
  const [resizeWidth, resizeHeight] = resizeDimensions(cropWidth, cropHeight);
  const resized = resizeBilinear(cropped, cropWidth, cropHeight, resizeWidth, resizeHeight);
  const padded = centerPad(resized, resizeWidth, resizeHeight);
  return {
    pixelValues: normalizeTensor(padded.image),
    imageUint8: padded.image,
    metadata: {
      bbox: foreground.bbox,
      input_size: [image.width, image.height],
      crop_size: [cropWidth, cropHeight],
      resize_size: [resizeWidth, resizeHeight],
      padding: padded.padding,
      inverted: polarity.inverted,
      empty: foreground.empty,
    },
  };
}

export async function decodeImage(image: Blob): Promise<PixelImage> {
  const bitmap = await createImageBitmap(image, { imageOrientation: "none", premultiplyAlpha: "none" });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (context === null) throw new Error("2D canvas is unavailable");
  context.globalCompositeOperation = "copy";
  context.drawImage(bitmap, 0, 0);
  const decoded = context.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return { width: decoded.width, height: decoded.height, data: decoded.data };
}

export async function preprocess(image: Blob): Promise<PreprocessResult> {
  return preprocessPixels(await decodeImage(image));
}
