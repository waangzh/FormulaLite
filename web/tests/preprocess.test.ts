import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { resolve } from "node:path";

import { PNG } from "pngjs";

import { preprocessConfig, preprocessPixels } from "../runtime/preprocess.ts";

type FixtureRecord = {
  name: string;
  preprocess_spec_version: string;
  source: { path: string };
  expected_image_uint8: { path: string };
  expected_tensor: { path: string };
  metadata: unknown;
};

const fixtureRoot = resolve(process.cwd(), "../tests/fixtures/preprocess");
const fixtureSet = JSON.parse(readFileSync(resolve(fixtureRoot, "manifest.json"), "utf8")) as {
  preprocess_spec_version: string;
  fixtures: Array<{ manifest: string }>;
};

let globalMaximumTensorError = 0;

test("runtime consumes the shared preprocess config", () => {
  assert.equal(preprocessConfig.spec_version, fixtureSet.preprocess_spec_version);
});

for (const fixtureEntry of fixtureSet.fixtures) {
  const fixture = JSON.parse(
    readFileSync(resolve(fixtureRoot, fixtureEntry.manifest), "utf8"),
  ) as FixtureRecord;

  test(`${fixture.name}: metadata, image, and tensor parity`, () => {
    assert.equal(fixture.preprocess_spec_version, preprocessConfig.spec_version);
    const decoded = PNG.sync.read(readFileSync(resolve(fixtureRoot, fixture.source.path)));
    const rgba = new Uint8Array(
      decoded.data.buffer,
      decoded.data.byteOffset,
      decoded.data.byteLength,
    );
    const actual = preprocessPixels({ width: decoded.width, height: decoded.height, data: rgba });
    assert.deepEqual(actual.metadata, fixture.metadata);

    const expectedImage = readFileSync(resolve(fixtureRoot, fixture.expected_image_uint8.path));
    let maximumImageError = 0;
    for (let index = 0; index < actual.imageUint8.length; index += 1) {
      maximumImageError = Math.max(
        maximumImageError,
        Math.abs(actual.imageUint8[index] - expectedImage[index]),
      );
    }
    assert.ok(maximumImageError <= 1, `maximum uint8 error ${maximumImageError} exceeds 1`);

    const expectedTensor = readFileSync(resolve(fixtureRoot, fixture.expected_tensor.path));
    const expectedView = new DataView(
      expectedTensor.buffer,
      expectedTensor.byteOffset,
      expectedTensor.byteLength,
    );
    let maximumTensorError = 0;
    let maximumIndex = 0;
    for (let index = 0; index < actual.pixelValues.length; index += 1) {
      const expected = expectedView.getFloat32(index * Float32Array.BYTES_PER_ELEMENT, true);
      const difference = Math.abs(actual.pixelValues[index] - expected);
      if (difference > maximumTensorError) {
        maximumTensorError = difference;
        maximumIndex = index;
      }
      const tolerance = 1e-5 + 1e-5 * Math.abs(expected);
      assert.ok(
        difference <= tolerance,
        `${fixture.name} tensor mismatch at ${index}: actual=${actual.pixelValues[index]} ` +
          `expected=${expected} difference=${difference}`,
      );
    }
    globalMaximumTensorError = Math.max(globalMaximumTensorError, maximumTensorError);
    console.log(
      `${fixture.name}: max_uint8_error=${maximumImageError} ` +
        `max_tensor_error=${maximumTensorError} at_index=${maximumIndex}`,
    );
  });
}

test("parity summary", () => {
  console.log(`global_max_tensor_error=${globalMaximumTensorError}`);
  assert.ok(globalMaximumTensorError <= 1e-5);
});
