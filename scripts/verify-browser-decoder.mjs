import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";

import { decodeSonarBuffer } from "../web/src/sonar/decoder.js";

function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

function compareBytes(expected, actual) {
  const maxDiffs = 10;
  const diffs = [];
  let mismatchCount = 0;
  const total = Math.max(expected.length, actual.length);

  for (let index = 0; index < total; index += 1) {
    const left = expected[index];
    const right = actual[index];
    if (left !== right) {
      mismatchCount += 1;
      if (diffs.length < maxDiffs) {
        diffs.push({ index, expected: left, actual: right });
      }
    }
  }

  return { mismatchCount, diffs };
}

const repoRoot = resolve(new URL("..", import.meta.url).pathname);
const inputPath = process.argv[2]
  ? resolve(process.argv[2])
  : resolve(repoRoot, "example.aris");
const startFrame = process.argv[3] ? Number(process.argv[3]) : 0;
const endFrame = process.argv[4] ? Number(process.argv[4]) : -1;

const tempDir = mkdtempSync(join(tmpdir(), "echo-seg-browser-"));
const outputBin = join(tempDir, "python_echogram.bin");
const outputJson = join(tempDir, "python_echogram.json");

try {
  execFileSync(
    "python",
    [
      resolve(repoRoot, "scripts/export_python_echogram.py"),
      "--input",
      inputPath,
      "--output-bin",
      outputBin,
      "--output-json",
      outputJson,
      "--start-frame",
      String(startFrame),
      "--end-frame",
      String(endFrame),
    ],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        MPLCONFIGDIR: "/tmp/mplconfig",
      },
      stdio: "inherit",
    },
  );

  const pythonBytes = readFileSync(outputBin);
  const pythonMeta = JSON.parse(readFileSync(outputJson, "utf8"));
  const sourceBytes = readFileSync(inputPath);
  const jsDecoded = await decodeSonarBuffer(sourceBytes.buffer.slice(sourceBytes.byteOffset, sourceBytes.byteOffset + sourceBytes.byteLength), {
    startFrame,
    endFrame,
    bgs: true,
    rawThirdChannel: true,
    returnAsBgr: true,
  });

  const jsBytes = Buffer.from(jsDecoded.imageBgr);
  const pythonShape = pythonMeta.shape;
  const jsShape = [jsDecoded.height, jsDecoded.width, jsDecoded.channels];

  console.log("python shape", pythonShape.join(" x "));
  console.log("browser shape", jsShape.join(" x "));
  console.log("python sha256", sha256(pythonBytes));
  console.log("browser sha256", sha256(jsBytes));

  if (
    pythonShape.length !== jsShape.length ||
    pythonShape.some((value, index) => value !== jsShape[index])
  ) {
    console.error("Shape mismatch between Python and browser decoder");
    process.exitCode = 1;
  } else {
    const { mismatchCount, diffs } = compareBytes(pythonBytes, jsBytes);
    if (mismatchCount > 0) {
      console.error(`Byte mismatch count: ${mismatchCount}`);
      for (const diff of diffs) {
        console.error(
          `  index=${diff.index} python=${diff.expected} browser=${diff.actual}`,
        );
      }
      process.exitCode = 1;
    } else {
      console.log("Decoder parity OK: browser bytes match Python exactly.");
    }
  }
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}
