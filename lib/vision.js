'use strict';
const path = require('path');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs');
const wasm = require('@tensorflow/tfjs-backend-wasm');
const faceapi = require('@vladmandic/face-api/dist/face-api.node-wasm.js');

const MODEL_DIR = path.join(
  path.dirname(require.resolve('@vladmandic/face-api/package.json')),
  'model'
);
const WASM_DIR = path.join(
  path.dirname(require.resolve('@tensorflow/tfjs-backend-wasm/package.json')),
  'dist/'
);

// 68점 랜드마크 기준 눈 6점의 Eye Aspect Ratio(EAR). 값이 낮을수록 눈을 감은 상태.
function eyeAspectRatio(pts) {
  const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const vertical = d(pts[1], pts[5]) + d(pts[2], pts[4]);
  const horizontal = d(pts[0], pts[3]);
  if (horizontal === 0) return 0;
  return vertical / (2 * horizontal);
}

let ready = false;
async function initModels() {
  if (ready) return;
  wasm.setWasmPaths(WASM_DIR);
  await tf.setBackend('wasm');
  await tf.ready();
  await faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceExpressionNet.loadFromDisk(MODEL_DIR);
  ready = true;
}

async function imageToTensor(filePath, maxDim) {
  const { data, info } = await sharp(filePath)
    .rotate() // EXIF orientation 반영
    .resize({ width: maxDim, height: maxDim, fit: 'inside', withoutEnlargement: true })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const tensor = tf.tensor3d(new Uint8Array(data), [info.height, info.width, 3], 'int32');
  return { tensor, width: info.width, height: info.height };
}

// 라플라시안 분산: 값이 클수록 선명함(초점이 잘 맞음), 작을수록 흐릿함(블러/손떨림)
async function sharpnessScore(filePath) {
  const { data, info } = await sharp(filePath)
    .rotate()
    .resize({ width: 1024, height: 1024, fit: 'inside', withoutEnlargement: true })
    .greyscale()
    .convolve({ width: 3, height: 3, kernel: [0, 1, 0, 1, -4, 1, 0, 1, 0] })
    .raw()
    .toBuffer({ resolveWithObject: true });
  const n = data.length;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  return { variance, pixels: info.width * info.height };
}

const EAR_CLOSED = 0.16; // 이 값 이하면 눈 감음으로 간주
const EAR_OPEN = 0.28; // 이 값 이상이면 완전히 뜬 것으로 간주

async function analyzeImage(filePath, { detectorInputSize = 416 } = {}) {
  await initModels();
  const [{ tensor, width, height }, sharpness] = await Promise.all([
    imageToTensor(filePath, 1600),
    sharpnessScore(filePath),
  ]);

  const options = new faceapi.TinyFaceDetectorOptions({
    inputSize: detectorInputSize,
    scoreThreshold: 0.3,
  });
  const results = await faceapi
    .detectAllFaces(tensor, options)
    .withFaceLandmarks()
    .withFaceExpressions();
  tf.dispose(tensor);

  const imgArea = width * height;
  const faces = results.map((r) => {
    const box = r.detection.box;
    const areaFrac = (box.width * box.height) / imgArea;
    const leftEar = eyeAspectRatio(r.landmarks.getLeftEye());
    const rightEar = eyeAspectRatio(r.landmarks.getRightEye());
    const ear = (leftEar + rightEar) / 2;
    const openness = Math.min(1, Math.max(0, (ear - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED)));
    const eyesClosed = ear < EAR_CLOSED;
    const expr = r.expressions;
    const exprScoreRaw =
      expr.happy + 0.5 * expr.surprised + 0.3 * expr.neutral -
      expr.sad - expr.angry - expr.disgusted - 0.5 * expr.fearful;
    return {
      areaFrac,
      ear,
      openness,
      eyesClosed,
      expressions: expr,
      exprScoreRaw,
      dominantExpression: Object.entries(expr).sort((a, b) => b[1] - a[1])[0][0],
    };
  });

  return {
    width,
    height,
    sharpness: sharpness.variance,
    faces,
  };
}

module.exports = { analyzeImage, eyeAspectRatio, EAR_CLOSED, EAR_OPEN };
