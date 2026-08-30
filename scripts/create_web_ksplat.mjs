import * as GaussianSplats3D from '../frontend/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js';
import * as THREE from '../frontend/node_modules/three/build/three.module.js';
import * as fs from 'node:fs';

const inputFile = process.argv[2];
const outputFile = process.argv[3];
if (!inputFile || !outputFile) {
  throw new Error('Usage: node scripts/create_web_ksplat.mjs input.splat output.ksplat');
}

const source = fs.readFileSync(inputFile);
const fileData = source.buffer.slice(
  source.byteOffset,
  source.byteOffset + source.byteLength,
);
const splatArray = GaussianSplats3D.SplatParser.parseStandardSplatToUncompressedSplatArray(
  fileData,
);
const generator = GaussianSplats3D.SplatBufferGenerator.getStandardGenerator(
  1,
  1,
  0,
  new THREE.Vector3(0, 0, 0),
  5.0,
  256,
);
const splatBuffer = generator.generateFromUncompressedSplatArray(splatArray);
fs.writeFileSync(outputFile, Buffer.from(splatBuffer.bufferData));
console.log(
  JSON.stringify({
    inputFile,
    outputFile,
    splats: splatBuffer.getSplatCount(),
    bytes: splatBuffer.bufferData.byteLength,
  }),
);
