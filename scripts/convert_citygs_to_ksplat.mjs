import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modulePath = path.join(
  repoRoot,
  'tmp',
  'gaussian_viewer',
  'node_modules',
  '@mkkellogg',
  'gaussian-splats-3d',
  'build',
  'gaussian-splats-3d.module.js',
);

const [, , inputArg, outputArg, compressionArg = '2', alphaArg = '8', shArg = '2'] = process.argv;

if (!inputArg || !outputArg) {
  console.error(
    'Usage: node scripts/convert_citygs_to_ksplat.mjs <input.ply> <output.ksplat> [compression=2] [minimumAlpha=8] [shDegree=2]',
  );
  process.exit(2);
}

const inputPath = path.resolve(inputArg);
const outputPath = path.resolve(outputArg);
const compressionLevel = Number(compressionArg);
const minimumAlpha = Number(alphaArg);
const sphericalHarmonicsDegree = Number(shArg);

globalThis.window = globalThis;
const GaussianSplats3D = await import(pathToFileURL(modulePath).href);

console.log(`[CityGS] Reading ${inputPath}`);
const inputBuffer = await fs.readFile(inputPath);
const inputArrayBuffer = inputBuffer.buffer.slice(
  inputBuffer.byteOffset,
  inputBuffer.byteOffset + inputBuffer.byteLength,
);
console.log(`[CityGS] Parsing ${(inputBuffer.byteLength / 1024 / 1024).toFixed(1)} MiB PLY`);

const splatBuffer = await GaussianSplats3D.PlyLoader.loadFromFileData(
  inputArrayBuffer,
  minimumAlpha,
  compressionLevel,
  true,
  sphericalHarmonicsDegree,
);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, Buffer.from(splatBuffer.bufferData));

const outputStat = await fs.stat(outputPath);
console.log(
  `[CityGS] Wrote ${outputPath} (${(outputStat.size / 1024 / 1024).toFixed(1)} MiB, ${splatBuffer.getSplatCount().toLocaleString()} splats)`,
);
