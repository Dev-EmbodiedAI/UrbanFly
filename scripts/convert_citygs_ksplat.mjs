import fs from 'fs';
import path from 'path';

function parseArgs(argv) {
  const args = {
    minimumAlpha: 1,
    compressionLevel: 1,
    optimizeSplatData: true,
    sphericalHarmonicsDegree: 2,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--input') {
      args.input = argv[++i];
    } else if (arg === '--output') {
      args.output = argv[++i];
    } else if (arg === '--minimum-alpha') {
      args.minimumAlpha = Number(argv[++i]);
    } else if (arg === '--compression-level') {
      args.compressionLevel = Number(argv[++i]);
    } else if (arg === '--spherical-harmonics-degree') {
      args.sphericalHarmonicsDegree = Number(argv[++i]);
    } else if (arg === '--no-optimize') {
      args.optimizeSplatData = false;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.input || !args.output) {
    throw new Error('Usage: node convert_citygs_ksplat.mjs --input <ply> --output <ksplat> [--minimum-alpha 1] [--compression-level 1] [--spherical-harmonics-degree 2]');
  }
  return args;
}

async function main() {
  if (!globalThis.window) {
    globalThis.window = globalThis;
  }
  if (!globalThis.self) {
    globalThis.self = globalThis;
  }
  const GS = await import('../tmp/gaussian_viewer/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js');
  const args = parseArgs(process.argv);
  const inputPath = path.resolve(args.input);
  const outputPath = path.resolve(args.output);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  console.log(`Reading ${inputPath}`);
  const fileData = fs.readFileSync(inputPath);
  const inputBuffer = fileData.buffer.slice(
    fileData.byteOffset,
    fileData.byteOffset + fileData.byteLength,
  );

  console.log(`Converting to ksplat (compression=${args.compressionLevel}, shDegree=${args.sphericalHarmonicsDegree})`);
  const splatBuffer = await GS.PlyLoader.loadFromFileData(
    inputBuffer,
    args.minimumAlpha,
    args.compressionLevel,
    args.optimizeSplatData,
    args.sphericalHarmonicsDegree,
  );

  const outputBytes = Buffer.from(
    splatBuffer.bufferData,
    splatBuffer.bufferData.byteOffset || 0,
    splatBuffer.bufferData.byteLength || splatBuffer.bufferData.length,
  );
  fs.writeFileSync(outputPath, outputBytes);
  console.log(`Wrote ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
