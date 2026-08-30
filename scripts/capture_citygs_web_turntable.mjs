import fs from 'fs';
import path from 'path';
import puppeteer from '../tmp/gaussian_viewer/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js';

function parseArgs(argv) {
  const args = {
    width: 1600,
    height: 900,
    frames: 48,
    settleMs: 250,
    warmupMs: 6000,
    chromePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    baseUrl: 'http://127.0.0.1:8088',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--scene') {
      args.scene = argv[++i];
    } else if (arg === '--output-dir') {
      args.outputDir = argv[++i];
    } else if (arg === '--width') {
      args.width = Number(argv[++i]);
    } else if (arg === '--height') {
      args.height = Number(argv[++i]);
    } else if (arg === '--frames') {
      args.frames = Number(argv[++i]);
    } else if (arg === '--settle-ms') {
      args.settleMs = Number(argv[++i]);
    } else if (arg === '--warmup-ms') {
      args.warmupMs = Number(argv[++i]);
    } else if (arg === '--chrome-path') {
      args.chromePath = argv[++i];
    } else if (arg === '--base-url') {
      args.baseUrl = argv[++i];
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.scene || !args.outputDir) {
    throw new Error('Usage: node capture_citygs_web_turntable.mjs --scene <Residence|SciArt> --output-dir <dir> [--width 1600] [--height 900] [--frames 48]');
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv);
  const outputDir = path.resolve(args.outputDir);
  fs.mkdirSync(outputDir, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: args.chromePath,
    headless: true,
    protocolTimeout: 0,
    defaultViewport: {
      width: args.width,
      height: args.height,
      deviceScaleFactor: 1,
    },
    args: ['--disable-web-security', '--allow-file-access-from-files', '--no-sandbox'],
  });

  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(0);
    page.on('console', (message) => {
      console.log(`[browser:${message.type()}] ${message.text()}`);
    });
    page.on('pageerror', (error) => {
      console.log(`[pageerror] ${error}`);
    });
    const url = `${args.baseUrl}/citygs_web/index.html?scene=${encodeURIComponent(args.scene)}`;
    console.log(`Opening ${url}`);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 0 });
    await page.waitForFunction(() => window.captureReady === true, { timeout: 0 });
    await new Promise((resolve) => setTimeout(resolve, args.warmupMs));

    const metadata = await page.evaluate(() => window.getCaptureMetadata());
    fs.writeFileSync(
      path.join(outputDir, `${args.scene}_capture_metadata.json`),
      JSON.stringify(metadata, null, 2),
      'utf8',
    );

    for (let index = 0; index < args.frames; index += 1) {
      const t = index / args.frames;
      await page.evaluate((value) => window.setOrbitT(value), t);
      await new Promise((resolve) => setTimeout(resolve, args.settleMs));
      const framePath = path.join(outputDir, `${String(index).padStart(4, '0')}.png`);
      await page.screenshot({ path: framePath, type: 'png' });
      console.log(`Captured ${framePath}`);
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
