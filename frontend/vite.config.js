import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
    proxy: {
      '/data': 'http://localhost:8765',
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/@mkkellogg/gaussian-splats-3d')) return 'gaussian-splats';
          if (id.includes('node_modules/three-mesh-bvh')) return 'spatial-index';
          if (id.includes('node_modules/three/')) return 'three-engine';
          return undefined;
        },
      },
    },
  },
});
