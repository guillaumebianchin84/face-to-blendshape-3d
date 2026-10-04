import { defineConfig } from 'vite';

export default defineConfig({
  base: '/face-to-blendshape-3d/',
  server: {
    port: 3000,
    open: true,
    host: true
  },
  preview: {
    allowedHosts: [
      'face-to-blendshape-3d.onrender.com',
      '.onrender.com'
    ]
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three'],
          mediapipe: ['@mediapipe/tasks-vision']
        }
      }
    }
  },
  optimizeDeps: {
    include: ['three', '@mediapipe/tasks-vision']
  }
});
