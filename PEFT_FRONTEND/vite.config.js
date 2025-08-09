import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/wardrobex',  // 设置打包基础路径
  server: {
    proxy: {
      '/recommend': {
        target: 'http://wardrobex_backend:8000',
        changeOrigin: true,
      }
    }
  }
});
