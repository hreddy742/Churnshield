import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3002,
    proxy: {
      "/predict": { target: "http://localhost:8090", changeOrigin: true },
      "/explain": { target: "http://localhost:8090", changeOrigin: true },
      "/batch":   { target: "http://localhost:8090", changeOrigin: true },
      "/monitor": { target: "http://localhost:8090", changeOrigin: true },
      "/health":  { target: "http://localhost:8090", changeOrigin: true },
    },
  },
});
