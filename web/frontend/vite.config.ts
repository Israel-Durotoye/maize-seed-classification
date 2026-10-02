import { defineConfig } from "vite";

// `npm run dev` proxies API calls to the Python backend on port 8000.
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
