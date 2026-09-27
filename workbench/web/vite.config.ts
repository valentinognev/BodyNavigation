import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/catalog": "http://127.0.0.1:8001",
      "/cases": "http://127.0.0.1:8001",
      "/run": "http://127.0.0.1:8001",
      "/browse": "http://127.0.0.1:8001",
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
