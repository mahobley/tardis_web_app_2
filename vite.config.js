import { resolve } from "node:path";
import { defineConfig } from "vite";

const repoRoot = resolve(__dirname);

export default defineConfig({
  root: resolve(repoRoot, "web"),
  base: "./",
  assetsInclude: ["**/*.aris", "**/*.ddf"],
  server: {
    fs: {
      allow: [repoRoot],
    },
    watch: {
      usePolling: true,
    },
  },
  build: {
    outDir: resolve(repoRoot, "dist-web"),
    emptyOutDir: true,
  },
});
