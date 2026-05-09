import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/**/*.test.{ts,tsx}",
        "src/__tests__/**",
        // Bootstrap entry — exercised at runtime in the browser, no logic to
        // unit-test in JSDOM.
        "src/index.tsx",
      ],
      reporter: ["text", "html"],
      // #114 Phase C: enforce 75% line/statement coverage and 70% branch
      // coverage so regressions are caught at PR time.
      // #134 Phase 2.1 (2026-05-09): raised to 80% lines/statements and
      // 75% branches after closing the long-tail uncovered surfaces
      // (PlotViewer, DynForm nested, SearchSpace, ConfigTab branches).
      thresholds: {
        statements: 80,
        lines: 80,
        functions: 55,
        branches: 75,
      },
    },
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "preact",
  },
  resolve: {
    alias: {
      react: "preact/compat",
      "react-dom": "preact/compat",
    },
  },
});
