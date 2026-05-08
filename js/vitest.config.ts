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
      thresholds: {
        statements: 75,
        lines: 75,
        functions: 50,
        branches: 70,
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
