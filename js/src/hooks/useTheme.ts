/** useTheme — manages light/dark theme with system preference detection. */
import { useState, useEffect, useCallback } from "preact/hooks";

type ThemeMode = "light" | "dark" | "auto";
export type ResolvedTheme = "light" | "dark";

function getSystemTheme(): ResolvedTheme {
  if (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    return "dark";
  }
  return "light";
}

function resolve(mode: ThemeMode): ResolvedTheme {
  return mode === "auto" ? getSystemTheme() : mode;
}

export function useTheme(rootEl: HTMLElement | null) {
  const [mode, setMode] = useState<ThemeMode>("auto");
  const [resolved, setResolved] = useState<ResolvedTheme>(() => resolve("auto"));

  // Apply data attribute on root element and propagate to Jupyter containers
  useEffect(() => {
    if (!rootEl) return;
    const theme = resolve(mode);
    setResolved(theme);
    rootEl.setAttribute("data-lzw-theme", theme);

    // Make known Jupyter/Colab output containers transparent so dark mode shows through.
    // Only target recognized container classes to avoid affecting sibling widgets or host themes.
    const OUTPUT_CLASSES = [
      "jp-OutputArea-output", "jp-OutputArea-child", "jp-OutputArea",
      "widget-output", "widget-subarea", "output_subarea",
      "output_area", "output", "output_wrapper",
      "cell-output-ipywidget-background",  // VS Code Jupyter
    ];
    const modified: Array<{ el: HTMLElement; prev: string }> = [];
    let ancestor: HTMLElement | null = rootEl.parentElement;
    const maxDepth = 10;
    for (let i = 0; ancestor && i < maxDepth; i++) {
      const tag = ancestor.tagName.toLowerCase();
      if (tag === "body" || tag === "html") break;
      const isOutputContainer = OUTPUT_CLASSES.some((cls) => ancestor!.classList.contains(cls));
      if (isOutputContainer) {
        modified.push({ el: ancestor, prev: ancestor.style.background });
        ancestor.style.setProperty("background", "transparent", "important");
      }
      ancestor = ancestor.parentElement;
    }
    return () => {
      for (const { el, prev } of modified) {
        el.style.background = prev;
      }
    };
  }, [rootEl, mode]);

  // Listen for system preference changes when mode is "auto"
  useEffect(() => {
    if (mode !== "auto") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      const theme = resolve("auto");
      setResolved(theme);
      if (rootEl) rootEl.setAttribute("data-lzw-theme", theme);
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [mode, rootEl]);

  const toggle = useCallback(() => {
    setMode((prev) => {
      if (prev === "auto") return getSystemTheme() === "dark" ? "light" : "dark";
      return prev === "light" ? "dark" : "light";
    });
  }, []);

  return { mode, resolved, toggle } as const;
}
