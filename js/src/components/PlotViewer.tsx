/**
 * PlotViewer — renders a single Plotly chart.
 * Loads Plotly from CDN (or reuses window.Plotly if available).
 */
import { useEffect, useRef } from "preact/hooks";

import type { ResolvedTheme } from "../hooks/useTheme";

interface PlotViewerProps {
  plotType: string;
  plots: Record<string, any>;
  loading: Record<string, boolean>;
  onRequest: (plotType: string) => void;
  theme?: ResolvedTheme;
}

/** Plotly layout overrides for dark mode. */
const DARK_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#2d2d2d",
  font: { color: "#e0e0e0" },
  xaxis: { gridcolor: "#555", zerolinecolor: "#555" },
  yaxis: { gridcolor: "#555", zerolinecolor: "#555" },
  colorway: [
    "#64b5f6", "#81c784", "#ffb74d", "#f28b82",
    "#ce93d8", "#4dd0e1", "#fff176", "#a1887f",
  ],
} as const;

const LIGHT_LAYOUT = {
  paper_bgcolor: "transparent",
} as const;

async function getPlotly(): Promise<any> {
  if ((window as any).Plotly) return (window as any).Plotly;
  // Dynamic import from CDN
  const mod = await import(
    /* @vite-ignore */ // @ts-expect-error dynamic CDN import
    "https://esm.sh/plotly.js-dist-min@2.35.0"
  );
  return mod.default ?? mod;
}

export function PlotViewer({
  plotType,
  plots,
  loading,
  onRequest,
  theme = "light",
}: PlotViewerProps) {
  const plotRef = useRef<HTMLDivElement>(null);

  // Request plot data when plotType changes
  useEffect(() => {
    if (plotType) onRequest(plotType);
  }, [plotType, onRequest]);

  // Render with Plotly
  useEffect(() => {
    const spec = plots[plotType];
    const el = plotRef.current;
    if (!spec || !el) return;

    let cancelled = false;
    getPlotly().then((Plotly) => {
      if (cancelled) return;
      const isDark = theme === "dark";
      const themeLayout = isDark ? DARK_LAYOUT : LIGHT_LAYOUT;
      const mergedLayout = { ...spec.layout, ...themeLayout };
      // Deep-merge axis properties to preserve backend-set labels/ranges (dark only)
      if (isDark) {
        if (spec.layout?.xaxis) {
          mergedLayout.xaxis = { ...spec.layout.xaxis, ...DARK_LAYOUT.xaxis };
        }
        if (spec.layout?.yaxis) {
          mergedLayout.yaxis = { ...spec.layout.yaxis, ...DARK_LAYOUT.yaxis };
        }
      }
      Plotly.newPlot(el, spec.data, mergedLayout, {
        responsive: true,
        displayModeBar: false,
      });
    });

    return () => {
      cancelled = true;
    };
  }, [plots, plotType, theme]);

  if (!plotType) {
    return <p class="lzw-muted">No plot selected.</p>;
  }

  return (
    <div class="lzw-plot-viewer">
      <div class="lzw-plot-viewer__container">
        {loading[plotType] && (
          <p class="lzw-muted">Loading plot...</p>
        )}
        <div ref={plotRef} class="lzw-plot-viewer__canvas" />
      </div>
    </div>
  );
}
