/**
 * PlotViewer — renders a single Plotly chart.
 * Loads Plotly from CDN (or reuses window.Plotly if available).
 */
import { useEffect, useRef } from "preact/hooks";

interface PlotViewerProps {
  plotType: string;
  plots: Record<string, any>;
  loading: Record<string, boolean>;
  onRequest: (plotType: string) => void;
}

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
      Plotly.newPlot(el, spec.data, spec.layout ?? {}, {
        responsive: true,
        displayModeBar: false,
      });
    });

    return () => {
      cancelled = true;
    };
  }, [plots, plotType]);

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
