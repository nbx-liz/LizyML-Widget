/**
 * PlotViewer — plot selector + Plotly rendering.
 * Loads Plotly from CDN (or reuses window.Plotly if available).
 */
import { useState, useEffect, useRef } from "preact/hooks";

interface PlotViewerProps {
  availablePlots: string[];
  plots: Record<string, any>;
  loading: Record<string, boolean>;
  onRequest: (plotType: string) => void;
}

async function getPlotly(): Promise<any> {
  if ((window as any).Plotly) return (window as any).Plotly;
  // Dynamic import from CDN
  const mod = await import(
    /* @vite-ignore */
    "https://esm.sh/plotly.js-dist-min@2.35.0"
  );
  return mod.default ?? mod;
}

export function PlotViewer({
  availablePlots,
  plots,
  loading,
  onRequest,
}: PlotViewerProps) {
  const [selected, setSelected] = useState<string>("");
  const plotRef = useRef<HTMLDivElement>(null);

  // Auto-select first plot
  useEffect(() => {
    if (!selected && availablePlots.length) {
      setSelected(availablePlots[0]);
    }
  }, [availablePlots, selected]);

  // Request plot when selected
  useEffect(() => {
    if (selected) onRequest(selected);
  }, [selected, onRequest]);

  // Render with Plotly
  useEffect(() => {
    const spec = plots[selected];
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
  }, [plots, selected]);

  if (!availablePlots.length) {
    return <p class="lzw-muted">No plots available.</p>;
  }

  return (
    <div class="lzw-plot-viewer">
      <div class="lzw-form-row">
        <label class="lzw-label">Plot</label>
        <select
          class="lzw-select"
          value={selected}
          onChange={(e) => setSelected((e.target as HTMLSelectElement).value)}
        >
          {availablePlots.map((p) => (
            <option key={p} value={p}>
              {p.replace(/-/g, " ")}
            </option>
          ))}
        </select>
      </div>

      <div class="lzw-plot-viewer__container">
        {loading[selected] && (
          <p class="lzw-muted">Loading plot...</p>
        )}
        <div ref={plotRef} class="lzw-plot-viewer__canvas" />
      </div>
    </div>
  );
}
