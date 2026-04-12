/**
 * ScoreHistoryChart — Plotly.js trial-score history with round separators.
 *
 * Reads the `tune_summary` traitlet (`trials` + `rounds`) and renders:
 *   - scatter of per-trial scores (color-coded by state)
 *   - line of the running best score
 *   - vertical dashed lines at round boundaries (re-tune rounds)
 *   - round annotations listing the dims expanded on each boundary
 *
 * Plotly is imported dynamically from the same CDN as PlotViewer so the main
 * bundle stays small.
 */
import { useEffect, useRef, useMemo } from "preact/hooks";

import type { ResolvedTheme } from "../hooks/useTheme";

export interface TrialRecord {
  number: number;
  score: number;
  state: string; // "COMPLETE" | "PRUNED" | "FAIL"
  round?: number;
  params?: Record<string, unknown>;
}

export interface RoundRecord {
  round: number;
  n_trials: number;
  best_score_before: number | null;
  best_score_after: number;
  expanded_dims: string[];
}

interface ScoreHistoryChartProps {
  trials: TrialRecord[];
  rounds?: RoundRecord[];
  direction: string; // "minimize" | "maximize"
  metricName?: string;
  theme?: ResolvedTheme;
}

async function getPlotly(): Promise<any> {
  if ((window as any).Plotly) return (window as any).Plotly;
  const mod = await import(
    /* @vite-ignore */ // @ts-expect-error dynamic CDN import
    "https://esm.sh/plotly.js-dist-min@2.35.0"
  );
  return mod.default ?? mod;
}

/** State → color for completed/pruned/failed trials. */
const STATE_COLOR: Record<string, string> = {
  COMPLETE: "#4caf50",
  PRUNED: "#ffb74d",
  FAIL: "#e57373",
};

/** Compute the running best score (monotone) based on direction. */
function runningBest(
  trials: TrialRecord[],
  direction: string,
): (number | null)[] {
  const maximize = direction === "maximize";
  const out: (number | null)[] = [];
  let best: number | null = null;
  for (const t of trials) {
    const s = t.score;
    if (!Number.isFinite(s) || t.state !== "COMPLETE") {
      out.push(best);
      continue;
    }
    if (best === null) best = s;
    else if (maximize && s > best) best = s;
    else if (!maximize && s < best) best = s;
    out.push(best);
  }
  return out;
}

export function ScoreHistoryChart({
  trials,
  rounds = [],
  direction,
  metricName = "score",
  theme = "light",
}: ScoreHistoryChartProps) {
  const divRef = useRef<HTMLDivElement>(null);

  // Filter to trials with a numeric score; keep original trial number for X.
  const cleanTrials = useMemo(
    () =>
      (trials ?? []).filter(
        (t) => typeof t.score === "number" && Number.isFinite(t.score),
      ),
    [trials],
  );

  // Group trials by state so each gets its own trace color.
  const tracesByState = useMemo(() => {
    const groups: Record<string, { x: number[]; y: number[] }> = {};
    for (const t of cleanTrials) {
      const state = t.state || "COMPLETE";
      if (!groups[state]) groups[state] = { x: [], y: [] };
      groups[state].x.push(t.number);
      groups[state].y.push(t.score);
    }
    return groups;
  }, [cleanTrials]);

  // Compute round boundaries (cumulative trial count at the end of each round).
  const boundaries = useMemo(() => {
    if (!rounds || rounds.length <= 1) return [] as RoundRecord[];
    return rounds.slice(0, -1); // skip the last round (no line after it)
  }, [rounds]);

  const bestSeries = useMemo(
    () => runningBest(cleanTrials, direction),
    [cleanTrials, direction],
  );

  useEffect(() => {
    const el = divRef.current;
    if (!el || cleanTrials.length === 0) return;

    let cancelled = false;
    getPlotly().then((Plotly) => {
      if (cancelled) return;

      const isDark = theme === "dark";
      const gridColor = isDark ? "#555" : "#ddd";
      const fontColor = isDark ? "#e0e0e0" : "#333";

      // Build the scatter traces per state.
      const traces: any[] = [];
      for (const [state, { x, y }] of Object.entries(tracesByState)) {
        traces.push({
          type: "scatter",
          mode: "markers",
          name: state,
          x,
          y,
          marker: {
            color: STATE_COLOR[state] ?? "#888",
            size: 7,
            line: { width: 0 },
          },
          hovertemplate: `Trial %{x}<br>${metricName}: %{y:.4f}<br>${state}<extra></extra>`,
        });
      }

      // Running best line.
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "Best",
        x: cleanTrials.map((t) => t.number),
        y: bestSeries,
        line: {
          color: isDark ? "#64b5f6" : "#1976d2",
          width: 2,
          shape: "hv",
        },
        hovertemplate: `Trial %{x}<br>Best: %{y:.4f}<extra></extra>`,
      });

      // Vertical dashed lines at round boundaries.  Each `boundary` is the
      // round index where a new resume started; we place the divider after
      // the cumulative trial count of the round that just finished.
      let cumulative = 0;
      const shapes: any[] = [];
      const annotations: any[] = [];
      for (const r of rounds ?? []) {
        cumulative += r.n_trials;
        const isLast = r === rounds[rounds.length - 1];
        if (!isLast) {
          shapes.push({
            type: "line",
            xref: "x",
            yref: "paper",
            x0: cumulative + 0.5,
            x1: cumulative + 0.5,
            y0: 0,
            y1: 1,
            line: { color: isDark ? "#999" : "#aaa", width: 1, dash: "dash" },
          });
        }
        if ((r.expanded_dims ?? []).length > 0) {
          annotations.push({
            x: cumulative,
            y: 1.02,
            xref: "x",
            yref: "paper",
            text: `R${r.round + 1}: +${r.expanded_dims.join(", ")}`,
            showarrow: false,
            font: { size: 10, color: fontColor },
            xanchor: "right",
          });
        }
      }
      void boundaries; // referenced to stabilize memoized hook inputs

      const layout: any = {
        autosize: true,
        margin: { t: 32, r: 24, l: 56, b: 44 },
        showlegend: true,
        legend: {
          orientation: "h",
          y: -0.18,
          x: 0,
          font: { color: fontColor, size: 11 },
        },
        xaxis: {
          title: { text: "Trial", font: { color: fontColor, size: 12 } },
          gridcolor: gridColor,
          zerolinecolor: gridColor,
          tickfont: { color: fontColor },
        },
        yaxis: {
          title: { text: metricName, font: { color: fontColor, size: 12 } },
          gridcolor: gridColor,
          zerolinecolor: gridColor,
          tickfont: { color: fontColor },
        },
        paper_bgcolor: "transparent",
        plot_bgcolor: isDark ? "#2d2d2d" : "#fafafa",
        shapes,
        annotations,
        font: { color: fontColor },
      };

      Plotly.newPlot(el, traces, layout, {
        responsive: true,
        displayModeBar: false,
      });
    });

    return () => {
      cancelled = true;
    };
  }, [
    cleanTrials,
    tracesByState,
    bestSeries,
    rounds,
    boundaries,
    metricName,
    theme,
  ]);

  if (cleanTrials.length === 0) {
    return (
      <p class="lzw-muted">
        No trial data available yet.
      </p>
    );
  }

  return (
    <div class="lzw-score-history">
      <div
        ref={divRef}
        class="lzw-score-history__canvas"
        style="min-height: 260px;"
      />
    </div>
  );
}
