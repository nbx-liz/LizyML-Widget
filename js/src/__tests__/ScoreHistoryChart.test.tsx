/**
 * Tests for ScoreHistoryChart (P-027 re-tune monitoring).
 *
 * Plotly.js is loaded from a CDN inside the component, so these tests focus
 * on:
 *   - empty trial list → "No trial data available yet."
 *   - non-empty trial list → canvas div is rendered (Plotly attach is
 *     deferred to an async effect we stub out below)
 */
import { beforeEach, describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/preact";
import {
  ScoreHistoryChart,
  type TrialRecord,
  type RoundRecord,
} from "../components/ScoreHistoryChart";

// Provide a stub `window.Plotly` so the component's `getPlotly()` helper
// short-circuits before attempting the remote CDN import.  This keeps the
// tests hermetic and avoids jsdom trying to resolve an https:// ESM URL.
beforeEach(() => {
  (window as unknown as { Plotly: { newPlot: ReturnType<typeof vi.fn> } }).Plotly = {
    newPlot: vi.fn(),
  };
});

describe("ScoreHistoryChart", () => {
  it("shows placeholder when there are no trials", () => {
    render(
      <ScoreHistoryChart
        trials={[]}
        direction="minimize"
        metricName="rmse"
      />,
    );
    expect(screen.getByText(/No trial data available yet/)).toBeDefined();
  });

  it("ignores non-finite / NaN score trials", () => {
    const trials: TrialRecord[] = [
      { number: 1, score: Number.NaN, state: "FAIL" },
    ];
    render(
      <ScoreHistoryChart trials={trials} direction="minimize" metricName="rmse" />,
    );
    expect(screen.getByText(/No trial data available yet/)).toBeDefined();
  });

  it("renders the canvas wrapper when trials are present", () => {
    const trials: TrialRecord[] = [
      { number: 1, score: 0.5, state: "COMPLETE", round: 1 },
      { number: 2, score: 0.48, state: "COMPLETE", round: 1 },
      { number: 3, score: 0.47, state: "COMPLETE", round: 2 },
    ];
    const rounds: RoundRecord[] = [
      {
        round: 1,
        n_trials: 2,
        best_score_before: null,
        best_score_after: 0.48,
        expanded_dims: [],
      },
      {
        round: 2,
        n_trials: 1,
        best_score_before: 0.48,
        best_score_after: 0.47,
        expanded_dims: ["learning_rate"],
      },
    ];
    const { container } = render(
      <ScoreHistoryChart
        trials={trials}
        rounds={rounds}
        direction="minimize"
        metricName="rmse"
      />,
    );
    expect(container.querySelector(".lzw-score-history")).not.toBeNull();
    expect(container.querySelector(".lzw-score-history__canvas")).not.toBeNull();
  });
});
