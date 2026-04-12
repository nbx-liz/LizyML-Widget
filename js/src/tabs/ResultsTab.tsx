/** ResultsTab — status-based view: running/completed/failed + inference. */
import { useState, useEffect, useMemo, useCallback } from "preact/hooks";
import { Accordion } from "../components/Accordion";
import { ProgressView } from "../components/ProgressView";
import { ScoreTable } from "../components/ScoreTable";
import { ParamsTable } from "../components/ParamsTable";
import { PlotViewer } from "../components/PlotViewer";
import { PredTable } from "../components/PredTable";
import {
  BoundaryExpansionPanel,
  type BoundaryReport,
} from "../components/BoundaryExpansionPanel";
import {
  ScoreHistoryChart,
  type TrialRecord,
  type RoundRecord,
} from "../components/ScoreHistoryChart";
import { ConvergenceSignal } from "../components/ConvergenceSignal";
import { RetuneControls } from "../components/RetuneControls";
import type { ResolvedTheme } from "../hooks/useTheme";
import type { PlotRequestOptions } from "../hooks/usePlot";

interface ResultsTabProps {
  status: string;
  jobType: string;
  jobIndex: number;
  progress: Record<string, any>;
  elapsedSec: number;
  fitSummary: Record<string, any>;
  tuneSummary: Record<string, any>;
  availablePlots: string[];
  inferenceResult: Record<string, any>;
  error: Record<string, any>;
  plots: Record<string, any>;
  plotLoading: Record<string, boolean>;
  onRequestPlot: (plotType: string, options?: PlotRequestOptions) => void;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  onSwitchToFit?: () => void;
  /** Evaluation params for metric display annotations. */
  evaluationParams?: Record<string, any>;
  theme?: ResolvedTheme;
  /** Whether a code export is in progress (triggers browser download when complete). */
  exportLoading?: boolean;
  /** Callback to initiate a code export download. */
  onExportCode?: () => void;
}

/** Format a plot type slug into a display label. */
function plotLabel(slug: string): string {
  // Feature importance variants: "feature-importance-split" → "Importance (Split)"
  const importanceMatch = slug.match(/^feature-importance-(\w+)$/);
  if (importanceMatch) {
    const kind = importanceMatch[1].charAt(0).toUpperCase() + importanceMatch[1].slice(1);
    return `Importance (${kind})`;
  }
  return slug.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function ResultsTab({
  status,
  jobType,
  jobIndex,
  progress,
  elapsedSec,
  fitSummary,
  tuneSummary,
  availablePlots,
  inferenceResult,
  error,
  plots,
  plotLoading,
  onRequestPlot,
  sendAction,
  onSwitchToFit,
  evaluationParams,
  theme = "light",
  exportLoading = false,
  onExportCode,
}: ResultsTabProps) {
  const [selectedPlot, setSelectedPlot] = useState<string | null>(null);

  // Idle — no results yet
  if (status === "idle" || status === "data_loaded") {
    return (
      <div class="lzw-tab-placeholder">
        <p class="lzw-muted">Run Fit or Tune to see results here.</p>
      </div>
    );
  }

  // Running
  if (status === "running") {
    return (
      <ProgressView
        jobType={jobType}
        jobIndex={jobIndex}
        progress={progress}
        elapsedSec={elapsedSec}
        onCancel={() => sendAction("cancel")}
      />
    );
  }

  // Failed
  if (status === "failed") {
    return (
      <div class="lzw-results">
        <div class="lzw-results__header">
          <span class="lzw-badge lzw-badge--error">&#x2717; Failed</span>
          <span class="lzw-muted" style="margin-left:8px">
            {jobType.charAt(0).toUpperCase() + jobType.slice(1)} #{jobIndex}
          </span>
        </div>
        <div class="lzw-results-error">
          <p><strong>{error.code ?? "BACKEND_ERROR"}</strong></p>
          <p>{error.message ?? "Unknown error"}</p>
          {error.traceback && (
            <div style="margin-top: 12px;">
              <Accordion title="Show Full Traceback" defaultOpen={false}>
                <pre class="lzw-pre">{error.traceback}</pre>
              </Accordion>
            </div>
          )}
          <div style="margin-top: 12px;">
            <button class="lzw-btn" onClick={() => sendAction(jobType)} type="button">
              Re-run
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Completed — show fit or tune results
  const hasFit = fitSummary && Object.keys(fitSummary).length > 0;
  const hasTune = tuneSummary && Object.keys(tuneSummary).length > 0;
  const hasInference =
    inferenceResult &&
    inferenceResult.status === "completed" &&
    inferenceResult.data?.length > 0;

  // Auto-select first available plot, or keep user's selection if still valid
  const activePlot = (selectedPlot && availablePlots.includes(selectedPlot))
    ? selectedPlot
    : availablePlots[0] ?? null;

  // P-026: Extract model metric list from fitSummary.params for learning curve selector.
  // Combines native metrics (from "metric" row) and feval display names
  // (from "feval_metrics" row, e.g. "precision_at_k (k=20)").
  const modelMetrics: string[] = useMemo(() => {
    const params = fitSummary?.params;
    if (!Array.isArray(params)) return [];
    const result: string[] = [];
    // Native metrics from "metric" row (skip "None" — means feval-only)
    const metricRow = params.find((r: Record<string, any>) => r.parameter === "metric");
    if (metricRow) {
      const val = metricRow.value;
      if (Array.isArray(val)) {
        for (const item of val) {
          if (typeof item === "string" && item !== "None") result.push(item);
        }
      } else if (typeof val === "string" && val !== "None") {
        result.push(val);
      }
    }
    // Feval display names from "feval_metrics" row (comma-separated string)
    const fevalRow = params.find((r: Record<string, any>) => r.parameter === "feval_metrics");
    if (fevalRow && typeof fevalRow.value === "string") {
      for (const name of fevalRow.value.split(",")) {
        const trimmed = name.trim();
        if (trimmed) result.push(trimmed);
      }
    }
    return result;
  }, [fitSummary?.params]);

  // P-026: Track selected metric for learning curve (default: first metric)
  const [lcMetric, setLcMetric] = useState<string | null>(null);
  const activeLcMetric = lcMetric && modelMetrics.includes(lcMetric) ? lcMetric : modelMetrics[0] ?? null;

  // P-026: Request learning curve with selected metric
  const handleLcMetricChange = useCallback(
    (metric: string) => {
      setLcMetric(metric);
      onRequestPlot("learning-curve", { metrics: [metric] });
    },
    [onRequestPlot],
  );

  // P-026: Stable wrapper for PlotViewer — always applies metric filter for learning-curve
  const handlePlotRequest = useCallback(
    (pt: string) => {
      if (pt === "learning-curve" && activeLcMetric) {
        onRequestPlot(pt, { metrics: [activeLcMetric] });
      } else {
        onRequestPlot(pt);
      }
    },
    [onRequestPlot, activeLcMetric],
  );

  return (
    <div class="lzw-results">
      <div class="lzw-results__header">
        <span class="lzw-badge lzw-badge--success">&#x2713; Completed</span>
        <span class="lzw-muted" style="margin-left:8px">
          {jobType.charAt(0).toUpperCase() + jobType.slice(1)} #{jobIndex} — {elapsedSec.toFixed(1)}s
        </span>
        {hasFit && (
          <button
            class="lzw-btn lzw-btn--accent-outline"
            style="margin-left:auto"
            onClick={onExportCode}
            disabled={exportLoading}
            type="button"
          >
            {exportLoading ? "Exporting..." : "Export Code"}
          </button>
        )}
      </div>

      {/* Tune specific results */}
      {hasTune && (
        <>
          <Accordion title="Best Params" defaultOpen={true}>
            {tuneSummary.best_params && Object.keys(tuneSummary.best_params).length > 0 ? (
              <>
                <ParamsTable params={[tuneSummary.best_params]} />
                <div style="margin-top: 8px;">
                  <button
                    class="lzw-btn lzw-btn--primary"
                    onClick={() => {
                      sendAction("apply_best_params", { params: tuneSummary.best_params });
                      onSwitchToFit?.();
                    }}
                    type="button"
                  >
                    Apply to Fit &#x25B8;
                  </button>
                </div>
              </>
            ) : (
              <p class="lzw-muted">No best params available.</p>
            )}

            {/* P-028: Re-tune launcher — shown on tune completion only.
                When a new job is running the ResultsTab switches to the
                Running view earlier in this component, so in practice the
                Best Params accordion is not mounted during that window.
                The ``disabled`` prop keeps the button inert as a belt-and-
                suspenders measure in case a future layout change keeps
                both views visible at once. */}
            <RetuneControls
              disabled={status === "running"}
              onRetune={(payload) => sendAction("retune", payload)}
            />
            <div class="lzw-form-row" style="margin-top: 8px;">
              <span class="lzw-label">Best Score</span>
              <span>
                {tuneSummary.metric_name}: {tuneSummary.best_score?.toFixed(4)}
              </span>
            </div>
            {(() => {
              const trials: any[] = tuneSummary.trials ?? [];
              const stateOf = (t: any) => String(t.state ?? "").toUpperCase();
              const complete = trials.filter((t: any) => stateOf(t) === "COMPLETE").length;
              const pruned = trials.filter((t: any) => stateOf(t) === "PRUNED").length;
              const failed = trials.filter((t: any) => stateOf(t) === "FAIL").length;
              const rounds: any[] = tuneSummary.rounds ?? [];
              return (
                <>
                  <div class="lzw-form-row">
                    <span class="lzw-label">Trials</span>
                    <span>
                      {trials.length} total
                      {complete > 0 && ` / ${complete} complete`}
                      {pruned > 0 && ` / ${pruned} pruned`}
                      {failed > 0 && ` / ${failed} failed`}
                    </span>
                  </div>
                  {rounds.length > 1 && (
                    <div class="lzw-form-row">
                      <span class="lzw-label">Rounds</span>
                      <span>{rounds.length} resume rounds</span>
                    </div>
                  )}
                </>
              );
            })()}
          </Accordion>

          {/* P-027: Convergence signal when the last resume round did not expand anything. */}
          {(() => {
            const rounds = (tuneSummary.rounds ?? []) as RoundRecord[];
            const lastRound = rounds.length > 0 ? rounds[rounds.length - 1] : null;
            const report = (tuneSummary.boundary_report ?? null) as BoundaryReport | null;
            const expandedInLast = lastRound ? lastRound.expanded_dims.length : 0;
            if (lastRound && lastRound.round >= 1 && expandedInLast === 0 && rounds.length > 1) {
              return (
                <ConvergenceSignal
                  round={lastRound.round + 1 /* 0-indexed → 1-indexed */}
                  checkedDims={report?.dims.length ?? 0}
                  onProceedToFit={() => {
                    if (tuneSummary.best_params) {
                      sendAction("apply_best_params", { params: tuneSummary.best_params });
                    }
                    onSwitchToFit?.();
                  }}
                />
              );
            }
            return null;
          })()}

          {/* P-027: Boundary Expansion panel — only when re-tune ran resume rounds. */}
          {tuneSummary.boundary_report && (
            <Accordion title="Boundary Expansion" defaultOpen={true}>
              <BoundaryExpansionPanel
                report={tuneSummary.boundary_report as BoundaryReport}
              />
            </Accordion>
          )}

          {/* P-027: Score History chart — always shown for tune results. */}
          <Accordion title="Score History" defaultOpen={true}>
            <ScoreHistoryChart
              trials={(tuneSummary.trials ?? []) as TrialRecord[]}
              rounds={(tuneSummary.rounds ?? []) as RoundRecord[]}
              direction={String(tuneSummary.direction ?? "minimize")}
              metricName={String(tuneSummary.metric_name ?? "score")}
              theme={theme}
            />
          </Accordion>
        </>
      )}

      {/* Score (Fit) */}
      {hasFit && (
        <div style="margin-bottom: 12px;">
          <div style="font-weight: 600; margin-bottom: 8px;">Score</div>
          <ScoreTable metrics={fitSummary.metrics} evaluationParams={evaluationParams} />
        </div>
      )}

      {/* Unified Plots section */}
      {availablePlots.length > 0 && (
        <div style="margin-bottom: 12px;">
          <div style="font-weight: 600; margin-bottom: 8px;">Plots</div>
          <div class="lzw-chip-group" style="margin-bottom: 8px;">
            {availablePlots.map((p) => (
              <button
                key={p}
                type="button"
                class={`lzw-chip ${activePlot === p ? "lzw-chip--active" : ""}`}
                onClick={() => {
                  setSelectedPlot(p);
                  handlePlotRequest(p);
                }}
              >
                {plotLabel(p)}
              </button>
            ))}
          </div>
          {/* P-026: Metric selector for learning curve */}
          {activePlot === "learning-curve" && modelMetrics.length > 1 && (
            <div class="lzw-chip-group" style="margin-bottom: 8px;">
              {modelMetrics.map((m) => (
                <button
                  key={m}
                  type="button"
                  class={`lzw-chip ${activeLcMetric === m ? "lzw-chip--active" : ""}`}
                  onClick={() => handleLcMetricChange(m)}
                >
                  {m}
                </button>
              ))}
            </div>
          )}
          {activePlot && (
            <PlotViewer
              plotType={activePlot}
              plots={plots}
              loading={plotLoading}
              onRequest={handlePlotRequest}
              theme={theme}
            />
          )}
        </div>
      )}

      {/* Details: Fold Details + Parameters */}
      {hasFit && (
        <Accordion title="Details" defaultOpen={false}>
          {fitSummary.fold_details && fitSummary.fold_details.length > 0 && (
            <>
              <div style="font-weight: 600; margin-bottom: 4px; font-size: 13px;">Fold Details</div>
              <div class="lzw-table-wrap" style="margin-bottom: 12px;">
                <table class="lzw-table">
                  <thead>
                    <tr>
                      <th>Fold</th>
                      <th>Train Size</th>
                      <th>Val Size</th>
                      <th>Metrics</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fitSummary.fold_details.map((fd: any, i: number) => (
                      <tr key={i}>
                        <td class="lzw-table__num">{i + 1}</td>
                        <td class="lzw-table__num">{fd.train_size}</td>
                        <td class="lzw-table__num">{fd.val_size}</td>
                        <td>{JSON.stringify(fd.metrics)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {fitSummary.params?.length > 0 && (
            <>
              <div style="font-weight: 600; margin-bottom: 4px; font-size: 13px;">Parameters</div>
              <ParamsTable params={fitSummary.params} />
            </>
          )}
        </Accordion>
      )}

      {/* Inference */}
      <InferenceSection
        inferenceResult={inferenceResult}
        hasInference={hasInference}
        plots={plots}
        plotLoading={plotLoading}
        sendAction={sendAction}
        theme={theme}
      />
    </div>
  );
}

/** Inference section with SHAP checkbox per BLUEPRINT §5.4. */
function InferenceSection({
  inferenceResult,
  hasInference,
  plots,
  plotLoading,
  sendAction,
  theme = "light",
}: {
  inferenceResult: Record<string, any>;
  hasInference: boolean;
  plots: Record<string, any>;
  plotLoading: Record<string, boolean>;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  theme?: ResolvedTheme;
}) {
  const [returnShap, setReturnShap] = useState(false);
  const [inferenceLoading, setInferenceLoading] = useState(false);

  // Reset loading state when inference result arrives
  useEffect(() => {
    if (inferenceResult?.status === "completed" || inferenceResult?.status === "failed") {
      setInferenceLoading(false);
    }
  }, [inferenceResult?.status]);

  return (
    <Accordion title="Inference" defaultOpen={hasInference}>
      {!hasInference && inferenceResult?.status !== "failed" && (
        <div class="lzw-inference-actions">
          <p class="lzw-muted">
            Load inference data from Python: <code>w.load_inference(df)</code>
          </p>
          <div class="lzw-form-row" style="margin: 8px 0;">
            <label>
              <input
                type="checkbox"
                checked={returnShap}
                onChange={(e) => setReturnShap((e.target as HTMLInputElement).checked)}
              />
              {" "}Return SHAP values
            </label>
          </div>
          {inferenceResult?.status === "ready" && (
            <button
              class="lzw-btn lzw-btn--primary"
              disabled={inferenceLoading}
              onClick={() => {
                setInferenceLoading(true);
                sendAction("run_inference", { return_shap: returnShap });
              }}
              type="button"
            >
              {inferenceLoading ? "Running..." : `Run Inference (${inferenceResult.rows} rows)`}
            </button>
          )}
        </div>
      )}
      {inferenceResult?.status === "failed" && (
        <div class="lzw-results-error">
          <div class="lzw-badge lzw-badge--error">Inference Failed</div>
          <p>{inferenceResult.message}</p>
        </div>
      )}
      {hasInference && (
        <>
          <PredTable data={inferenceResult.data} />
          <Accordion title="Prediction Distribution" defaultOpen={false}>
            <PlotViewer
              plotType="prediction-distribution"
              plots={plots}
              loading={plotLoading}
              onRequest={(pt) => sendAction("request_inference_plot", { plot_type: pt })}
              theme={theme}
            />
          </Accordion>
          {returnShap && (
            <Accordion title="SHAP Summary" defaultOpen={false}>
              <PlotViewer
                plotType="shap-summary"
                plots={plots}
                loading={plotLoading}
                onRequest={(pt) => sendAction("request_inference_plot", { plot_type: pt })}
                theme={theme}
              />
            </Accordion>
          )}
          {inferenceResult.warnings?.length > 0 && (
            <Accordion title="Warnings" defaultOpen={false}>
              <ul>
                {inferenceResult.warnings.map((w: string, i: number) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </Accordion>
          )}
        </>
      )}
    </Accordion>
  );
}
