/** ResultsTab — status-based view: running/completed/failed + inference. */
import { useState } from "preact/hooks";
import { Accordion } from "../components/Accordion";
import { ProgressView } from "../components/ProgressView";
import { ScoreTable } from "../components/ScoreTable";
import { ParamsTable } from "../components/ParamsTable";
import { PlotViewer } from "../components/PlotViewer";
import { PredTable } from "../components/PredTable";

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
  onRequestPlot: (plotType: string) => void;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  onSwitchToFit?: () => void;
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
}: ResultsTabProps) {
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
          <span class="lzw-badge lzw-badge--error">✗ Failed</span>
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

  const otherPlots = availablePlots.filter((p) => p !== "learning-curve" && p !== "optimization-history");

  return (
    <div class="lzw-results">
      <div class="lzw-results__header">
        <span class="lzw-badge lzw-badge--success">✓ Completed</span>
        <span class="lzw-muted" style="margin-left:8px">
          {jobType.charAt(0).toUpperCase() + jobType.slice(1)} #{jobIndex} — {elapsedSec.toFixed(1)}s
        </span>
      </div>

      {/* Tune specific results */}
      {hasTune && (
        <>
          <Accordion title="Optimization History" defaultOpen={true}>
            {availablePlots.includes("optimization-history") && (
              <PlotViewer
                availablePlots={["optimization-history"]}
                plots={plots}
                loading={plotLoading}
                onRequest={onRequestPlot}
              />
            )}
          </Accordion>
          <Accordion title="Best Params">
            {tuneSummary.best_params && Object.keys(tuneSummary.best_params).length > 0 ? (
              <>
                <ParamsTable params={[tuneSummary.best_params]} />
                <div style="margin-top: 8px;">
                  <button
                    class="lzw-btn"
                    onClick={() => {
                      sendAction("apply_best_params", { params: tuneSummary.best_params });
                      onSwitchToFit?.();
                    }}
                    type="button"
                  >
                    Apply to Fit ▸
                  </button>
                </div>
              </>
            ) : (
              <p class="lzw-muted">No best params available.</p>
            )}
            <div class="lzw-form-row" style="margin-top: 8px;">
              <span class="lzw-label">Best Score</span>
              <span>
                {tuneSummary.metric_name}: {tuneSummary.best_score?.toFixed(4)}
              </span>
            </div>
            <div class="lzw-form-row">
              <span class="lzw-label">Trials</span>
              <span>{tuneSummary.trials?.length ?? 0}</span>
            </div>
          </Accordion>
        </>
      )}

      {/* Fit results */}
      {hasFit && (
        <>
          <div style="margin-bottom: 8px;">
            <div style="font-weight: 600; margin-bottom: 8px;">── Score ──</div>
            <ScoreTable metrics={fitSummary.metrics} />
          </div>

          {availablePlots.includes("learning-curve") && (
            <div style="margin-bottom: 16px;">
              <div style="font-weight: 600; margin-bottom: 8px;">── Learning Curve ──</div>
              <PlotViewer
                availablePlots={["learning-curve"]}
                plots={plots}
                loading={plotLoading}
                onRequest={onRequestPlot}
              />
            </div>
          )}

          {otherPlots.length > 0 && (
            <div style="font-weight: 600; margin-bottom: 8px;">
              ── Plots{" "}
              <select
                class="lzw-select"
                style="font-weight: 400; margin-left: 8px;"
                onChange={(e) => onRequestPlot((e.target as HTMLSelectElement).value)}
              >
                <option value="">-- Select --</option>
                {otherPlots.map(p => <option key={p} value={p}>{p.replace(/-/g, " ")}</option>)}
              </select>
              {" "}──
            </div>
          )}
          {otherPlots.some(p => plots[p]) && (
            <div style="margin-bottom: 16px;">
              <PlotViewer
                availablePlots={otherPlots.filter(p => plots[p])}
                plots={plots}
                loading={plotLoading}
                onRequest={() => {}} // Handle dynamically by external select
              />
            </div>
          )}

          <Accordion title="Feature Importance" defaultOpen={false}>
            {availablePlots.includes("feature-importance") ? (
              <PlotViewer
                availablePlots={["feature-importance"]}
                plots={plots}
                loading={plotLoading}
                onRequest={onRequestPlot}
              />
            ) : (
              <p class="lzw-muted">Feature importance not available.</p>
            )}
          </Accordion>
          
          {fitSummary.fold_details && fitSummary.fold_details.length > 0 && (
            <Accordion title="Fold Details" defaultOpen={false}>
              <div class="lzw-table-wrap">
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
            </Accordion>
          )}

          {fitSummary.params?.length > 0 && (
            <Accordion title="Parameters" defaultOpen={false}>
              <ParamsTable params={fitSummary.params} />
            </Accordion>
          )}
        </>
      )}

      {/* Inference */}
      <InferenceSection
        inferenceResult={inferenceResult}
        hasInference={hasInference}
        plots={plots}
        plotLoading={plotLoading}
        sendAction={sendAction}
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
}: {
  inferenceResult: Record<string, any>;
  hasInference: boolean;
  plots: Record<string, any>;
  plotLoading: Record<string, boolean>;
  sendAction: (type: string, payload?: Record<string, any>) => void;
}) {
  const [returnShap, setReturnShap] = useState(false);

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
              onClick={() => sendAction("run_inference", { return_shap: returnShap })}
              type="button"
            >
              Run Inference ({inferenceResult.rows} rows)
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
              availablePlots={["prediction-distribution"]}
              plots={plots}
              loading={plotLoading}
              onRequest={(pt) => sendAction("request_inference_plot", { plot_type: pt })}
            />
          </Accordion>
          {returnShap && (
            <Accordion title="SHAP Summary" defaultOpen={false}>
              <PlotViewer
                availablePlots={["shap-summary"]}
                plots={plots}
                loading={plotLoading}
                onRequest={(pt) => sendAction("request_inference_plot", { plot_type: pt })}
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
