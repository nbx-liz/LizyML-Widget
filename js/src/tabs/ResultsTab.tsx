/** ResultsTab — status-based view: running/completed/failed + inference. */
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
      <div class="lzw-results-error">
        <div class="lzw-badge lzw-badge--error">Failed</div>
        <p>{error.message ?? "Unknown error"}</p>
        {error.traceback && (
          <pre class="lzw-pre">{error.traceback}</pre>
        )}
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

  return (
    <div class="lzw-results">
      <div class="lzw-results__header">
        <span class="lzw-badge lzw-badge--success">Completed</span>
        <span class="lzw-muted" style="margin-left:8px">
          {jobType} #{jobIndex} — {elapsedSec.toFixed(1)}s
        </span>
      </div>

      {/* Fit results */}
      {hasFit && (
        <>
          <Accordion title="Metrics">
            <ScoreTable metrics={fitSummary.metrics} />
          </Accordion>
          {fitSummary.params?.length > 0 && (
            <Accordion title="Parameters" defaultOpen={false}>
              <ParamsTable params={fitSummary.params} />
            </Accordion>
          )}
        </>
      )}

      {/* Tune results */}
      {hasTune && (
        <Accordion title="Tuning Results">
          <div class="lzw-form-row">
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
      )}

      {/* Plots */}
      {availablePlots.length > 0 && (
        <Accordion title="Plots">
          <PlotViewer
            availablePlots={availablePlots}
            plots={plots}
            loading={plotLoading}
            onRequest={onRequestPlot}
          />
        </Accordion>
      )}

      {/* Inference */}
      <Accordion title="Inference" defaultOpen={hasInference}>
        {!hasInference && inferenceResult?.status !== "failed" && (
          <div class="lzw-inference-actions">
            <p class="lzw-muted">
              Load inference data from Python: <code>w.load_inference(df)</code>
            </p>
            {inferenceResult?.status === "ready" && (
              <button
                class="lzw-btn lzw-btn--primary"
                onClick={() => sendAction("run_inference", { return_shap: false })}
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
            {inferenceResult.warnings?.length > 0 && (
              <div class="lzw-muted" style="margin-bottom:8px">
                Warnings: {inferenceResult.warnings.join(", ")}
              </div>
            )}
            <PredTable data={inferenceResult.data} />
          </>
        )}
      </Accordion>
    </div>
  );
}
