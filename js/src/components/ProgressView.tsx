/** ProgressView — progress bar + elapsed time + cancel button + fold logs.
 *
 * Tune job progress carries optional re-tune fields (P-027) so the widget
 * can render round-aware status:
 *   - `round` / `total_rounds`: current Optuna study round (1-indexed)
 *   - `cumulative_trials`: total trials across every round so far
 *   - `expanded_dims`: dim names that were expanded on the current round
 *   - `latest_score` / `latest_state`: most recent trial outcome
 *   - `best_score`: running best of the current study
 *
 * All re-tune fields are optional: single-round tuning and pre-0.9.0 backends
 * collapse to the legacy display.
 */

interface ProgressViewProps {
  jobType: string;
  jobIndex: number;
  progress: {
    current?: number;
    total?: number;
    message?: string;
    round?: number;
    total_rounds?: number;
    cumulative_trials?: number;
    expanded_dims?: string[];
    latest_score?: number | null;
    latest_state?: string;
    best_score?: number | null;
    fold_results?: { label: string; score?: string; done: boolean }[];
  };
  elapsedSec: number;
  onCancel: () => void;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function ProgressView({
  jobType,
  jobIndex,
  progress,
  elapsedSec,
  onCancel,
}: ProgressViewProps) {
  const current = progress.current ?? 0;
  const total = progress.total ?? 0;
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  const folds = progress.fold_results ?? [];

  // Re-tune (P-027) round-aware fields.
  const round = progress.round ?? 1;
  const totalRounds = progress.total_rounds;
  const cumulative = progress.cumulative_trials ?? current;
  const expanded = progress.expanded_dims ?? [];
  const bestScore = progress.best_score ?? null;
  const showRoundBadge = jobType === "tune" && (round > 1 || (totalRounds ?? 0) > 1);

  return (
    <div class="lzw-progress">
      <div class="lzw-progress__header">
        <span class="lzw-progress__title">
          {jobType.charAt(0).toUpperCase() + jobType.slice(1)} #{jobIndex}
        </span>
        {showRoundBadge && (
          <span class="lzw-badge lzw-badge--round" title="Re-tune round">
            Round {round}
            {totalRounds ? `/${totalRounds}` : ""}
          </span>
        )}
        <span class="lzw-badge lzw-badge--running">Running</span>
      </div>

      {jobType === "tune" ? (
        <>
          {total > 0 && current > 0 ? (
            <div class="lzw-progress__bar-wrap">
              <div class="lzw-progress__bar" style={{ width: `${pct}%` }} />
            </div>
          ) : (
            <div class="lzw-progress__bar-wrap">
              <div class="lzw-progress__bar lzw-progress__bar--indeterminate" />
            </div>
          )}
          <div class="lzw-progress__info">
            <span>
              {total > 0
                ? `Trial ${current}/${total}`
                : "Tuning..."}
              {showRoundBadge && total > 0 && (
                <span class="lzw-progress__cumulative">
                  {" "}(cumulative: {cumulative})
                </span>
              )}
              {progress.message && total <= 0 ? ` ${progress.message}` : ""}
            </span>
            <span>Elapsed: {formatTime(elapsedSec)}</span>
          </div>
          {bestScore !== null && bestScore !== undefined && (
            <div class="lzw-progress__best">
              Best Score: <strong>{bestScore.toFixed(4)}</strong>
            </div>
          )}
          {expanded.length > 0 && (
            <div class="lzw-progress__expanded">
              <span class="lzw-muted">Boundary expansion this round:</span>{" "}
              {expanded.map((d) => (
                <span key={d} class="lzw-chip lzw-chip--info">
                  {d}
                </span>
              ))}
            </div>
          )}
        </>
      ) : (
        <>
          {total > 0 && (
            <div class="lzw-progress__bar-wrap">
              {current > 0 ? (
                <div class="lzw-progress__bar" style={{ width: `${pct}%` }} />
              ) : (
                <div class="lzw-progress__bar lzw-progress__bar--indeterminate" />
              )}
            </div>
          )}
          <div class="lzw-progress__info">
            <span>
              {total > 0
                ? `Fold ${current} / ${total}${progress.message ? `  ${progress.message}` : ""}`
                : progress.message ?? "Processing..."}
            </span>
            <span>Elapsed: {formatTime(elapsedSec)}</span>
          </div>
        </>
      )}

      {folds.length > 0 && (
        <div class="lzw-progress__folds">
          {folds.map((f, i) => (
            <div key={i} class="lzw-progress__fold-row">
              <span class="lzw-progress__fold-label">{f.label}</span>
              <span class="lzw-progress__fold-score">
                {f.done ? `${f.score ?? ""} \u2713` : f.score ? f.score : "\u2500"}
              </span>
            </div>
          ))}
        </div>
      )}

      <div class="lzw-progress__actions">
        <button class="lzw-btn" onClick={onCancel} type="button">
          Cancel
        </button>
      </div>
    </div>
  );
}
