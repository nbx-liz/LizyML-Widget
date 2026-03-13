/** ProgressView — progress bar + elapsed time + cancel button + fold logs. */

interface ProgressViewProps {
  jobType: string;
  jobIndex: number;
  progress: {
    current?: number;
    total?: number;
    message?: string;
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

  return (
    <div class="lzw-progress">
      <div class="lzw-progress__header">
        <span class="lzw-progress__title">
          {jobType.charAt(0).toUpperCase() + jobType.slice(1)} #{jobIndex}
        </span>
        <span class="lzw-badge lzw-badge--running">Running</span>
      </div>

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
          {jobType === "tune" && total > 0
            ? `Trial ${current} / ${total}${progress.message ? `  ${progress.message}` : ""}`
            : total > 0
              ? `Fold ${current} / ${total}${progress.message ? `  ${progress.message}` : ""}`
              : progress.message ?? "Processing..."}
        </span>
        <span>Elapsed: {formatTime(elapsedSec)}</span>
      </div>

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
