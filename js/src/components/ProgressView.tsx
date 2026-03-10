/** ProgressView — progress bar + elapsed time + cancel button. */

interface ProgressViewProps {
  jobType: string;
  jobIndex: number;
  progress: { current?: number; total?: number; message?: string };
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
          <div class="lzw-progress__bar" style={{ width: `${pct}%` }} />
        </div>
      )}

      <div class="lzw-progress__info">
        <span>{progress.message ?? "Processing..."}</span>
        <span>Elapsed: {formatTime(elapsedSec)}</span>
      </div>

      <div class="lzw-progress__actions">
        <button class="lzw-btn" onClick={onCancel} type="button">
          Cancel
        </button>
      </div>
    </div>
  );
}
