/**
 * ConvergenceSignal — banner shown when a re-tune round completes without
 * expanding any boundary (P-027).
 *
 * LizyML interprets "no expansion needed" as a convergence signal: the
 * optimizer believes the current search space is already sufficient for the
 * best-score region, so further resume rounds are unlikely to help.  We
 * surface this as an affirmative banner with a direct path to `fit()`.
 *
 * The banner is only shown for rounds 2+ (i.e. a resume round) because a
 * first-round tune has nothing to compare against.
 */

interface ConvergenceSignalProps {
  /** Round number that produced this verdict (1-indexed). */
  round: number;
  /** Number of dims checked — used only for the "Nothing to expand" copy. */
  checkedDims: number;
  /** Called when the user clicks "Proceed to Fit". */
  onProceedToFit?: () => void;
}

export function ConvergenceSignal({
  round,
  checkedDims,
  onProceedToFit,
}: ConvergenceSignalProps) {
  if (round < 2) return null;

  return (
    <div class="lzw-convergence-signal" role="status">
      <div class="lzw-convergence-signal__icon" aria-hidden="true">
        \u2713
      </div>
      <div class="lzw-convergence-signal__body">
        <div class="lzw-convergence-signal__title">
          Search space converged
        </div>
        <p class="lzw-convergence-signal__text">
          Round {round} finished without expanding any boundary
          {checkedDims > 0 ? ` (${checkedDims} dims checked)` : ""}.
          The current search space is sufficient — additional resume rounds
          are unlikely to help.
        </p>
      </div>
      {onProceedToFit && (
        <button
          type="button"
          class="lzw-btn lzw-btn--primary"
          onClick={onProceedToFit}
        >
          Proceed to Fit &#x25B8;
        </button>
      )}
    </div>
  );
}
