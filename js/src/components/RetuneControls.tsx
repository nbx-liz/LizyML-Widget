/**
 * RetuneControls — launcher for P-028 re-tune (Study Resume + Boundary Expansion).
 *
 * Displays input controls for ``n_trials``, ``expand_boundary`` and
 * ``boundary_threshold`` alongside a "Re-tune (resume)" button.  When the
 * button is clicked the parent receives an ``onRetune`` callback with a
 * plain-object payload it can forward to ``sendAction("retune", ...)``.
 *
 * This component only shows after an initial tune has completed — the
 * parent is responsible for conditional rendering via ``hasTune``.
 *
 * Numeric inputs are wrapped in ``NumericStepper`` per the project-wide
 * numeric-input contract (see tests/test_frontend_contract.py); each
 * stepper receives an ``inputId`` so the surrounding ``<label htmlFor>``
 * reaches the actual ``<input>`` element inside the stepper.
 */
import { useCallback, useState } from "preact/hooks";
import { NumericStepper } from "./NumericStepper";

export interface RetunePayload {
  n_trials: number;
  expand_boundary: boolean;
  boundary_threshold: number;
}

interface RetuneControlsProps {
  /** Fired when the user clicks the Re-tune button. */
  onRetune: (payload: RetunePayload) => void;
  /** When true the button is grayed out (job already running, etc.). */
  disabled: boolean;
  /** Optional default n_trials; defaults to 20 so the bump is visible. */
  defaultNTrials?: number;
}

export function RetuneControls({
  onRetune,
  disabled,
  defaultNTrials = 20,
}: RetuneControlsProps) {
  const [nTrials, setNTrials] = useState<number>(defaultNTrials);
  const [expandBoundary, setExpandBoundary] = useState<boolean>(true);
  const [boundaryThreshold, setBoundaryThreshold] = useState<number>(0.05);

  const handleLaunch = useCallback(() => {
    onRetune({
      n_trials: nTrials,
      expand_boundary: expandBoundary,
      boundary_threshold: boundaryThreshold,
    });
  }, [onRetune, nTrials, expandBoundary, boundaryThreshold]);

  return (
    <div class="lzw-retune-controls" aria-label="Re-tune controls">
      <div class="lzw-retune-controls__title">Re-tune</div>
      <p class="lzw-muted lzw-retune-controls__hint">
        Resume the existing Optuna study with additional trials.
        Optionally widen boundaries that the best trial pushed against.
      </p>
      <div class="lzw-form-row">
        <label class="lzw-label" for="lzw-retune-n-trials">
          n_trials
        </label>
        <NumericStepper
          inputId="lzw-retune-n-trials"
          ariaLabel="n_trials"
          value={nTrials}
          min={1}
          max={1000}
          step={1}
          onChange={(v) => setNTrials(v ?? defaultNTrials)}
        />
      </div>
      <div class="lzw-form-row">
        <label class="lzw-label" for="lzw-retune-expand-boundary">
          expand_boundary
        </label>
        <input
          id="lzw-retune-expand-boundary"
          type="checkbox"
          checked={expandBoundary}
          onChange={(e) =>
            setExpandBoundary((e.target as HTMLInputElement).checked)
          }
        />
      </div>
      <div class="lzw-form-row">
        <label class="lzw-label" for="lzw-retune-threshold">
          boundary_threshold
        </label>
        <NumericStepper
          inputId="lzw-retune-threshold"
          ariaLabel="boundary_threshold"
          value={boundaryThreshold}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => setBoundaryThreshold(v ?? 0.05)}
        />
      </div>
      <div class="lzw-retune-controls__actions">
        <button
          class="lzw-btn lzw-btn--primary"
          type="button"
          disabled={disabled}
          onClick={handleLaunch}
        >
          Re-tune (resume) &#x21BB;
        </button>
      </div>
    </div>
  );
}
