/** BlockedGroupKFold — 2-axis CV configuration (Period x Entity). */
import { useState, useEffect } from "preact/hooks";
import { DistributionBar } from "./DistributionBar";
import { FoldPreview } from "./FoldPreview";
import { NumericStepper } from "./NumericStepper";

interface BlockedGroupKFoldProps {
  cv: any;
  allColumns: string[];
  columnStats: Record<string, any> | null;
  splitPreview: any | null;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  sendCv: (cv: any) => void;
}

/**
 * Compute resulting periods from sorted unique values and selected cutoffs.
 * Cutoffs mark the last value in a period boundary.
 */
function computePeriods(
  uniqueValues: string[],
  cutoffs: string[],
): { label: string; values: string[] }[] {
  if (uniqueValues.length === 0) return [];

  const cutoffSet = new Set(cutoffs);
  const periods: { label: string; values: string[] }[] = [];
  let current: string[] = [];

  for (const v of uniqueValues) {
    current.push(v);
    if (cutoffSet.has(v) || v === uniqueValues[uniqueValues.length - 1]) {
      const first = current[0];
      const last = current[current.length - 1];
      const label = first === last ? first : `${first} ~ ${last}`;
      periods.push({ label, values: [...current] });
      current = [];
    }
  }

  // If there are remaining values (shouldn't happen, but defensive)
  if (current.length > 0) {
    const first = current[0];
    const last = current[current.length - 1];
    periods.push({ label: first === last ? first : `${first} ~ ${last}`, values: [...current] });
  }

  return periods;
}

export function BlockedGroupKFold({
  cv,
  allColumns,
  columnStats,
  splitPreview,
  sendAction,
  sendCv,
}: BlockedGroupKFoldProps) {
  const blocks = cv.blocks ?? {};
  const groups = cv.groups ?? {};
  const blocksCol = blocks.col ?? "";
  const groupsCol = groups.col ?? "";
  const cutoffs: string[] = blocks.cutoffs ?? [];
  const mode: string = blocks.mode ?? "expanding";
  const trainWindow: number = blocks.train_window ?? 1;
  const nSplits: number = groups.n_splits ?? 3;
  const stratify: string = groups.stratify ?? "auto";
  const shuffle: boolean = groups.shuffle ?? true;
  const minTrainRows: number = cv.min_train_rows ?? 10;
  const minValidRows: number = cv.min_valid_rows ?? 5;

  // Track last requested column to avoid redundant requests
  const [lastRequestedCol, setLastRequestedCol] = useState<string>("");

  // Request column stats when blocks.col changes
  useEffect(() => {
    if (blocksCol && blocksCol !== lastRequestedCol) {
      sendAction("get_column_stats", { column: blocksCol });
      setLastRequestedCol(blocksCol);
    }
  }, [blocksCol, lastRequestedCol, sendAction]);

  // Current column stats data
  const statsForCol =
    columnStats && blocksCol && columnStats.column === blocksCol
      ? columnStats
      : null;
  const uniqueValues: { value: string; count: number }[] =
    statsForCol?.values ?? [];
  const uniqueCount: number = statsForCol?.unique_count ?? uniqueValues.length;

  // Compute resulting periods from cutoffs
  const allUniqueVals = uniqueValues.map((v) => v.value);
  const periods = computePeriods(allUniqueVals, cutoffs);

  // Group column stats (for showing unique count)
  const [lastRequestedGroupCol, setLastRequestedGroupCol] = useState<string>("");
  const groupStatsForCol =
    columnStats && groupsCol && columnStats.column === groupsCol
      ? columnStats
      : null;

  useEffect(() => {
    if (groupsCol && groupsCol !== lastRequestedGroupCol && groupsCol !== blocksCol) {
      sendAction("get_column_stats", { column: groupsCol });
      setLastRequestedGroupCol(groupsCol);
    }
  }, [groupsCol, lastRequestedGroupCol, blocksCol, sendAction]);

  // Toggle cutoff selection
  const toggleCutoff = (value: string) => {
    const isLast = value === allUniqueVals[allUniqueVals.length - 1];
    if (isLast) return; // Can't toggle last value as cutoff

    const newCutoffs = cutoffs.includes(value)
      ? cutoffs.filter((c) => c !== value)
      : [...cutoffs, value];

    sendCv({
      ...cv,
      blocks: { ...blocks, cutoffs: newCutoffs },
    });
  };

  // Request preview when config changes meaningfully
  const requestPreview = () => {
    if (blocksCol && groupsCol && cutoffs.length > 0) {
      sendAction("preview_splits", {});
    }
  };

  // Period labels for fold preview
  const periodLabels = periods.map((_, i) => `P${i}`);

  return (
    <div>
      <div class="lzw-info-box">
        Period (time) x Entity (group) 2-axis CV. Evaluates generalization to
        future periods AND unseen entities.
      </div>

      {/* ── Blocks Section ── */}
      <div class="lzw-section-card" style={{ marginTop: "12px" }}>
        <div class="lzw-section-card__title">Blocks (Period Axis)</div>

        <div class="lzw-form-row">
          <label class="lzw-label">Period Column</label>
          <select
            class="lzw-select"
            value={blocksCol}
            onChange={(e) => {
              const v = (e.target as HTMLSelectElement).value;
              sendCv({
                ...cv,
                blocks: { ...blocks, col: v || null, cutoffs: [] },
              });
            }}
          >
            <option value="">-- Select --</option>
            {allColumns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        {/* Value distribution */}
        {statsForCol && uniqueValues.length > 0 && (
          <>
            <div style={{ margin: "8px 0 4px 0" }}>
              <span class="lzw-label">Value Distribution</span>
              <span
                class="lzw-muted"
                style={{ marginLeft: "8px", fontFamily: "monospace", fontSize: "10px" }}
              >
                {uniqueCount} unique values
              </span>
            </div>
            <div style={{ padding: "0 4px" }}>
              <DistributionBar values={uniqueValues} />
            </div>
          </>
        )}

        {/* Cutoff chips */}
        {allUniqueVals.length > 0 && (
          <>
            <div style={{ marginTop: "12px" }}>
              <span class="lzw-label">Cutoffs</span>
              <span
                class="lzw-muted"
                style={{ marginLeft: "8px", fontSize: "10px" }}
              >
                Click values to set period boundaries
              </span>
            </div>
            <div class="lzw-cutoff-list">
              {allUniqueVals.map((v) => {
                const isSelected = cutoffs.includes(v);
                const isLast = v === allUniqueVals[allUniqueVals.length - 1];
                return (
                  <span
                    key={v}
                    class={`lzw-cutoff-chip ${isSelected ? "lzw-cutoff-chip--selected" : ""}`}
                    style={isLast ? { opacity: 0.4, cursor: "default" } : undefined}
                    onClick={() => toggleCutoff(v)}
                  >
                    {v}
                  </span>
                );
              })}
            </div>
          </>
        )}

        {/* Resulting periods */}
        {periods.length > 1 && (
          <>
            <div style={{ marginTop: "10px" }}>
              <span class="lzw-label">Resulting Periods</span>
            </div>
            <div
              style={{
                fontFamily: "monospace",
                fontSize: "11px",
                color: "var(--lzw-fg-1)",
                marginTop: "4px",
                lineHeight: "1.8",
              }}
            >
              {periods.map((p, i) => {
                const rowCount = uniqueValues
                  .filter((v) => p.values.includes(v.value))
                  .reduce((sum, v) => sum + v.count, 0);
                return (
                  <div key={i}>
                    P{i}: {p.label}{" "}
                    <span style={{ color: "var(--lzw-fg-2)" }}>
                      ({rowCount.toLocaleString()} rows)
                    </span>
                  </div>
                );
              })}
            </div>
          </>
        )}

        <div
          style={{ borderTop: "1px solid var(--lzw-border)", margin: "12px 0" }}
        />

        {/* Mode */}
        <div class="lzw-form-row">
          <label class="lzw-label">Mode</label>
          <div class="lzw-segment">
            {["expanding", "sliding"].map((m) => (
              <button
                key={m}
                type="button"
                class={`lzw-segment__btn ${mode === m ? "lzw-segment__btn--active" : ""}`}
                aria-pressed={mode === m}
                onClick={() => {
                  sendCv({ ...cv, blocks: { ...blocks, mode: m } });
                }}
              >
                {m.charAt(0).toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div class="lzw-muted" style={{ marginLeft: "98px", marginTop: "-2px", fontSize: "10px" }}>
          {mode === "expanding"
            ? "Training data grows cumulatively with each fold"
            : "Fixed-size training window slides forward"}
        </div>

        {/* Train Window (only for sliding) */}
        {mode === "sliding" && (
          <div class="lzw-form-row" style={{ marginTop: "8px" }}>
            <label class="lzw-label">Train Window</label>
            <NumericStepper
              value={trainWindow}
              min={1}
              max={periods.length > 0 ? periods.length - 1 : 10}
              step={1}
              onChange={(v) =>
                sendCv({ ...cv, blocks: { ...blocks, train_window: v ?? 1 } })
              }
            />
          </div>
        )}
      </div>

      {/* ── Groups Section ── */}
      <div class="lzw-section-card">
        <div class="lzw-section-card__title">Groups (Entity Axis)</div>

        <div class="lzw-form-row">
          <label class="lzw-label">Group Column</label>
          <select
            class="lzw-select"
            value={groupsCol}
            onChange={(e) => {
              const v = (e.target as HTMLSelectElement).value;
              sendCv({ ...cv, groups: { ...groups, col: v || null } });
            }}
          >
            <option value="">-- Select --</option>
            {allColumns
              .filter((c) => c !== blocksCol)
              .map((c) => {
                const hint =
                  groupStatsForCol && groupStatsForCol.column === c
                    ? ` (${groupStatsForCol.unique_count} unique)`
                    : "";
                return (
                  <option key={c} value={c}>
                    {c}{hint}
                  </option>
                );
              })}
            {blocksCol && (
              <option disabled style={{ color: "var(--lzw-fg-2)" }}>
                {blocksCol} (used as blocks.col)
              </option>
            )}
          </select>
        </div>

        <div class="lzw-form-row">
          <label class="lzw-label">Folds (n_splits)</label>
          <NumericStepper
            value={nSplits}
            min={2}
            max={10}
            step={1}
            onChange={(v) =>
              sendCv({ ...cv, groups: { ...groups, n_splits: v ?? 3 } })
            }
          />
        </div>

        <div class="lzw-form-row">
          <label class="lzw-label">Stratify</label>
          <div class="lzw-segment">
            {["auto", "on", "off"].map((s) => (
              <button
                key={s}
                type="button"
                class={`lzw-segment__btn ${stratify === s ? "lzw-segment__btn--active" : ""}`}
                aria-pressed={stratify === s}
                onClick={() =>
                  sendCv({ ...cv, groups: { ...groups, stratify: s } })
                }
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div class="lzw-form-row">
          <label class="lzw-label">Shuffle</label>
          <label class="lzw-toggle">
            <input
              type="checkbox"
              checked={shuffle}
              onChange={(e) =>
                sendCv({
                  ...cv,
                  groups: {
                    ...groups,
                    shuffle: (e.target as HTMLInputElement).checked,
                  },
                })
              }
            />
            <span class="lzw-toggle__slider" />
          </label>
        </div>
      </div>

      {/* ── Fold Preview Section ── */}
      <div class="lzw-section-card">
        <div class="lzw-section-card__title">
          Fold Preview
          <button
            type="button"
            class="lzw-btn"
            style={{ marginLeft: "8px", padding: "2px 8px", fontSize: "10px" }}
            onClick={requestPreview}
            disabled={!blocksCol || !groupsCol || cutoffs.length === 0}
          >
            Refresh
          </button>
        </div>

        {splitPreview ? (
          <FoldPreview
            totalFolds={splitPreview.total_folds ?? 0}
            timeFolds={splitPreview.time_folds ?? 0}
            groupFolds={splitPreview.group_folds ?? 0}
            periods={splitPreview.periods ?? periodLabels}
            folds={splitPreview.folds ?? []}
            mode={mode}
          />
        ) : (
          <p class="lzw-muted">
            {blocksCol && groupsCol && cutoffs.length > 0
              ? "Click Refresh to preview folds."
              : "Configure blocks and groups to preview folds."}
          </p>
        )}

        <div
          style={{ borderTop: "1px solid var(--lzw-border)", margin: "12px 0" }}
        />

        <div class="lzw-form-row">
          <label class="lzw-label">Min Train Rows</label>
          <NumericStepper
            value={minTrainRows}
            min={1}
            step={1}
            onChange={(v) => sendCv({ ...cv, min_train_rows: v ?? 10 })}
          />
        </div>
        <div class="lzw-form-row">
          <label class="lzw-label">Min Valid Rows</label>
          <NumericStepper
            value={minValidRows}
            min={1}
            step={1}
            onChange={(v) => sendCv({ ...cv, min_valid_rows: v ?? 5 })}
          />
        </div>
      </div>
    </div>
  );
}
