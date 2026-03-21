/** FoldPreview — fold preview visualization with period flow and detail table. */

interface FoldInfo {
  fold: number;
  period_label: string;
  group_label: string;
  train_size: number;
  valid_size: number;
}

interface FoldPreviewProps {
  totalFolds: number;
  timeFolds: number;
  groupFolds: number;
  periods: string[];
  folds: FoldInfo[];
  mode: string;
}

/**
 * Build visual period flow rows from the fold data.
 * Groups folds by time fold (unique period_label patterns).
 */
function buildTimeFoldRows(
  folds: FoldInfo[],
  groupFolds: number,
  periods: string[],
): { label: string; blocks: { text: string; role: "train" | "valid" | "unused" }[] }[] {
  if (folds.length === 0 || periods.length === 0) return [];

  const rows: { label: string; blocks: { text: string; role: "train" | "valid" | "unused" }[] }[] = [];
  const timeFoldCount = groupFolds > 0 ? Math.ceil(folds.length / groupFolds) : folds.length;

  for (let tf = 0; tf < timeFoldCount; tf++) {
    const representative = folds[tf * groupFolds];
    if (!representative) break;

    // Parse the period_label to identify train/valid periods
    // Format: "P0+P1 -> P2" or "P0 -> P1"
    const label = `Time Fold ${tf}`;
    const pl = representative.period_label;
    const arrowIdx = pl.indexOf("->");
    const trainPart = arrowIdx >= 0 ? pl.substring(0, arrowIdx).trim() : "";
    const validPart = arrowIdx >= 0 ? pl.substring(arrowIdx + 2).trim() : "";

    const trainPeriods = new Set(trainPart.split("+").map((s) => s.trim()));
    const validPeriods = new Set(validPart.split("+").map((s) => s.trim()));

    const blocks = periods.map((p) => {
      if (trainPeriods.has(p)) return { text: `${p} (train)`, role: "train" as const };
      if (validPeriods.has(p)) return { text: `${p} (valid)`, role: "valid" as const };
      return { text: p, role: "unused" as const };
    });

    rows.push({ label, blocks });
  }

  return rows;
}

export function FoldPreview({
  totalFolds,
  timeFolds,
  groupFolds,
  periods,
  folds,
  mode,
}: FoldPreviewProps) {
  const flowRows = buildTimeFoldRows(folds, groupFolds, periods);

  return (
    <div>
      {/* Summary badge */}
      <div class="lzw-summary-badge">
        Total: <span class="lzw-summary-badge__num">{totalFolds}</span> folds
        <span class="lzw-muted">
          ({timeFolds} time x {groupFolds} groups)
        </span>
      </div>

      {/* Visual period flow diagram */}
      {flowRows.length > 0 && (
        <div style={{ marginTop: "12px" }}>
          {flowRows.map((row, i) => (
            <div key={i} class="lzw-fold-vis-row">
              <div class="lzw-fold-vis-label">
                {row.label} x Group Folds
              </div>
              <div class="lzw-period-flow">
                {row.blocks.map((block, j) => (
                  <div
                    key={j}
                    class={`lzw-period-block lzw-period-block--${block.role}`}
                  >
                    {block.text}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Divider */}
      {folds.length > 0 && (
        <div style={{ borderTop: "1px solid var(--lzw-border)", margin: "12px 0" }} />
      )}

      {/* Detailed fold table */}
      {folds.length > 0 && (
        <div class="lzw-fold-preview">
          <div class="lzw-fold-line lzw-fold-header">
            <span>Fold</span>
            <span>Structure</span>
            <span style={{ textAlign: "right" }}>Train</span>
            <span style={{ textAlign: "right" }}>Valid</span>
          </div>
          {folds.map((f) => (
            <div key={f.fold} class="lzw-fold-line">
              <span>{f.fold}</span>
              <span>
                {(() => {
                  const arrowIdx = f.period_label.indexOf("->");
                  const trainP = arrowIdx >= 0 ? f.period_label.substring(0, arrowIdx).trim() : f.period_label;
                  const validP = arrowIdx >= 0 ? f.period_label.substring(arrowIdx + 2).trim() : "";
                  return (
                    <>
                      <span class="lzw-train-tag">{trainP}</span>
                      {" \u2192 "}
                      <span class="lzw-valid-tag">{validP}</span>
                      {f.group_label && ` \u00B7 ${f.group_label}`}
                    </>
                  );
                })()}
              </span>
              <span style={{ textAlign: "right" }}>{f.train_size.toLocaleString()}</span>
              <span style={{ textAlign: "right" }}>{f.valid_size.toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}

      {folds.length === 0 && (
        <p class="lzw-muted" style={{ marginTop: "8px" }}>
          Configure blocks and groups to preview folds.
          {mode && ""}
        </p>
      )}
    </div>
  );
}
