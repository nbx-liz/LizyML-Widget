/**
 * BoundaryExpansionPanel — renders the re-tune boundary expansion report (P-027).
 *
 * Given the `boundary_report` emitted by lizyml>=0.9.0 TuningResult, this panel
 * shows every search-space dimension that pushed against its boundary and
 * whether it was expanded on a subsequent resume round.
 *
 * Columns:
 *   - dim name
 *   - edge  (lower / upper / mid)
 *   - before  (original [low, high] range)
 *   - after   (expanded range, if any)
 *   - status  (▲ upper | ▼ lower | • unchanged)
 *
 * Unchanged dims are collapsed into a single footer line so the panel stays
 * compact when most dims converged.
 */

export interface BoundaryDimStatus {
  name: string;
  best_value: number | string | null;
  low: number | null;
  high: number | null;
  position_pct: number | null;
  edge: string; // "lower" | "upper" | "mid"
  expanded: boolean;
  new_low: number | null;
  new_high: number | null;
}

export interface BoundaryReport {
  dims: BoundaryDimStatus[];
  expanded_names: string[];
}

interface BoundaryExpansionPanelProps {
  report: BoundaryReport | null | undefined;
  /** When true the panel always renders even if every dim was unchanged. */
  showUnchanged?: boolean;
}

function formatRange(low: number | null, high: number | null): string {
  const lo = low === null || low === undefined ? "?" : formatNumber(low);
  const hi = high === null || high === undefined ? "?" : formatNumber(high);
  return `[${lo}, ${hi}]`;
}

function formatNumber(n: number): string {
  if (!Number.isFinite(n)) return String(n);
  const abs = Math.abs(n);
  if (abs !== 0 && (abs < 0.001 || abs >= 10000)) {
    return n.toExponential(2);
  }
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function edgeIcon(edge: string, expanded: boolean): string {
  if (!expanded) return "\u2022"; // bullet
  if (edge === "upper") return "\u25B2"; // ▲
  if (edge === "lower") return "\u25BC"; // ▼
  return "\u2194"; // ↔
}

export function BoundaryExpansionPanel({
  report,
  showUnchanged = false,
}: BoundaryExpansionPanelProps) {
  if (!report || !report.dims || report.dims.length === 0) {
    return null;
  }

  const dims = report.dims;
  const expanded = dims.filter((d) => d.expanded);
  const unchanged = dims.filter((d) => !d.expanded);

  // Hide panel entirely when nothing expanded and caller did not opt in.
  if (expanded.length === 0 && !showUnchanged) {
    return (
      <div class="lzw-boundary-panel lzw-boundary-panel--converged">
        <div class="lzw-boundary-panel__header">
          Boundary Expansion
        </div>
        <p class="lzw-muted">
          No expansion needed — every dimension stayed within its original
          range ({dims.length} dims checked).
        </p>
      </div>
    );
  }

  const rows = showUnchanged ? dims : expanded;

  return (
    <div class="lzw-boundary-panel">
      <div class="lzw-boundary-panel__header">
        Boundary Expansion
        <span class="lzw-muted">
          {" "}
          — {expanded.length}/{dims.length} dims expanded
        </span>
      </div>
      <div class="lzw-table-wrap">
        <table class="lzw-table lzw-boundary-table">
          <thead>
            <tr>
              <th>Dim</th>
              <th>Edge</th>
              <th>Before</th>
              <th>After</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d) => {
              const before = formatRange(d.low, d.high);
              const hadNew = d.new_low !== null || d.new_high !== null;
              const after = hadNew
                ? formatRange(d.new_low ?? d.low, d.new_high ?? d.high)
                : "\u2014";
              return (
                <tr key={d.name}>
                  <td class="lzw-boundary-table__name">{d.name}</td>
                  <td class="lzw-boundary-table__edge">
                    <span
                      class={`lzw-boundary-table__icon ${d.expanded ? "lzw-boundary-table__icon--expanded" : ""}`}
                    >
                      {edgeIcon(d.edge, d.expanded)}
                    </span>
                    {d.edge || "\u2014"}
                  </td>
                  <td class="lzw-boundary-table__range">{before}</td>
                  <td class="lzw-boundary-table__range">{after}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!showUnchanged && unchanged.length > 0 && (
        <p class="lzw-muted lzw-boundary-panel__footer">
          {unchanged.length} dim{unchanged.length === 1 ? "" : "s"} unchanged
        </p>
      )}
    </div>
  );
}
