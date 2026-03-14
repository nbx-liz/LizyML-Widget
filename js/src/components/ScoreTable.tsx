/** ScoreTable — renders fit metrics in a structured table. */

interface ScoreTableProps {
  metrics: Record<string, any>;
  /** Optional evaluation params for metric display (e.g. precision_at_k_k). */
  evaluationParams?: Record<string, any>;
}

export function ScoreTable({ metrics, evaluationParams }: ScoreTableProps) {
  if (!metrics || !Object.keys(metrics).length) {
    return <p class="lzw-muted">No metrics available.</p>;
  }

  // Determine if CV was used by checking for oos_std
  const hasOosStd = Object.values(metrics).some(
    (v: any) => v && typeof v === "object" && v.oos_std !== undefined
  );

  return (
    <div class="lzw-table-wrap">
      <table class="lzw-table lzw-table--compact">
        <thead>
          <tr>
            <th>Metric</th>
            <th>IS</th>
            <th>OOS</th>
            {hasOosStd && <th>OOS Std</th>}
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([metricName, metricData]) => {
            const isVal = metricData?.is;
            const oosVal = metricData?.oos;
            const oosStd = metricData?.oos_std;
            // Annotate precision_at_k with k value
            let displayName = metricName;
            if (metricName === "precision_at_k") {
              const k = evaluationParams?.precision_at_k_k ?? 10;
              displayName = `precision_at_k (k=${k})`;
            }
            return (
              <tr key={metricName}>
                <td class="lzw-table__name">{displayName}</td>
                <td class="lzw-table__num">{formatValue(isVal)}</td>
                <td class="lzw-table__num">{formatValue(oosVal)}</td>
                {hasOosStd && (
                  <td class="lzw-table__num">{formatValue(oosStd)}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function formatValue(v: any): string {
  if (v === undefined || v === null) return "-";
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  }
  return String(v);
}
