/** ScoreTable — renders fit metrics in a structured table. */

interface ScoreTableProps {
  metrics: Record<string, any>;
}

export function ScoreTable({ metrics }: ScoreTableProps) {
  if (!metrics || !Object.keys(metrics).length) {
    return <p class="lzw-muted">No metrics available.</p>;
  }

  // Determine if CV was used by checking for oos_std
  const hasOosStd = Object.values(metrics).some(
    (v: any) => v && typeof v === "object" && v.oos_std !== undefined
  );

  return (
    <div class="lzw-table-wrap">
      <table class="lzw-table">
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
            return (
              <tr key={metricName}>
                <td class="lzw-table__name">{metricName}</td>
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
