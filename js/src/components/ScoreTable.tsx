/** ScoreTable — renders fit metrics in a structured table. */

interface ScoreTableProps {
  metrics: Record<string, any>;
}

export function ScoreTable({ metrics }: ScoreTableProps) {
  if (!metrics || !Object.keys(metrics).length) {
    return <p class="lzw-muted">No metrics available.</p>;
  }

  // Flatten nested metrics into rows
  const rows = flattenMetrics(metrics);

  return (
    <div class="lzw-table-wrap">
      <table class="lzw-table">
        <thead>
          <tr>
            <th>Category</th>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td>{row.category}</td>
              <td>{row.metric}</td>
              <td class="lzw-table__num">{formatValue(row.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface MetricRow {
  category: string;
  metric: string;
  value: number | string;
}

function flattenMetrics(
  obj: Record<string, any>,
  prefix = "",
): MetricRow[] {
  const rows: MetricRow[] = [];
  for (const [key, val] of Object.entries(obj)) {
    if (val && typeof val === "object" && !Array.isArray(val)) {
      rows.push(...flattenMetrics(val, prefix ? `${prefix} / ${key}` : key));
    } else {
      rows.push({
        category: prefix || "-",
        metric: key,
        value: val,
      });
    }
  }
  return rows;
}

function formatValue(v: any): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  }
  return String(v ?? "-");
}
