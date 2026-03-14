/** ParamsTable — key-value parameters table. */

interface ParamsTableProps {
  params: Record<string, any>[];
}

export function ParamsTable({ params }: ParamsTableProps) {
  if (!params.length) {
    return <p class="lzw-muted">No parameters available.</p>;
  }

  const keys = Object.keys(params[0]).filter((k) => k !== "index");

  return (
    <div class="lzw-table-wrap">
      <table class="lzw-table">
        <thead>
          <tr>
            {keys.map((k) => (
              <th key={k}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {params.map((row, i) => (
            <tr key={i}>
              {keys.map((k) => (
                <td key={k} class="lzw-table__num">
                  {formatCell(row[k])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(v: any): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toPrecision(4);
  }
  return String(v ?? "-");
}
