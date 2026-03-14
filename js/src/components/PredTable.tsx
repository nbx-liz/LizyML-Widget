/** PredTable — paginated prediction results table with CSV download. */
import { useState, useMemo } from "preact/hooks";

interface PredTableProps {
  data: Record<string, any>[];
  pageSize?: number;
}

export function PredTable({ data, pageSize = 50 }: PredTableProps) {
  const [page, setPage] = useState(0);

  const columns = useMemo(() => {
    if (!data.length) return [];
    return Object.keys(data[0]);
  }, [data]);

  if (!data.length) {
    return <p class="lzw-muted">No predictions available.</p>;
  }

  const totalPages = Math.ceil(data.length / pageSize);
  const pageData = data.slice(page * pageSize, (page + 1) * pageSize);

  const handleDownload = () => {
    const header = columns.join(",");
    const rows = data.map((row) =>
      columns.map((c) => String(row[c] ?? "")).join(","),
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div class="lzw-pred-table">
      <div class="lzw-pred-table__toolbar">
        <span class="lzw-muted">{data.length} rows</span>
        <button class="lzw-btn" onClick={handleDownload} type="button">
          Download CSV
        </button>
      </div>

      <div class="lzw-table-wrap">
        <table class="lzw-table">
          <thead>
            <tr>
              <th>#</th>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageData.map((row, i) => (
              <tr key={page * pageSize + i}>
                <td class="lzw-table__num">{page * pageSize + i}</td>
                {columns.map((c) => (
                  <td key={c} class="lzw-table__num">
                    {formatCell(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div class="lzw-pred-table__pagination">
          <button
            class="lzw-btn"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
            type="button"
          >
            Prev
          </button>
          <span class="lzw-muted">
            {page + 1} / {totalPages}
          </span>
          <button
            class="lzw-btn"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
            type="button"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

function formatCell(v: any): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  }
  return String(v ?? "-");
}
