/** ColumnTable — column settings table for the Data Tab. */

interface Column {
  name: string;
  unique_count: number;
  excluded: boolean;
  col_type: string;
  exclude_reason?: string | null;
}

interface ColumnTableProps {
  columns: Column[];
  onUpdate: (name: string, excluded: boolean, colType: string) => void;
}

export function ColumnTable({ columns, onUpdate }: ColumnTableProps) {
  if (!columns.length) {
    return <p class="lzw-muted">Select a target to see column settings.</p>;
  }

  return (
    <div class="lzw-table-wrap">
      <table class="lzw-table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Unique</th>
            <th>Exclude</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((col) => (
            <tr key={col.name} class={col.excluded ? "lzw-row--excluded" : ""}>
              <td class="lzw-table__name">
                {col.name}
                {col.exclude_reason && (
                  <span class="lzw-tag lzw-tag--muted">
                    {col.exclude_reason}
                  </span>
                )}
              </td>
              <td class="lzw-table__num">{col.unique_count}</td>
              <td>
                <input
                  type="checkbox"
                  checked={col.excluded}
                  onChange={(e) =>
                    onUpdate(
                      col.name,
                      (e.target as HTMLInputElement).checked,
                      col.col_type,
                    )
                  }
                />
              </td>
              <td>
                <select
                  value={col.col_type}
                  disabled={col.excluded}
                  onChange={(e) =>
                    onUpdate(
                      col.name,
                      col.excluded,
                      (e.target as HTMLSelectElement).value,
                    )
                  }
                >
                  <option value="numeric">Numeric</option>
                  <option value="categorical">Categorical</option>
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
