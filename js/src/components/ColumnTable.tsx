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
      <table class="lzw-table lzw-table--columns">
        <colgroup>
          <col class="lzw-col--name" />
          <col class="lzw-col--uniq" />
          <col class="lzw-col--exclude" />
          <col class="lzw-col--type" />
        </colgroup>
        <thead>
          <tr>
            <th>Column</th>
            <th>Uniq</th>
            <th>Excl</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((col) => (
            <tr key={col.name} class={col.excluded ? "lzw-row--excluded" : ""}>
              <td class="lzw-table__name">
                {col.name}
              </td>
              <td class="lzw-table__num lzw-table__num--uniq">{col.unique_count}</td>
              <td class="lzw-table__cell--excl">
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
                {col.excluded ? (
                  <span class="lzw-muted">
                    ── {col.exclude_reason === "id" ? "[ID]" : col.exclude_reason === "constant" ? "[Const]" : "[Manual]"}
                  </span>
                ) : (
                  <select
                    class="lzw-select lzw-table__type-select"
                    value={col.col_type}
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
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
