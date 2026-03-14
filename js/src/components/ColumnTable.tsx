/** ColumnTable — column settings grid for the Data Tab. */

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
      <div class="lzw-columns-grid" role="table">
        <div class="lzw-columns-grid__header" role="row">
          <div role="columnheader">Column</div>
          <div role="columnheader">Uniq</div>
          <div role="columnheader">Excl</div>
          <div role="columnheader">Type</div>
        </div>
        {columns.map((col) => (
          <div
            key={col.name}
            class={`lzw-columns-grid__row ${col.excluded ? "lzw-row--excluded" : ""}`}
            role="row"
          >
            <div class="lzw-table__name" role="cell">
              {col.name}
            </div>
            <div class="lzw-table__num lzw-table__num--uniq" role="cell">{col.unique_count}</div>
            <div class="lzw-table__cell--excl" role="cell">
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
            </div>
            <div role="cell">
              {col.excluded ? (
                <span class="lzw-muted">
                  ── {col.exclude_reason === "id" ? "[ID]" : col.exclude_reason === "constant" ? "[Const]" : "[Manual]"}
                </span>
              ) : (
                <div class="lzw-segment">
                  {["numeric", "categorical"].map((t) => (
                    <button
                      key={t}
                      type="button"
                      class={`lzw-segment__btn ${col.col_type === t ? "lzw-segment__btn--active" : ""}`}
                      aria-pressed={col.col_type === t}
                      onClick={() => onUpdate(col.name, col.excluded, t)}
                    >
                      {t.charAt(0).toUpperCase() + t.slice(1)}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
