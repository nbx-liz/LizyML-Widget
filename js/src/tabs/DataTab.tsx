/** DataTab — Target/Task selection, Column settings, CV settings, Feature summary. */
import { Accordion } from "../components/Accordion";
import { ColumnTable } from "../components/ColumnTable";

interface DfInfo {
  shape?: [number, number];
  target?: string | null;
  task?: string | null;
  columns?: any[];
  cv?: { strategy: string; n_splits: number; group_column?: string | null };
  feature_summary?: Record<string, number>;
}

interface DataTabProps {
  dfInfo: DfInfo;
  allColumns: string[];
  sendAction: (type: string, payload?: Record<string, any>) => void;
}

const CV_STRATEGIES = [
  { value: "kfold", label: "KFold" },
  { value: "stratified_kfold", label: "StratifiedKFold" },
  { value: "group_kfold", label: "GroupKFold" },
  { value: "time_series_split", label: "TimeSeriesSplit" },
];

export function DataTab({ dfInfo, allColumns, sendAction }: DataTabProps) {
  const shape = dfInfo.shape ?? [0, 0];
  const columns = dfInfo.columns ?? [];
  const cv = dfInfo.cv ?? { strategy: "kfold", n_splits: 5 };
  const fs = dfInfo.feature_summary;
  const featureCols = columns.filter((c: any) => !c.excluded).map((c: any) => c.name);

  return (
    <div class="lzw-data-tab">
      <div class="lzw-data-tab__shape">
        DataFrame: {shape[0]} rows &times; {shape[1]} cols
      </div>

      {/* Target / Task */}
      <Accordion title="Target / Task">
        <div class="lzw-form-row">
          <label class="lzw-label">Target</label>
          <select
            class="lzw-select"
            value={dfInfo.target ?? ""}
            onChange={(e) => {
              const v = (e.target as HTMLSelectElement).value;
              if (v) sendAction("set_target", { target: v });
            }}
          >
            <option value="">-- Select --</option>
            {allColumns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        {dfInfo.task && (
          <div class="lzw-form-row">
            <label class="lzw-label">Task</label>
            <span class="lzw-badge lzw-badge--info">{dfInfo.task}</span>
            <span class="lzw-muted" style="margin-left:4px">
              auto
            </span>
          </div>
        )}
      </Accordion>

      {/* Column Settings */}
      <Accordion title="Column Settings" defaultOpen={columns.length <= 30}>
        <ColumnTable
          columns={columns}
          onUpdate={(name, excluded, colType) =>
            sendAction("update_column", {
              name,
              excluded,
              col_type: colType,
            })
          }
        />
      </Accordion>

      {/* Cross Validation */}
      <Accordion title="Cross Validation">
        <div class="lzw-form-row">
          <label class="lzw-label">Strategy</label>
          <select
            class="lzw-select"
            value={cv.strategy}
            onChange={(e) =>
              sendAction("update_cv", {
                strategy: (e.target as HTMLSelectElement).value,
                n_splits: cv.n_splits,
                group_column: cv.group_column ?? null,
              })
            }
          >
            {CV_STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </div>
        <div class="lzw-form-row">
          <label class="lzw-label">Folds</label>
          <input
            class="lzw-input lzw-input--sm"
            type="number"
            min={2}
            max={50}
            value={cv.n_splits}
            onChange={(e) =>
              sendAction("update_cv", {
                strategy: cv.strategy,
                n_splits: parseInt((e.target as HTMLInputElement).value) || 5,
                group_column: cv.group_column ?? null,
              })
            }
          />
        </div>
        {cv.strategy === "group_kfold" && (
          <div class="lzw-form-row">
            <label class="lzw-label">Group column</label>
            <select
              class="lzw-select"
              value={cv.group_column ?? ""}
              onChange={(e) =>
                sendAction("update_cv", {
                  strategy: cv.strategy,
                  n_splits: cv.n_splits,
                  group_column:
                    (e.target as HTMLSelectElement).value || null,
                })
              }
            >
              <option value="">-- Select --</option>
              {featureCols.map((c: string) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        )}
      </Accordion>

      {/* Feature Summary */}
      {fs && (
        <div class="lzw-feature-summary">
          <span>
            Features: {fs.total} cols (Numeric: {fs.numeric}, Categ.:{" "}
            {fs.categorical})
          </span>
          <span>
            Excluded: {fs.excluded} (ID: {fs.excluded_id}, Const:{" "}
            {fs.excluded_const}, Manual: {fs.excluded_manual})
          </span>
        </div>
      )}
    </div>
  );
}
