/** DataTab — Target/Task selection, Column settings, CV settings, Feature summary. */
import { Accordion } from "../components/Accordion";
import { ColumnTable } from "../components/ColumnTable";

interface CvInfo {
  strategy: string;
  n_splits: number;
  group_column?: string | null;
  time_column?: string | null;
  random_state?: number | null;
  shuffle?: boolean | null;
  gap?: number;
  purge_gap?: number;
  embargo?: number;
  train_size_max?: number | null;
  test_size_max?: number | null;
}

interface DfInfo {
  shape?: [number, number];
  target?: string | null;
  task?: string | null;
  auto_task?: string | null;
  columns?: any[];
  cv?: CvInfo;
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
  { value: "time_series", label: "TimeSeriesSplit" },
  { value: "purged_time_series", label: "PurgedTimeSeriesSplit" },
  { value: "group_time_series", label: "GroupTimeSeriesSplit" },
];

const NEEDS_GROUP = new Set(["group_kfold", "group_time_series"]);
const NEEDS_TIME = new Set(["time_series", "purged_time_series", "group_time_series"]);
const NEEDS_RANDOM_STATE = new Set(["kfold", "stratified_kfold"]);
const NEEDS_GAP = new Set(["time_series", "group_time_series"]);
const NEEDS_PURGE = new Set(["purged_time_series"]);
const IS_TIME_SERIES = new Set(["time_series", "purged_time_series", "group_time_series"]);

export function DataTab({ dfInfo, allColumns, sendAction }: DataTabProps) {
  const shape = dfInfo.shape ?? [0, 0];
  const columns = dfInfo.columns ?? [];
  const cv = dfInfo.cv ?? { strategy: "kfold", n_splits: 5 };
  const fs = dfInfo.feature_summary;
  const featureCols = columns.filter((c: any) => !c.excluded).map((c: any) => c.name);
  const hasTarget = Boolean(dfInfo.target);

  const sendCv = (updated: CvInfo) => sendAction("update_cv", updated);

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
        <div class="lzw-form-row">
          <label class="lzw-label">Task</label>
          <select
            class="lzw-select"
            value={dfInfo.task ?? ""}
            disabled={!hasTarget}
            onChange={(e) => {
              const task = (e.target as HTMLSelectElement).value;
              if (hasTarget && task) {
                sendAction("set_task", { task });
              }
            }}
          >
            {!hasTarget && <option value="">-- Select target first --</option>}
            {["binary", "multiclass", "regression"].map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          {hasTarget && dfInfo.task === dfInfo.auto_task && (
            <span class="lzw-muted" style="margin-left:4px">
              ⚡auto
            </span>
          )}
        </div>
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
              sendCv({ ...cv, strategy: (e.target as HTMLSelectElement).value })
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
              sendCv({ ...cv, n_splits: parseInt((e.target as HTMLInputElement).value) || 5 })
            }
          />
        </div>
        {NEEDS_RANDOM_STATE.has(cv.strategy) && (
          <div class="lzw-form-row">
            <label class="lzw-label">Random state</label>
            <input
              class="lzw-input lzw-input--sm"
              type="number"
              value={cv.random_state ?? 42}
              onChange={(e) =>
                sendCv({ ...cv, random_state: parseInt((e.target as HTMLInputElement).value) || 42 })
              }
            />
          </div>
        )}
        {cv.strategy === "kfold" && (
          <div class="lzw-form-row">
            <label class="lzw-label">Shuffle</label>
            <input
              type="checkbox"
              checked={cv.shuffle ?? true}
              onChange={(e) =>
                sendCv({ ...cv, shuffle: (e.target as HTMLInputElement).checked })
              }
            />
          </div>
        )}
        {NEEDS_GROUP.has(cv.strategy) && (
          <div class="lzw-form-row">
            <label class="lzw-label">Group column</label>
            <select
              class="lzw-select"
              value={cv.group_column ?? ""}
              onChange={(e) =>
                sendCv({ ...cv, group_column: (e.target as HTMLSelectElement).value || null })
              }
            >
              <option value="">-- Select --</option>
              {featureCols.map((c: string) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        )}
        {NEEDS_TIME.has(cv.strategy) && (
          <div class="lzw-form-row">
            <label class="lzw-label">Time column</label>
            <select
              class="lzw-select"
              value={cv.time_column ?? ""}
              onChange={(e) =>
                sendCv({ ...cv, time_column: (e.target as HTMLSelectElement).value || null })
              }
            >
              <option value="">-- Select --</option>
              {featureCols.map((c: string) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        )}
        {NEEDS_GAP.has(cv.strategy) && (
          <div class="lzw-form-row">
            <label class="lzw-label">Gap</label>
            <input
              class="lzw-input lzw-input--sm"
              type="number"
              min={0}
              value={cv.gap ?? 0}
              onChange={(e) =>
                sendCv({ ...cv, gap: parseInt((e.target as HTMLInputElement).value) || 0 })
              }
            />
          </div>
        )}
        {NEEDS_PURGE.has(cv.strategy) && (
          <>
            <div class="lzw-form-row">
              <label class="lzw-label">Purge gap</label>
              <input
                class="lzw-input lzw-input--sm"
                type="number"
                min={0}
                value={cv.purge_gap ?? 0}
                onChange={(e) =>
                  sendCv({ ...cv, purge_gap: parseInt((e.target as HTMLInputElement).value) || 0 })
                }
              />
            </div>
            <div class="lzw-form-row">
              <label class="lzw-label">Embargo</label>
              <input
                class="lzw-input lzw-input--sm"
                type="number"
                min={0}
                value={cv.embargo ?? 0}
                onChange={(e) =>
                  sendCv({ ...cv, embargo: parseInt((e.target as HTMLInputElement).value) || 0 })
                }
              />
            </div>
          </>
        )}
        {IS_TIME_SERIES.has(cv.strategy) && (
          <>
            <div class="lzw-form-row">
              <label class="lzw-label">Train size max</label>
              <input
                class="lzw-input lzw-input--sm"
                type="number"
                min={0}
                value={cv.train_size_max ?? ""}
                placeholder="null"
                onChange={(e) => {
                  const v = (e.target as HTMLInputElement).value;
                  sendCv({ ...cv, train_size_max: v ? parseInt(v) : null });
                }}
              />
            </div>
            <div class="lzw-form-row">
              <label class="lzw-label">Test size max</label>
              <input
                class="lzw-input lzw-input--sm"
                type="number"
                min={0}
                value={cv.test_size_max ?? ""}
                placeholder="null"
                onChange={(e) => {
                  const v = (e.target as HTMLInputElement).value;
                  sendCv({ ...cv, test_size_max: v ? parseInt(v) : null });
                }}
              />
            </div>
          </>
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
