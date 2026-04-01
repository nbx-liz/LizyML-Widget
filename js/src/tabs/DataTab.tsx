/** DataTab — Target/Task selection, Column settings, CV settings, Feature summary. */
import { Accordion } from "../components/Accordion";
import { BlockedGroupKFold } from "../components/BlockedGroupKFold";
import { ColumnTable } from "../components/ColumnTable";
import { NumericStepper } from "../components/NumericStepper";

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
  blocks?: any;
  groups?: any;
  min_train_rows?: number;
  min_valid_rows?: number;
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
  columnStats: Record<string, any> | null;
  splitPreview: any | null;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  backendContract?: Record<string, any> | null;
}

const CV_STRATEGIES = [
  { value: "kfold", label: "KFold" },
  { value: "stratified_kfold", label: "StratifiedKFold" },
  { value: "group_kfold", label: "GroupKFold" },
  { value: "stratified_group_kfold", label: "StratifiedGroup" },
  { value: "time_series", label: "TimeSeriesSplit" },
  { value: "purged_time_series", label: "PurgedTimeSeriesSplit" },
  { value: "group_time_series", label: "GroupTimeSeriesSplit" },
  { value: "blocked_group_kfold", label: "BlockedGroup" },
];

/* Fallback Sets used when backend_contract does not provide cv_strategy_fields */
const FALLBACK_NEEDS_GROUP = new Set(["group_kfold", "stratified_group_kfold", "group_time_series"]);
const FALLBACK_NEEDS_TIME = new Set(["time_series", "purged_time_series", "group_time_series"]);
const FALLBACK_NEEDS_RANDOM_STATE = new Set(["kfold", "stratified_kfold", "stratified_group_kfold"]);
const FALLBACK_NEEDS_GAP = new Set(["time_series", "group_time_series"]);
const FALLBACK_NEEDS_PURGE = new Set(["purged_time_series"]);
const FALLBACK_IS_TIME_SERIES = new Set(["time_series", "purged_time_series", "group_time_series"]);

/** Derive a Set of strategy names whose fields include any of the given field names. */
function deriveStrategiesWithField(
  cvStrategyFields: Record<string, string[]>,
  fieldNames: string[],
): Set<string> {
  return new Set(
    Object.entries(cvStrategyFields)
      .filter(([, fields]) => fieldNames.some((f) => fields.includes(f)))
      .map(([strategy]) => strategy),
  );
}

export function DataTab({ dfInfo, allColumns, columnStats, splitPreview, sendAction, backendContract }: DataTabProps) {
  // Read CV strategy fields from backend contract, derive Sets with fallbacks
  const cvStrategyFields: Record<string, string[]> =
    backendContract?.capabilities?.cv_strategy_fields ?? {};
  const hasContractFields = Object.keys(cvStrategyFields).length > 0;

  const NEEDS_GROUP = hasContractFields
    ? deriveStrategiesWithField(cvStrategyFields, ["group_col", "groups_col"])
    : FALLBACK_NEEDS_GROUP;
  const NEEDS_TIME = hasContractFields
    ? deriveStrategiesWithField(cvStrategyFields, ["time_col"])
    : FALLBACK_NEEDS_TIME;
  const NEEDS_RANDOM_STATE = hasContractFields
    ? deriveStrategiesWithField(cvStrategyFields, ["random_state"])
    : FALLBACK_NEEDS_RANDOM_STATE;
  const NEEDS_GAP = hasContractFields
    ? deriveStrategiesWithField(cvStrategyFields, ["gap"])
    : FALLBACK_NEEDS_GAP;
  const NEEDS_PURGE = hasContractFields
    ? deriveStrategiesWithField(cvStrategyFields, ["purge_gap"])
    : FALLBACK_NEEDS_PURGE;
  const IS_TIME_SERIES = hasContractFields
    ? deriveStrategiesWithField(cvStrategyFields, ["max_train_size", "max_test_size"])
    : FALLBACK_IS_TIME_SERIES;
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
          <div class="lzw-segment">
            {["binary", "multiclass", "regression"].map((t) => (
              <button
                key={t}
                type="button"
                class={`lzw-segment__btn ${dfInfo.task === t ? "lzw-segment__btn--active" : ""}`}
                disabled={!hasTarget}
                aria-pressed={dfInfo.task === t}
                onClick={() => { if (hasTarget) sendAction("set_task", { task: t }); }}
              >
                {dfInfo.task === t && dfInfo.task === dfInfo.auto_task ? `⚡${t}` : t}
              </button>
            ))}
          </div>
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
        <div class="lzw-form-row" style="align-items:flex-start">
          <label class="lzw-label">Strategy</label>
          <div class="lzw-chip-group">
            {CV_STRATEGIES.map((s) => (
              <button
                key={s.value}
                type="button"
                class={`lzw-chip lzw-chip--square ${cv.strategy === s.value ? "lzw-chip--active" : ""}`}
                aria-pressed={cv.strategy === s.value}
                onClick={() => sendCv({ ...cv, strategy: s.value })}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
        {cv.strategy === "blocked_group_kfold" ? (
          <BlockedGroupKFold
            cv={cv}
            allColumns={featureCols}
            columnStats={columnStats}
            splitPreview={splitPreview}
            sendAction={sendAction}
            sendCv={sendCv}
          />
        ) : (
        <>
        <div class="lzw-form-row">
          <label class="lzw-label">Folds</label>
          <NumericStepper
            value={cv.n_splits}
            min={2}
            max={50}
            step={1}
            onChange={(v) => sendCv({ ...cv, n_splits: v ?? 5 })}
          />
        </div>
        {NEEDS_RANDOM_STATE.has(cv.strategy) && (
          <div class="lzw-form-row">
            <label class="lzw-label">Random state</label>
            <NumericStepper
              value={cv.random_state ?? 42}
              step={1}
              onChange={(v) => sendCv({ ...cv, random_state: v ?? 42 })}
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
            <NumericStepper
              value={cv.gap ?? 0}
              min={0}
              step={1}
              onChange={(v) => sendCv({ ...cv, gap: v ?? 0 })}
            />
          </div>
        )}
        {NEEDS_PURGE.has(cv.strategy) && (
          <>
            <div class="lzw-form-row">
              <label class="lzw-label">Purge gap</label>
              <NumericStepper
                value={cv.purge_gap ?? 0}
                min={0}
                step={1}
                onChange={(v) => sendCv({ ...cv, purge_gap: v ?? 0 })}
              />
            </div>
            <div class="lzw-form-row">
              <label class="lzw-label">Embargo</label>
              <NumericStepper
                value={cv.embargo ?? 0}
                min={0}
                step={1}
                onChange={(v) => sendCv({ ...cv, embargo: v ?? 0 })}
              />
            </div>
          </>
        )}
        {IS_TIME_SERIES.has(cv.strategy) && (
          <>
            <div class="lzw-form-row">
              <label class="lzw-label">Train size max</label>
              <NumericStepper
                value={cv.train_size_max ?? undefined}
                min={0}
                step={1}
                placeholder="null"
                onChange={(v) => sendCv({ ...cv, train_size_max: v ?? null })}
              />
            </div>
            <div class="lzw-form-row">
              <label class="lzw-label">Test size max</label>
              <NumericStepper
                value={cv.test_size_max ?? undefined}
                min={0}
                step={1}
                placeholder="null"
                onChange={(v) => sendCv({ ...cv, test_size_max: v ?? null })}
              />
            </div>
          </>
        )}
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
