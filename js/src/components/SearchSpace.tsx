/**
 * SearchSpace — per-parameter mode selection for tuning (Fixed/Range/Choice).
 *
 * P-014: Renders grouped rows from search_space_catalog (Model Params / Smart Params / Training).
 * Fixed values for model_params go to tuning.model_params, for training to tuning.training.
 * Range/Choice entries are stored in tuning.optuna.space.
 * All backend-specific constants are read from uiSchema (provided by backend contract).
 */
import { useState } from "preact/hooks";
import { NumericStepper } from "./NumericStepper";

/** Atomic update payload — all three stores in one call to avoid stale-closure races. */
interface SearchSpaceUpdate {
  space: Record<string, any>;
  fixedModelParams: Record<string, any>;
  fixedTraining: Record<string, any>;
}

interface SearchSpaceProps {
  schema: Record<string, any>;
  rootSchema?: Record<string, any>;
  /** tuning.optuna.space — Range/Choice entries */
  spaceValue: Record<string, any>;
  /** tuning.model_params — Fixed model param values */
  fixedModelParams: Record<string, any>;
  /** tuning.training — Fixed training values */
  fixedTraining: Record<string, any>;
  /** Fit model config (for initial values) */
  modelConfig?: Record<string, any>;
  /** Fit training config (for initial values) */
  trainingConfig?: Record<string, any>;
  task?: string;
  uiSchema?: Record<string, any>;
  /** Single atomic callback for all search space updates. */
  onChange: (update: SearchSpaceUpdate) => void;
}

type Mode = "fixed" | "range" | "choice";

interface ParamConfig {
  mode: Mode;
  fixed?: any;
  low?: number;
  high?: number;
  log?: boolean;
  choices?: any[];
}

/** Reverse-map a stored LizyML space entry back to UI ParamConfig. */
function toParamConfig(stored: Record<string, any>): ParamConfig {
  if (stored.type === "float" || stored.type === "int") {
    return { mode: "range", low: stored.low, high: stored.high, log: stored.log ?? false };
  }
  if (stored.type === "categorical") {
    return { mode: "choice", choices: stored.choices ?? [] };
  }
  if (stored.mode) return stored as ParamConfig;
  return { mode: "fixed" };
}

interface CatalogEntry {
  key: string;
  title: string;
  paramType: string;
  modes: Mode[];
  group: string;
  default?: any;
}

function isNumeric(type: string): boolean {
  return type === "number" || type === "integer";
}

/** Render the Fixed-mode cell. */
function renderFixed(
  fieldSchema: Record<string, any>,
  config: ParamConfig,
  onChange: (c: ParamConfig) => void,
  name?: string,
  stepMap?: Record<string, number>,
) {
  const sm = stepMap ?? {};

  if (fieldSchema.type === "boolean") {
    return (
      <label class="lzw-toggle">
        <input
          type="checkbox"
          checked={config.fixed ?? false}
          onChange={(e) =>
            onChange({ ...config, fixed: (e.target as HTMLInputElement).checked })
          }
        />
        <span class="lzw-toggle__slider" />
      </label>
    );
  }

  if (fieldSchema.enum) {
    if (fieldSchema.segment) {
      return (
        <div class="lzw-segment">
          {(fieldSchema.enum as string[]).map((opt) => (
            <button
              key={opt}
              type="button"
              class={`lzw-segment__btn ${config.fixed === opt ? "lzw-segment__btn--active" : ""}`}
              aria-pressed={config.fixed === opt}
              onClick={() => onChange({ ...config, fixed: opt })}
            >
              {opt}
            </button>
          ))}
        </div>
      );
    }
    return (
      <select
        class="lzw-select"
        value={config.fixed ?? ""}
        onChange={(e) =>
          onChange({ ...config, fixed: (e.target as HTMLSelectElement).value })
        }
      >
        {(fieldSchema.enum as string[]).map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    );
  }

  if (fieldSchema.type === "array" && fieldSchema.items?.enum) {
    const selected: string[] = Array.isArray(config.fixed) ? config.fixed : [];
    return (
      <div class="lzw-chip-group">
        {(fieldSchema.items.enum as string[]).map((opt) => (
          <button
            key={opt}
            type="button"
            class={`lzw-chip ${selected.includes(opt) ? "lzw-chip--active" : ""}`}
            onClick={() => {
              const next = selected.includes(opt)
                ? selected.filter((v) => v !== opt)
                : [...selected, opt];
              onChange({ ...config, fixed: next });
            }}
          >
            {opt}
          </button>
        ))}
      </div>
    );
  }

  if (isNumeric(fieldSchema.type ?? "")) {
    return (
      <NumericStepper
        value={config.fixed}
        step={sm[name ?? ""] ?? (fieldSchema.type === "integer" ? 1 : "any")}
        onChange={(v) => onChange({ ...config, fixed: v })}
      />
    );
  }

  // Object type (e.g. feature_weights): ON/OFF toggle
  if (fieldSchema.type === "object") {
    return (
      <label class="lzw-toggle">
        <input
          type="checkbox"
          checked={config.fixed != null}
          onChange={(e) =>
            onChange({ ...config, fixed: (e.target as HTMLInputElement).checked ? {} : null })
          }
        />
        <span class="lzw-toggle__slider" />
      </label>
    );
  }

  // Fallback: select for inner_valid etc.
  if (fieldSchema.options) {
    const opts = fieldSchema.options as string[];
    return (
      <select
        class="lzw-select"
        value={config.fixed ?? opts[0] ?? ""}
        onChange={(e) => onChange({ ...config, fixed: (e.target as HTMLSelectElement).value })}
      >
        {opts.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    );
  }

  return (
    <input
      class="lzw-input"
      type="text"
      value={config.fixed ?? ""}
      onChange={(e) =>
        onChange({ ...config, fixed: (e.target as HTMLInputElement).value })
      }
    />
  );
}

/** Check conditional visibility for a catalog entry.
 * Rules map dep_key → required_value. If dep is in space (Choice/Range), always visible.
 * Otherwise compare Fixed value. */
function isParamVisible(
  key: string,
  conditionalVisibility: Record<string, any>,
  spaceValue: Record<string, any>,
  getFixed: (k: string) => any,
): boolean {
  const rule = conditionalVisibility[key];
  if (!rule) return true;
  for (const [depKey, required] of Object.entries(rule)) {
    if (depKey in spaceValue) return true; // dep in Choice/Range → always show
    const current = getFixed(depKey);
    if (current !== required) return false;
  }
  return true;
}

const GROUP_TITLES: Record<string, string> = {
  model_params: "Model Params",
  smart_params: "Smart Params",
  training: "Training",
};

export function SearchSpace({
  spaceValue,
  fixedModelParams,
  fixedTraining,
  modelConfig,
  trainingConfig,
  task,
  uiSchema,
  onChange,
}: SearchSpaceProps) {
  const [addedParams, setAddedParams] = useState<string[]>([]);
  const optionSets: Record<string, Record<string, string[]>> = uiSchema?.option_sets ?? {};
  const stepMap: Record<string, number> = uiSchema?.step_map ?? {};
  const searchSpaceCatalog: CatalogEntry[] = uiSchema?.search_space_catalog ?? [];
  const innerValidOpts: string[] = uiSchema?.inner_valid_options ?? [];
  const conditionalVisibility: Record<string, any> = uiSchema?.conditional_visibility ?? {};
  const additionalParamsList: string[] = uiSchema?.additional_params ?? [];

  const fitParams = modelConfig?.params ?? {};
  const fitTraining = trainingConfig ?? {};

  /** Get current Fixed value for a catalog entry.
   * Priority: tuning fixed storage → fit config → catalog default */
  const getFixedValue = (entry: CatalogEntry): any => {
    if (entry.group === "training") {
      const parts = entry.key.split(".");
      if (parts.length === 1) {
        const v = fixedTraining[entry.key] ?? fitTraining[entry.key];
        return v !== undefined ? v : entry.default;
      }
      // Dot-path: early_stopping.enabled → fixedTraining.early_stopping.enabled
      let obj: any = fixedTraining;
      for (const p of parts) obj = obj?.[p];
      if (obj !== undefined) return obj;
      // Fallback to fit training
      obj = fitTraining;
      for (const p of parts) obj = obj?.[p];
      return obj !== undefined ? obj : entry.default;
    }
    // model_params / smart_params → from fixedModelParams or fit or default
    if (entry.key in fixedModelParams) return fixedModelParams[entry.key];
    const v = fitParams[entry.key] ?? modelConfig?.[entry.key];
    return v !== undefined ? v : entry.default;
  };

  /** Get a Fixed value by key (for conditional visibility checks). */
  const getFixedByKey = (key: string): any => {
    const entry = searchSpaceCatalog.find((e) => e.key === key);
    if (!entry) return undefined;
    return getFixedValue(entry);
  };

  /** Update handler: builds a single atomic update to avoid stale-closure races. */
  const handleUpdate = (entry: CatalogEntry, config: ParamConfig) => {
    let newSpace: Record<string, any>;
    let newFixedMP = fixedModelParams;
    let newFixedTr = fixedTraining;

    if (config.mode === "fixed") {
      // Remove from space
      const { [entry.key]: _removed, ...rest } = spaceValue;
      newSpace = rest;

      // Store in appropriate fixed storage
      if (entry.group === "training") {
        const parts = entry.key.split(".");
        if (parts.length === 1) {
          newFixedTr = { ...fixedTraining, [entry.key]: config.fixed };
        } else if (parts.length === 2) {
          const [parent, child] = parts;
          newFixedTr = {
            ...fixedTraining,
            [parent]: { ...(fixedTraining[parent] ?? {}), [child]: config.fixed },
          };
        }
      } else {
        newFixedMP = { ...fixedModelParams, [entry.key]: config.fixed };
      }
    } else {
      // Range or Choice → store in space
      newSpace = { ...spaceValue };
      if (config.mode === "range") {
        const spaceType = entry.paramType === "integer" ? "int" : "float";
        newSpace[entry.key] = { type: spaceType, low: config.low, high: config.high, log: config.log ?? false };
      } else {
        newSpace[entry.key] = { type: "categorical", choices: config.choices ?? [] };
      }
    }

    onChange({ space: newSpace, fixedModelParams: newFixedMP, fixedTraining: newFixedTr });
  };

  /** Build ParamConfig for a catalog entry. */
  const getParamConfig = (entry: CatalogEntry): ParamConfig => {
    if (entry.key in spaceValue) return toParamConfig(spaceValue[entry.key]);
    return { mode: "fixed", fixed: getFixedValue(entry) };
  };

  /** Build fake schema for rendering a catalog entry's Fixed control. */
  const specialFields: Record<string, string> = uiSchema?.special_search_space_fields ?? {};
  const buildFieldSchema = (entry: CatalogEntry): Record<string, any> => {
    const fieldKind = specialFields[entry.key];
    if (fieldKind === "objective") {
      return { title: entry.title, type: "string", enum: optionSets.objective?.[task ?? ""] ?? [], segment: true };
    }
    if (fieldKind === "model_metric") {
      return { title: entry.title, type: "array", items: { enum: optionSets.model_metric?.[task ?? ""] ?? [] } };
    }
    if (fieldKind === "inner_valid_picker") {
      return { title: entry.title, type: "string", options: innerValidOpts };
    }
    // Fallback: match by key name for backward compatibility when contract lacks special_search_space_fields
    if (!fieldKind) {
      if (entry.key === "objective") {
        return { title: entry.title, type: "string", enum: optionSets.objective?.[task ?? ""] ?? [], segment: true };
      }
      if (entry.key === "metric") {
        return { title: entry.title, type: "array", items: { enum: optionSets.model_metric?.[task ?? ""] ?? [] } };
      }
      if (entry.key === "inner_valid") {
        return { title: entry.title, type: "string", options: innerValidOpts };
      }
    }
    return { title: entry.title, type: entry.paramType };
  };

  // Group catalog entries
  const groups = new Map<string, CatalogEntry[]>();
  for (const entry of searchSpaceCatalog) {
    const g = entry.group ?? "model_params";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g)!.push(entry);
  }

  return (
    <div class="lzw-search-space">
      <div class="lzw-search-space-grid" role="table">
        <div class="lzw-search-space-grid__header" role="row">
          <div role="columnheader">Parameter</div>
          <div role="columnheader">Mode</div>
          <div role="columnheader">Configuration</div>
        </div>

        {/* Catalog groups */}
        {Array.from(groups.entries()).map(([groupKey, entries]) => (
          <div key={groupKey} class="lzw-search-space-grid__group-wrap">
            <div class="lzw-search-space-grid__group" role="row">
              <div role="cell">
                {GROUP_TITLES[groupKey] ?? groupKey}
              </div>
            </div>
            {entries.map((entry) => {
              if (!isParamVisible(entry.key, conditionalVisibility, spaceValue, getFixedByKey)) {
                return null;
              }
              const paramConfig = getParamConfig(entry);
              const fieldSchema = buildFieldSchema(entry);
              return (
                <SearchSpaceRow
                  key={entry.key}
                  name={entry.key}
                  fieldSchema={fieldSchema}
                  config={paramConfig}
                  modes={entry.modes}
                  onChange={(c) => handleUpdate(entry, c)}
                  stepMap={stepMap}
                />
              );
            })}
          </div>
        ))}

        {/* Added params (from [+ Add]) */}
        {addedParams.map((paramKey) => {
          const addedEntry: CatalogEntry = {
            key: paramKey,
            title: paramKey,
            paramType: "number",
            modes: ["fixed", "range"],
            group: "model_params",
          };
          const paramConfig = getParamConfig(addedEntry);
          const fieldSchema = { title: paramKey, type: "number" };
          return (
            <div key={`added-${paramKey}`} class="lzw-search-space-grid__row" role="row">
              <div class="lzw-table__name" role="cell">
                {paramKey}
                <button
                  type="button"
                  class="lzw-btn lzw-btn--icon"
                  style="margin-left: 4px; font-size: 10px; padding: 0 4px;"
                  onClick={() => {
                    setAddedParams(addedParams.filter((k) => k !== paramKey));
                    // Remove from space and fixedModelParams atomically
                    const { [paramKey]: _s, ...newSpace } = spaceValue;
                    const { [paramKey]: _, ...restFixed } = fixedModelParams;
                    onChange({ space: newSpace, fixedModelParams: restFixed, fixedTraining });
                  }}
                >
                  ×
                </button>
              </div>
              <div role="cell">
                <div class="lzw-segment">
                  {(["fixed", "range"] as Mode[]).map((m) => (
                    <button
                      key={m}
                      type="button"
                      class={`lzw-segment__btn ${paramConfig.mode === m ? "lzw-segment__btn--active" : ""}`}
                      aria-pressed={paramConfig.mode === m}
                      onClick={() => {
                        if (m === paramConfig.mode) return;
                        if (m === "fixed") {
                          handleUpdate(addedEntry, { mode: m, fixed: 0 });
                        } else {
                          handleUpdate(addedEntry, { mode: m, low: 0, high: 1, log: false });
                        }
                      }}
                    >
                      {m.charAt(0).toUpperCase() + m.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
              <div role="cell">
                {paramConfig.mode === "fixed" && renderFixed(fieldSchema, paramConfig, (c) => handleUpdate(addedEntry, c), paramKey, stepMap)}
                {paramConfig.mode === "range" && (
                  <div class="lzw-search-space__range">
                    <NumericStepper
                      value={paramConfig.low ?? undefined}
                      placeholder="low"
                      onChange={(v) => handleUpdate(addedEntry, { ...paramConfig, low: v })}
                    />
                    <span>~</span>
                    <NumericStepper
                      value={paramConfig.high ?? undefined}
                      placeholder="high"
                      onChange={(v) => handleUpdate(addedEntry, { ...paramConfig, high: v })}
                    />
                    <label class="lzw-search-space__log">
                      <input
                        type="checkbox"
                        checked={paramConfig.log ?? false}
                        onChange={(e) =>
                          handleUpdate(addedEntry, { ...paramConfig, log: (e.target as HTMLInputElement).checked })
                        }
                      />
                      Log
                    </label>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* [+ Add] button */}
        {(() => {
          const catalogKeys = new Set(searchSpaceCatalog.map((e) => e.key));
          const usedKeys = new Set(addedParams);
          const available = additionalParamsList.filter(
            (p) => !catalogKeys.has(p) && !usedKeys.has(p),
          );
          if (available.length === 0) return null;
          return (
            <div class="lzw-search-space-grid__row" role="row">
              <div role="cell" style="padding: 6px 8px;">
                <select
                  class="lzw-select"
                  value=""
                  onChange={(e) => {
                    const v = (e.target as HTMLSelectElement).value;
                    if (v) {
                      setAddedParams([...addedParams, v]);
                      // Initialize with Fixed mode, value 0 — atomic update
                      onChange({ space: spaceValue, fixedModelParams: { ...fixedModelParams, [v]: 0 }, fixedTraining });
                    }
                  }}
                >
                  <option value="">+ Add</option>
                  {available.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
              <div role="cell" />
              <div role="cell" />
            </div>
          );
        })()}

      </div>
    </div>
  );
}

interface SearchSpaceRowProps {
  name: string;
  fieldSchema: Record<string, any>;
  config: ParamConfig;
  modes: Mode[];
  onChange: (config: ParamConfig) => void;
  stepMap?: Record<string, number>;
}

function SearchSpaceRow({
  name,
  fieldSchema,
  config,
  modes,
  onChange,
  stepMap,
}: SearchSpaceRowProps) {
  const title = fieldSchema.title ?? name;
  const sm = stepMap ?? {};

  return (
    <div class="lzw-search-space-grid__row" role="row">
      <div class="lzw-table__name" role="cell">{title}</div>
      <div role="cell">
        {modes.length > 1 ? (
          <div class="lzw-segment">
            {modes.map((m) => (
              <button
                key={m}
                type="button"
                class={`lzw-segment__btn ${config.mode === m ? "lzw-segment__btn--active" : ""}`}
                aria-pressed={config.mode === m}
                onClick={() => {
                  if (m === config.mode) return;
                  if (m === "fixed") {
                    // Recover value from previous mode or use schema default
                    const recovered = config.low ?? config.choices?.[0] ?? fieldSchema.default ?? 0;
                    onChange({ mode: m, fixed: recovered });
                  } else if (m === "range") {
                    const base = config.fixed ?? fieldSchema.default ?? 0;
                    onChange({ mode: m, low: base, high: typeof base === "number" ? base * 2 || 1 : 1, log: false });
                  } else {
                    onChange({ mode: m, choices: fieldSchema.enum ?? [config.fixed ?? fieldSchema.default] });
                  }
                }}
              >
                {m.charAt(0).toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>
        ) : (
          <span class="lzw-tag lzw-tag--muted">Fixed</span>
        )}
      </div>
      <div role="cell">
        {config.mode === "fixed" && renderFixed(fieldSchema, config, onChange, name, stepMap)}
        {config.mode === "range" && (() => {
          const stepVal = sm[name] ?? (fieldSchema.type === "integer" ? 1 : "any");
          return (
          <div class="lzw-search-space__range">
            <NumericStepper
              value={config.low ?? undefined}
              step={typeof stepVal === "number" ? stepVal : undefined}
              placeholder="low"
              onChange={(v) => onChange({ ...config, low: v })}
            />
            <span>~</span>
            <NumericStepper
              value={config.high ?? undefined}
              step={typeof stepVal === "number" ? stepVal : undefined}
              placeholder="high"
              onChange={(v) => onChange({ ...config, high: v })}
            />
            <label class="lzw-search-space__log">
              <input
                type="checkbox"
                checked={config.log ?? false}
                onChange={(e) =>
                  onChange({ ...config, log: (e.target as HTMLInputElement).checked })
                }
              />
              Log
            </label>
          </div>
          );
        })()}
        {config.mode === "choice" && (() => {
          const opts: string[] = fieldSchema.enum ?? fieldSchema.items?.enum ?? [];
          if (opts.length === 0) {
            return <span class="lzw-muted">{(config.choices ?? []).join(", ")}</span>;
          }
          return (
            <div class="lzw-chip-group">
              {opts.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  class={`lzw-chip ${(config.choices ?? []).includes(opt) ? "lzw-chip--active" : ""}`}
                  onClick={() => {
                    const choices = config.choices ?? [];
                    onChange({
                      ...config,
                      choices: choices.includes(opt)
                        ? choices.filter((v) => v !== opt)
                        : [...choices, opt],
                    });
                  }}
                >
                  {opt}
                </button>
              ))}
            </div>
          );
        })()}
      </div>
    </div>
  );
}
