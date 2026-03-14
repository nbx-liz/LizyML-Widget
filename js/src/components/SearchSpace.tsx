/**
 * SearchSpace — per-parameter mode selection for tuning (Fixed/Range/Choice).
 *
 * Renders combined rows: model.params (from search_space_catalog) + LGBMConfig schema fields + verbose.
 * Only Range/Choice entries are stored in tuning.optuna.space; Fixed = not in space dict.
 * All backend-specific constants are read from uiSchema (provided by backend contract).
 */
import { NumericStepper } from "./NumericStepper";

interface SearchSpaceProps {
  schema: Record<string, any>;
  rootSchema?: Record<string, any>;
  value: Record<string, any>;
  onChange: (value: Record<string, any>) => void;
  modelConfig?: Record<string, any>;
  task?: string;
  uiSchema?: Record<string, any>;
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

/** Resolve $ref in schema. */
function resolveRef(ref: string, root: Record<string, any>): Record<string, any> {
  const parts = ref.replace("#/", "").split("/");
  let node: any = root;
  for (const p of parts) node = node?.[p];
  return node ?? {};
}

function resolve(schema: Record<string, any>, root: Record<string, any>): Record<string, any> {
  if (schema.$ref) return resolveRef(schema.$ref, root);
  if (schema.allOf?.length === 1 && schema.allOf[0].$ref) {
    const resolved = resolveRef(schema.allOf[0].$ref, root);
    const { allOf: _, ...rest } = schema;
    return { ...resolved, ...rest };
  }
  return schema;
}

function isNumeric(resolved: Record<string, any>): boolean {
  return resolved.type === "number" || resolved.type === "integer";
}

function availableModes(resolved: Record<string, any>): Mode[] {
  if (isNumeric(resolved)) return ["fixed", "range"];
  if (resolved.enum || resolved.type === "boolean") return ["fixed", "choice"];
  return ["fixed"];
}

/** Reverse-map a stored LizyML space entry back to UI ParamConfig. */
function toParamConfig(stored: Record<string, any>): ParamConfig {
  if (stored.type === "float" || stored.type === "int") {
    return { mode: "range", low: stored.low, high: stored.high, log: stored.log ?? false };
  }
  if (stored.type === "categorical") {
    return { mode: "choice", choices: stored.choices ?? [] };
  }
  // Legacy format fallback (mode-based) — pass through as-is
  if (stored.mode) return stored as ParamConfig;
  return { mode: "fixed" };
}

/** Keys that are not tunable hyperparameters. */
const NON_TUNABLE = new Set(["name"]);

/** LGBMConfig schema fields managed separately in ModelSection (not shown as schema rows in SearchSpace). */
const SCHEMA_EXCLUDE = new Set(["name", "params", "auto_num_leaves", "num_leaves_ratio"]);

interface CatalogEntry {
  key: string;
  title: string;
  paramType: "number" | "integer" | "boolean" | "string";
  modes: Mode[];
}

/** Render the Fixed-mode cell with an appropriate typed control per field schema. */
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

  if (isNumeric(fieldSchema)) {
    return (
      <NumericStepper
        value={config.fixed}
        step={sm[name ?? ""] ?? (fieldSchema.type === "integer" ? 1 : "any")}
        onChange={(v) => onChange({ ...config, fixed: v })}
      />
    );
  }

  return (
    <input
      class="lzw-input"
      type="text"
      value={config.fixed ?? ""}
      onChange={(e) =>
        onChange({
          ...config,
          fixed: (e.target as HTMLInputElement).value,
        })
      }
    />
  );
}

export function SearchSpace({ schema, rootSchema, value, onChange, modelConfig, task, uiSchema }: SearchSpaceProps) {
  const root = rootSchema ?? schema;
  const modelParams = modelConfig?.params ?? {};
  const autoNumLeaves = modelConfig?.auto_num_leaves ?? true;
  const autoNumLeavesConfig = value["auto_num_leaves"] ? toParamConfig(value["auto_num_leaves"]) : undefined;

  // Read from uiSchema (backend contract)
  const optionSets: Record<string, Record<string, string[]>> = uiSchema?.option_sets ?? {};
  const stepMap: Record<string, number> = uiSchema?.step_map ?? {};
  const searchSpaceCatalog: CatalogEntry[] = uiSchema?.search_space_catalog ?? [];

  // Convert UI ParamConfig to LizyML search space format and emit.
  // Only non-fixed entries are stored in the space dict.
  const handleUpdate = (key: string, config: ParamConfig, fieldType?: string) => {
    const newValue = { ...value };
    if (config.mode === "fixed") {
      delete newValue[key];
    } else if (config.mode === "range") {
      const spaceType = fieldType === "integer" ? "int" : "float";
      newValue[key] = { type: spaceType, low: config.low, high: config.high, log: config.log ?? false };
    } else if (config.mode === "choice") {
      newValue[key] = { type: "categorical", choices: config.choices ?? [] };
    }
    onChange(newValue);
  };

  // LGBMConfig schema-level tunable fields (excluding auto_num_leaves, num_leaves_ratio handled separately)
  const resolved = resolve(schema, root);
  const properties = resolved.properties ?? {};
  const schemaKeys = Object.keys(properties).filter(
    (k) =>
      !SCHEMA_EXCLUDE.has(k) &&
      !NON_TUNABLE.has(k) &&
      resolve(properties[k], root).type !== "object",
  );

  return (
    <div class="lzw-search-space">
      <div class="lzw-search-space-grid" role="table">
        <div class="lzw-search-space-grid__header" role="row">
          <div role="columnheader">Parameter</div>
          <div role="columnheader">Mode</div>
          <div role="columnheader">Configuration</div>
        </div>
          {/* Catalog rows (from backend contract search_space_catalog) */}
          {searchSpaceCatalog.map(({ key, title, paramType, modes }) => {
            const rawVal = key in modelParams ? modelParams[key] : undefined;
            let fakeSchema: Record<string, any>;
            let configFixed: any;

            if (key === "objective") {
              fakeSchema = {
                title,
                type: "string",
                enum: optionSets.objective?.[task ?? ""] ?? [],
                default: rawVal,
              };
              configFixed = rawVal;
            } else if (key === "metric") {
              fakeSchema = {
                title,
                type: "array",
                items: { enum: optionSets.model_metric?.[task ?? ""] ?? [] },
                default: rawVal,
              };
              configFixed = rawVal; // keep as string[], not JSON.stringify
            } else {
              const fixedVal = Array.isArray(rawVal) ? JSON.stringify(rawVal) : rawVal;
              fakeSchema = { title, type: paramType, default: fixedVal };
              configFixed = fixedVal;
            }

            const paramConfig: ParamConfig = value[key] ? toParamConfig(value[key]) : { mode: "fixed", fixed: configFixed };
            return (
              <SearchSpaceRow
                key={key}
                name={key}
                fieldSchema={fakeSchema}
                config={paramConfig}
                modes={modes}
                onChange={(c) => handleUpdate(key, c, paramType)}
                stepMap={stepMap}
              />
            );
          })}

          {/* auto_num_leaves (LGBMConfig schema) */}
          {(() => {
            const fs = { ...resolve(properties["auto_num_leaves"] ?? {}, root), type: "boolean" };
            const config: ParamConfig = value["auto_num_leaves"] ? toParamConfig(value["auto_num_leaves"]) : {
              mode: "fixed",
              fixed: autoNumLeaves,
            };
            return (
              <SearchSpaceRow
                key="auto_num_leaves"
                name="auto_num_leaves"
                fieldSchema={fs}
                config={config}
                modes={["fixed", "choice"]}
                onChange={(c) => handleUpdate("auto_num_leaves", c, "boolean")}
                stepMap={stepMap}
              />
            );
          })()}

          {/* Conditional: num_leaves_ratio (auto_num_leaves=ON) */}
          {autoNumLeaves && (() => {
              const fs = resolve(properties["num_leaves_ratio"] ?? {}, root);
              const config: ParamConfig = value["num_leaves_ratio"] ? toParamConfig(value["num_leaves_ratio"]) : {
                mode: "fixed",
                fixed: modelConfig?.num_leaves_ratio ?? 1.0,
              };
              return (
                <SearchSpaceRow
                  key="num_leaves_ratio"
                  name="num_leaves_ratio"
                  fieldSchema={fs}
                  config={config}
                  modes={["fixed", "range"]}
                  onChange={(c) => handleUpdate("num_leaves_ratio", c, "number")}
                  stepMap={stepMap}
                />
              );
            })()}

          {/* num_leaves: shown when auto_num_leaves=OFF or auto_num_leaves=Choice */}
          {(!autoNumLeaves || autoNumLeavesConfig?.mode === "choice") && (
            <SearchSpaceRow
              key="num_leaves"
              name="num_leaves"
              fieldSchema={{ title: "Num Leaves", type: "integer", default: 256 }}
              config={
                value["num_leaves"] ? toParamConfig(value["num_leaves"]) : {
                  mode: "fixed",
                  fixed: modelParams.num_leaves ?? 256,
                }
              }
              modes={["fixed", "range"]}
              onChange={(c) => handleUpdate("num_leaves", c, "integer")}
              stepMap={stepMap}
            />
          )}

          {/* Remaining LGBMConfig schema fields (min_data_in_leaf_ratio, balanced, etc.) */}
          {schemaKeys.map((key) => {
            const fieldSchema = resolve(properties[key], root);
            const config: ParamConfig = value[key] ? toParamConfig(value[key]) : {
              mode: "fixed",
              fixed: fieldSchema.default,
            };
            return (
              <SearchSpaceRow
                key={key}
                name={key}
                fieldSchema={fieldSchema}
                config={config}
                modes={availableModes(fieldSchema)}
                onChange={(c) => handleUpdate(key, c, fieldSchema.type)}
                stepMap={stepMap}
              />
            );
          })}

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
                  onChange({ mode: m, fixed: fieldSchema.default });
                } else if (m === "range") {
                  onChange({ mode: m, low: 0, high: 1, log: false });
                } else {
                  onChange({
                    mode: m,
                    choices: fieldSchema.enum ?? [fieldSchema.default],
                  });
                }
              }}
            >
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
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
                  onChange({
                    ...config,
                    log: (e.target as HTMLInputElement).checked,
                  })
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
