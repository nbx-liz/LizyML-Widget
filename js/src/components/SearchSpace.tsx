/**
 * SearchSpace — per-parameter mode selection for tuning (Fixed/Range/Choice).
 *
 * Renders combined rows: model.params (LGBM_PARAMS) + LGBMConfig schema fields + verbose.
 * Only Range/Choice entries are stored in tuning.optuna.space; Fixed = not in space dict.
 */

interface SearchSpaceProps {
  schema: Record<string, any>;
  rootSchema?: Record<string, any>;
  value: Record<string, any>;
  onChange: (value: Record<string, any>) => void;
  modelConfig?: Record<string, any>;
  task?: string;
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

/** Task-dependent objective options. */
const OBJECTIVE_OPTIONS: Record<string, string[]> = {
  regression: ["huber", "mse", "mae", "quantile", "mape", "cross_entropy"],
  binary: ["binary", "cross_entropy", "cross_entropy_lambda"],
  multiclass: ["multiclass", "softmax", "multiclassova"],
};

/** Task-dependent metric options. */
const METRIC_OPTIONS: Record<string, string[]> = {
  regression: ["huber", "mae", "mape", "mse", "rmse", "quantile"],
  binary: ["auc", "binary_logloss", "binary_error", "average_precision"],
  multiclass: ["auc_mu", "multi_logloss", "multi_error"],
};

/** Step values for numeric fields (BLUEPRINT §5.3). */
const STEP_MAP: Record<string, number> = {
  n_estimators: 100,
  learning_rate: 0.001,
  max_depth: 1,
  max_bin: 1,
  feature_fraction: 0.05,
  bagging_fraction: 0.05,
  bagging_freq: 1,
  lambda_l1: 0.0001,
  lambda_l2: 0.0001,
  num_leaves_ratio: 0.05,
  num_leaves: 1,
};

/** Keys that are not tunable hyperparameters. */
const NON_TUNABLE = new Set(["name"]);

/** LGBMConfig schema fields managed separately in ModelSection (not shown as schema rows in SearchSpace). */
const SCHEMA_EXCLUDE = new Set(["name", "params", "auto_num_leaves", "num_leaves_ratio"]);

interface LgbmParamMeta {
  key: string;
  title: string;
  paramType: "number" | "integer" | "boolean" | "string";
}

/** model.params tunable params in display order (BLUEPRINT §5.3). */
const LGBM_PARAMS: LgbmParamMeta[] = [
  { key: "objective", title: "Objective", paramType: "string" },
  { key: "metric", title: "Metric", paramType: "string" },
  { key: "n_estimators", title: "N Estimators", paramType: "integer" },
  { key: "learning_rate", title: "Learning Rate", paramType: "number" },
  { key: "max_depth", title: "Max Depth", paramType: "integer" },
  { key: "max_bin", title: "Max Bin", paramType: "integer" },
  { key: "feature_fraction", title: "Feature Fraction", paramType: "number" },
  { key: "bagging_fraction", title: "Bagging Fraction", paramType: "number" },
  { key: "bagging_freq", title: "Bagging Freq", paramType: "integer" },
  { key: "lambda_l1", title: "Lambda L1", paramType: "number" },
  { key: "lambda_l2", title: "Lambda L2", paramType: "number" },
  { key: "first_metric_only", title: "First Metric Only", paramType: "boolean" },
];

function modesForParamType(paramType: LgbmParamMeta["paramType"]): Mode[] {
  if (paramType === "number" || paramType === "integer") return ["fixed", "range"];
  if (paramType === "boolean") return ["fixed", "choice"];
  return ["fixed"];
}

/** Render the Fixed-mode cell with an appropriate typed control per field schema. */
function renderFixed(
  fieldSchema: Record<string, any>,
  config: ParamConfig,
  onChange: (c: ParamConfig) => void,
) {
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
      <div class="lzw-checkbox-group">
        {(fieldSchema.items.enum as string[]).map((opt) => (
          <label key={opt}>
            <input
              type="checkbox"
              checked={selected.includes(opt)}
              onChange={(e) => {
                const checked = (e.target as HTMLInputElement).checked;
                onChange({
                  ...config,
                  fixed: checked ? [...selected, opt] : selected.filter((v) => v !== opt),
                });
              }}
            />
            {opt}
          </label>
        ))}
      </div>
    );
  }

  return (
    <input
      class="lzw-input"
      type={isNumeric(fieldSchema) ? "number" : "text"}
      value={config.fixed ?? ""}
      onChange={(e) =>
        onChange({
          ...config,
          fixed: isNumeric(fieldSchema)
            ? parseFloat((e.target as HTMLInputElement).value)
            : (e.target as HTMLInputElement).value,
        })
      }
    />
  );
}

export function SearchSpace({ schema, rootSchema, value, onChange, modelConfig, task }: SearchSpaceProps) {
  const root = rootSchema ?? schema;
  const modelParams = modelConfig?.params ?? {};
  const autoNumLeaves = modelConfig?.auto_num_leaves ?? true;
  const autoNumLeavesConfig = value["auto_num_leaves"];

  // Only non-fixed entries are stored in the space dict
  const handleUpdate = (key: string, config: ParamConfig) => {
    const newValue = { ...value };
    if (config.mode === "fixed") {
      delete newValue[key];
    } else {
      newValue[key] = config;
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
      <table class="lzw-table">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Mode</th>
            <th>Configuration</th>
          </tr>
        </thead>
        <tbody>
          {/* LGBMParam rows (from model.params) */}
          {LGBM_PARAMS.map(({ key, title, paramType }) => {
            const rawVal = key in modelParams ? modelParams[key] : undefined;
            let fakeSchema: Record<string, any>;
            let configFixed: any;

            if (key === "objective") {
              fakeSchema = {
                title,
                type: "string",
                enum: OBJECTIVE_OPTIONS[task ?? ""] ?? [],
                default: rawVal,
              };
              configFixed = rawVal;
            } else if (key === "metric") {
              fakeSchema = {
                title,
                type: "array",
                items: { enum: METRIC_OPTIONS[task ?? ""] ?? [] },
                default: rawVal,
              };
              configFixed = rawVal; // keep as string[], not JSON.stringify
            } else {
              const fixedVal = Array.isArray(rawVal) ? JSON.stringify(rawVal) : rawVal;
              fakeSchema = { title, type: paramType, default: fixedVal };
              configFixed = fixedVal;
            }

            const config: ParamConfig = value[key] ?? { mode: "fixed", fixed: configFixed };
            return (
              <SearchSpaceRow
                key={key}
                name={key}
                fieldSchema={fakeSchema}
                config={config}
                modes={
                  key === "objective" || key === "metric"
                    ? ["fixed", "choice"]
                    : modesForParamType(paramType)
                }
                onChange={(c) => handleUpdate(key, c)}
              />
            );
          })}

          {/* auto_num_leaves (LGBMConfig schema) */}
          {(() => {
            const fs = { ...resolve(properties["auto_num_leaves"] ?? {}, root), type: "boolean" };
            const config: ParamConfig = value["auto_num_leaves"] ?? {
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
                onChange={(c) => handleUpdate("auto_num_leaves", c)}
              />
            );
          })()}

          {/* Conditional: num_leaves_ratio (auto_num_leaves=ON) */}
          {autoNumLeaves && (() => {
              const fs = resolve(properties["num_leaves_ratio"] ?? {}, root);
              const config: ParamConfig = value["num_leaves_ratio"] ?? {
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
                  onChange={(c) => handleUpdate("num_leaves_ratio", c)}
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
                value["num_leaves"] ?? {
                  mode: "fixed",
                  fixed: modelParams.num_leaves ?? 256,
                }
              }
              modes={["fixed", "range"]}
              onChange={(c) => handleUpdate("num_leaves", c)}
            />
          )}

          {/* Remaining LGBMConfig schema fields (min_data_in_leaf_ratio, balanced, etc.) */}
          {schemaKeys.map((key) => {
            const fieldSchema = resolve(properties[key], root);
            const config: ParamConfig = value[key] ?? {
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
                onChange={(c) => handleUpdate(key, c)}
              />
            );
          })}

        </tbody>
      </table>
    </div>
  );
}

interface SearchSpaceRowProps {
  name: string;
  fieldSchema: Record<string, any>;
  config: ParamConfig;
  modes: Mode[];
  onChange: (config: ParamConfig) => void;
}

function SearchSpaceRow({
  name,
  fieldSchema,
  config,
  modes,
  onChange,
}: SearchSpaceRowProps) {
  const title = fieldSchema.title ?? name;

  return (
    <tr>
      <td class="lzw-table__name">{title}</td>
      <td>
        <select
          class="lzw-select"
          value={config.mode}
          onChange={(e) => {
            const mode = (e.target as HTMLSelectElement).value as Mode;
            if (mode === "fixed") {
              onChange({ mode, fixed: fieldSchema.default });
            } else if (mode === "range") {
              onChange({ mode, low: 0, high: 1, log: false });
            } else {
              onChange({
                mode,
                choices: fieldSchema.enum ?? [fieldSchema.default],
              });
            }
          }}
        >
          {modes.map((m) => (
            <option key={m} value={m}>
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </option>
          ))}
        </select>
      </td>
      <td>
        {config.mode === "fixed" && renderFixed(fieldSchema, config, onChange)}
        {config.mode === "range" && (() => {
          const stepVal = STEP_MAP[name] ?? (fieldSchema.type === "integer" ? 1 : "any");
          return (
          <div class="lzw-search-space__range">
            <input
              class="lzw-input lzw-input--sm"
              type="number"
              step={stepVal}
              placeholder="low"
              value={config.low ?? ""}
              onChange={(e) =>
                onChange({
                  ...config,
                  low: parseFloat((e.target as HTMLInputElement).value),
                })
              }
            />
            <span>~</span>
            <input
              class="lzw-input lzw-input--sm"
              type="number"
              step={stepVal}
              placeholder="high"
              value={config.high ?? ""}
              onChange={(e) =>
                onChange({
                  ...config,
                  high: parseFloat((e.target as HTMLInputElement).value),
                })
              }
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
            <div class="lzw-checkbox-group">
              {opts.map((opt) => (
                <label key={opt}>
                  <input
                    type="checkbox"
                    checked={(config.choices ?? []).includes(opt)}
                    onChange={(e) => {
                      const checked = (e.target as HTMLInputElement).checked;
                      const choices = config.choices ?? [];
                      onChange({
                        ...config,
                        choices: checked
                          ? [...choices, opt]
                          : choices.filter((v) => v !== opt),
                      });
                    }}
                  />
                  {opt}
                </label>
              ))}
            </div>
          );
        })()}
      </td>
    </tr>
  );
}
