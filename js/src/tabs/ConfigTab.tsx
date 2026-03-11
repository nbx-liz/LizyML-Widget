/**
 * ConfigTab — Fit sub-tab with sectioned Accordion layout + Tune sub-tab with SearchSpace.
 * Form changes are debounced (300ms) before syncing to Python.
 */
import { useState, useRef, useCallback, useEffect } from "preact/hooks";
import { useCustomMsg } from "../hooks/useModel";
import { Accordion } from "../components/Accordion";
import { DynForm } from "../components/DynForm";
import { SearchSpace } from "../components/SearchSpace";

type SubTab = "fit" | "tune";

/** Schema top-level keys managed by Data Tab — hidden from Config Tab. */
const DATA_TAB_KEYS = ["data", "features", "split"];

/** Fit sub-tab section layout per BLUEPRINT §5.3. */
const FIT_SECTIONS = [
  { key: "model", title: "Model" },
  { key: "training", title: "Training" },
  { key: "evaluation", title: "Evaluation" },
  { key: "calibration", title: "Calibration" },
] as const;

interface ConfigTabProps {
  configSchema: Record<string, any>;
  config: Record<string, any>;
  dfInfo: Record<string, any>;
  status: string;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  model?: any;
}

/** Resolve a $ref like "#/$defs/FooBar" against the root schema. */
function resolveRef(
  ref: string,
  root: Record<string, any>,
): Record<string, any> {
  const parts = ref.replace("#/", "").split("/");
  let node: any = root;
  for (const p of parts) {
    node = node?.[p];
  }
  return node ?? {};
}

/** Resolve schema, handling $ref and allOf with single ref. */
function resolveSchema(
  schema: Record<string, any>,
  root: Record<string, any>,
): Record<string, any> {
  if (schema.$ref) return resolveRef(schema.$ref, root);
  if (schema.allOf?.length === 1 && schema.allOf[0].$ref) {
    const resolved = resolveRef(schema.allOf[0].$ref, root);
    const { allOf: _, ...rest } = schema;
    return { ...resolved, ...rest };
  }
  return schema;
}

/** Extract a section sub-schema from the root config schema. */
function getSectionSchema(
  rootSchema: Record<string, any>,
  key: string,
): Record<string, any> | null {
  const resolved = resolveSchema(rootSchema, rootSchema);
  const props = resolved.properties ?? {};
  if (!props[key]) return null;
  return resolveSchema(props[key], rootSchema);
}

/** Model section fields handled by custom ModelSection (not delegated to DynForm). */
const HANDLED_MODEL_FIELDS = new Set(["name", "auto_num_leaves", "num_leaves_ratio", "params"]);

const OBJECTIVE_OPTIONS: Record<string, string[]> = {
  regression: ["huber", "mse", "mae", "quantile", "mape", "cross_entropy"],
  binary: ["binary", "cross_entropy", "cross_entropy_lambda"],
  multiclass: ["multiclass", "softmax", "multiclassova"],
};

const METRIC_OPTIONS: Record<string, string[]> = {
  regression: ["huber", "mae", "mape", "mse", "rmse", "quantile"],
  binary: ["auc", "binary_logloss", "binary_error", "average_precision"],
  multiclass: ["auc_mu", "multi_logloss", "multi_error"],
};

type TypedParamKind = "objective" | "metric" | "integer" | "number" | "boolean";
interface TypedParamMeta { key: string; label: string; kind: TypedParamKind; }

const TYPED_PARAMS: TypedParamMeta[] = [
  { key: "objective",         label: "Objective",         kind: "objective" },
  { key: "metric",            label: "Metric",            kind: "metric" },
  { key: "n_estimators",      label: "N Estimators",      kind: "integer" },
  { key: "learning_rate",     label: "Learning Rate",     kind: "number" },
  { key: "max_depth",         label: "Max Depth",         kind: "integer" },
  { key: "max_bin",           label: "Max Bin",           kind: "integer" },
  { key: "feature_fraction",  label: "Feature Fraction",  kind: "number" },
  { key: "bagging_fraction",  label: "Bagging Fraction",  kind: "number" },
  { key: "bagging_freq",      label: "Bagging Freq",      kind: "integer" },
  { key: "lambda_l1",         label: "Lambda L1",         kind: "number" },
  { key: "lambda_l2",         label: "Lambda L2",         kind: "number" },
  { key: "first_metric_only", label: "First Metric Only", kind: "boolean" },
];

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
};

function TypedParamsEditor({
  task,
  autoNumLeaves,
  value,
  onChange,
}: {
  task: string;
  autoNumLeaves: boolean;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
}) {
  const set = (k: string, v: any) => onChange({ ...value, [k]: v });

  return (
    <div>
      {TYPED_PARAMS.map(({ key, label, kind }) => {
        const current = value[key];

        if (kind === "objective") {
          const opts = OBJECTIVE_OPTIONS[task] ?? [];
          return (
            <div key={key} class="lzw-form-row">
              <label class="lzw-label">{label}</label>
              <select
                class="lzw-select"
                value={current ?? ""}
                onChange={(e) => set(key, (e.target as HTMLSelectElement).value)}
              >
                {current && !opts.includes(current) && (
                  <option value={current}>{current}</option>
                )}
                {opts.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>
          );
        }

        if (kind === "metric") {
          const opts = METRIC_OPTIONS[task] ?? [];
          const selected: string[] = Array.isArray(current) ? current : [];
          return (
            <div key={key} class="lzw-form-row" style="align-items:flex-start">
              <label class="lzw-label">{label}</label>
              <div class="lzw-checkbox-group">
                {opts.map((opt) => (
                  <label key={opt}>
                    <input
                      type="checkbox"
                      checked={selected.includes(opt)}
                      onChange={(e) => {
                        const checked = (e.target as HTMLInputElement).checked;
                        set(key, checked ? [...selected, opt] : selected.filter((v) => v !== opt));
                      }}
                    />
                    {opt}
                  </label>
                ))}
              </div>
            </div>
          );
        }

        if (kind === "boolean") {
          return (
            <div key={key} class="lzw-form-row">
              <label class="lzw-label">{label}</label>
              <label class="lzw-toggle">
                <input
                  type="checkbox"
                  checked={current ?? false}
                  onChange={(e) => set(key, (e.target as HTMLInputElement).checked)}
                />
                <span class="lzw-toggle__slider" />
              </label>
            </div>
          );
        }

        return (
          <div key={key} class="lzw-form-row">
            <label class="lzw-label">{label}</label>
            <input
              class="lzw-input lzw-input--sm"
              type="number"
              step={STEP_MAP[key] ?? (kind === "integer" ? 1 : "any")}
              value={current ?? ""}
              onChange={(e) => {
                const raw = (e.target as HTMLInputElement).value;
                set(key, kind === "integer" ? parseInt(raw) : parseFloat(raw));
              }}
            />
          </div>
        );
      })}

      {!autoNumLeaves && (
        <div class="lzw-form-row">
          <label class="lzw-label">Num Leaves</label>
          <input
            class="lzw-input lzw-input--sm"
            type="number"
            min={2}
            step={1}
            value={value.num_leaves ?? 256}
            onChange={(e) => set("num_leaves", parseInt((e.target as HTMLInputElement).value))}
          />
        </div>
      )}
    </div>
  );
}

/** Custom Model section with typed params, auto_num_leaves toggle, verbose at bottom. */
function ModelSection({
  schema,
  rootSchema,
  value,
  onChange,
  task,
}: {
  schema: Record<string, any>;
  rootSchema: Record<string, any>;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
  task: string;
}) {
  const params = (value.params ?? {}) as Record<string, any>;
  const autoNumLeaves = value.auto_num_leaves ?? true;

  const setField = (k: string, v: any) => onChange({ ...value, [k]: v });
  const setParam = (k: string, v: any) =>
    onChange({ ...value, params: { ...params, [k]: v } });
  const setParams = (newParams: Record<string, any>) =>
    onChange({ ...value, params: newParams });

  const filteredSchema = {
    ...schema,
    properties: Object.fromEntries(
      Object.entries((schema.properties ?? {}) as Record<string, any>).filter(
        ([k]) => !HANDLED_MODEL_FIELDS.has(k),
      ),
    ),
  };

  return (
    <div>
      {/* name: read-only const */}
      <div class="lzw-form-row">
        <label class="lzw-label">Model Type</label>
        <span class="lzw-tag lzw-tag--muted">{value.name ?? "lgbm"}</span>
      </div>

      {/* auto_num_leaves toggle */}
      <div class="lzw-form-row">
        <label class="lzw-label">Auto Num Leaves</label>
        <label class="lzw-toggle">
          <input
            type="checkbox"
            checked={autoNumLeaves}
            onChange={(e) => {
              const v = (e.target as HTMLInputElement).checked;
              const newParams = { ...params };
              if (v) {
                delete newParams.num_leaves;
              } else {
                newParams.num_leaves = newParams.num_leaves ?? 256;
              }
              onChange({ ...value, auto_num_leaves: v, params: newParams });
            }}
          />
          <span class="lzw-toggle__slider" />
        </label>
      </div>

      {/* num_leaves_ratio (auto ON only) — num_leaves is inside TypedParamsEditor */}
      {autoNumLeaves && (
        <div class="lzw-form-row">
          <label class="lzw-label">Num Leaves Ratio</label>
          <input
            class="lzw-input lzw-input--sm"
            type="number"
            step={0.05}
            min={0.01}
            max={1}
            value={value.num_leaves_ratio ?? 1.0}
            onChange={(e) =>
              setField("num_leaves_ratio", parseFloat((e.target as HTMLInputElement).value))
            }
          />
        </div>
      )}

      {/* Remaining schema fields (min_data_in_leaf_ratio, etc.) */}
      <DynForm
        schema={filteredSchema}
        rootSchema={rootSchema}
        value={value}
        onChange={onChange}
      />

      {/* Typed params (objective/metric/numerics/booleans + conditional num_leaves) */}
      <TypedParamsEditor
        task={task}
        autoNumLeaves={autoNumLeaves}
        value={params}
        onChange={setParams}
      />

      {/* Log Output (verbose) */}
      <div class="lzw-form-row">
        <label class="lzw-label">Log Output</label>
        <input
          class="lzw-input lzw-input--sm"
          type="number"
          value={params.verbose ?? -1}
          onChange={(e) =>
            setParam("verbose", parseInt((e.target as HTMLInputElement).value))
          }
        />
      </div>
    </div>
  );
}

/** Get schema keys not in known sections or data tab keys. */
function getUnknownKeys(rootSchema: Record<string, any>): string[] {
  const resolved = resolveSchema(rootSchema, rootSchema);
  const props = resolved.properties ?? {};
  const knownKeys = new Set([
    ...FIT_SECTIONS.map((s) => s.key),
    ...DATA_TAB_KEYS,
    "tuning",
    "config_version",  // Read-only, shown separately
    "task",            // Managed by Data tab
  ]);
  return Object.keys(props).filter((k) => !knownKeys.has(k));
}

export function ConfigTab({
  configSchema,
  config,
  dfInfo,
  status,
  sendAction,
  model,
}: ConfigTabProps) {
  const [subTab, setSubTab] = useState<SubTab>("fit");
  const [localConfig, setLocalConfig] = useState<Record<string, any>>(config);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync from Python → local when Python config changes externally
  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Debounced send to Python
  const handleChange = useCallback(
    (newConfig: Record<string, any>) => {
      setLocalConfig(newConfig);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        sendAction("update_config", { config: newConfig });
      }, 300);
    },
    [sendAction],
  );

  /** Update a single section of config. */
  const handleSectionChange = useCallback(
    (sectionKey: string, sectionValue: Record<string, any>) => {
      handleChange({ ...localConfig, [sectionKey]: sectionValue });
    },
    [localConfig, handleChange],
  );

  const canRun =
    status === "data_loaded" || status === "completed" || status === "failed" || status === "running";

  // Tune requires at least one range/choice param
  const tuneSpace = localConfig.tuning?.optuna?.space ?? {};
  const hasSearchParam = Object.values(tuneSpace).some(
    (p: any) => p.mode === "range" || p.mode === "choice",
  );

  const task = dfInfo?.task ?? "";

  /** Update a single tuning.optuna.params field. */
  const handleTuneParam = useCallback(
    (key: string, value: any) => {
      const tuning = localConfig.tuning ?? {};
      const optuna = tuning.optuna ?? {};
      const params = optuna.params ?? {};
      handleChange({
        ...localConfig,
        tuning: { ...tuning, optuna: { ...optuna, params: { ...params, [key]: value } } },
      });
    },
    [localConfig, handleChange],
  );
  const unknownKeys = getUnknownKeys(configSchema);

  // Import/Export YAML
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [rawYaml, setRawYaml] = useState<string | null>(null);

  useCustomMsg(model, useCallback((msg: any) => {
    if (msg.type === "yaml_export") {
      const blob = new Blob([msg.content], { type: "text/yaml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "config.yaml";
      a.click();
      URL.revokeObjectURL(url);
    } else if (msg.type === "raw_config") {
      setRawYaml(msg.content);
    }
  }, []));

  const handleImportFile = useCallback((e: Event) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      sendAction("import_yaml", { content: reader.result as string });
    };
    reader.readAsText(file);
    // Reset so the same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [sendAction]);

  // Calibration toggle: null = OFF, object = ON (per BLUEPRINT §5.3)
  const calibrationEnabled = localConfig.calibration != null;
  const handleCalibrationToggle = useCallback(
    (enabled: boolean) => {
      handleChange({
        ...localConfig,
        calibration: enabled
          ? (localConfig.calibration ?? { method: "platt", n_splits: 5, params: {} })
          : null,
      });
    },
    [localConfig, handleChange],
  );

  return (
    <div class="lzw-config-tab">
      {/* Sub-tab bar */}
      <div class="lzw-subtabs">
        <button
          class={`lzw-subtabs__btn ${subTab === "fit" ? "lzw-subtabs__btn--active" : ""}`}
          onClick={() => setSubTab("fit")}
          type="button"
        >
          Fit
        </button>
        <button
          class={`lzw-subtabs__btn ${subTab === "tune" ? "lzw-subtabs__btn--active" : ""}`}
          onClick={() => setSubTab("tune")}
          type="button"
        >
          Tune
        </button>

        {/* Action button (sticky right) */}
        <div class="lzw-subtabs__right">
          {subTab === "fit" && (
            <button
              class="lzw-btn lzw-btn--primary"
              disabled={!canRun || status === "running"}
              onClick={() => sendAction("fit")}
              type="button"
            >
              {status === "running" ? "Running..." : "Fit"}
            </button>
          )}
          {subTab === "tune" && (
            <button
              class="lzw-btn lzw-btn--primary"
              disabled={!canRun || status === "running" || !hasSearchParam}
              onClick={() => sendAction("tune")}
              type="button"
            >
              {status === "running" ? "Running..." : "Tune"}
            </button>
          )}
        </div>
      </div>

      {/* Sub-tab content */}
      <div class="lzw-config-tab__body">
        {subTab === "fit" && (
          <>
            {/* config_version: read-only per BLUEPRINT §5.3 */}
            <div class="lzw-form-row" style="padding: 4px 12px; opacity: 0.7;">
              <span class="lzw-label">Config Version</span>
              <span>{localConfig.config_version ?? 1}</span>
            </div>

            {/* BLUEPRINT §5.3 sections: Model, Training, Evaluation, Calibration */}
            {FIT_SECTIONS.map(({ key, title }) => {
              // Calibration: only show for binary tasks
              if (key === "calibration" && task !== "binary") return null;

              const sectionSchema = getSectionSchema(configSchema, key);
              if (!sectionSchema) return null;

              // Model section: custom rendering (auto_num_leaves conditional, verbose, params KVEditor)
              if (key === "model") {
                return (
                  <Accordion key="model" title="Model">
                    <ModelSection
                      schema={sectionSchema}
                      rootSchema={configSchema}
                      value={localConfig.model ?? {}}
                      onChange={(v) => handleSectionChange("model", v)}
                      task={task}
                    />
                  </Accordion>
                );
              }

              // Evaluation section: custom checkbox group for metrics
              if (key === "evaluation") {
                const evalMetrics: string[] = (localConfig.evaluation as any)?.metrics ?? [];
                const metricOpts = METRIC_OPTIONS[task] ?? [];
                return (
                  <Accordion key="evaluation" title="Evaluation">
                    <div class="lzw-form-row" style="align-items:flex-start">
                      <label class="lzw-label">Metrics</label>
                      <div class="lzw-checkbox-group">
                        {metricOpts.map((opt) => (
                          <label key={opt}>
                            <input
                              type="checkbox"
                              checked={evalMetrics.includes(opt)}
                              onChange={(e) => {
                                const checked = (e.target as HTMLInputElement).checked;
                                const next = checked
                                  ? [...evalMetrics, opt]
                                  : evalMetrics.filter((v: string) => v !== opt);
                                handleSectionChange("evaluation", {
                                  ...((localConfig.evaluation as any) ?? {}),
                                  metrics: next,
                                });
                              }}
                            />
                            {opt}
                          </label>
                        ))}
                      </div>
                    </div>
                  </Accordion>
                );
              }

              // Training section: DynForm + custom inner_valid select
              if (key === "training") {
                const training = (localConfig.training ?? {}) as Record<string, any>;
                const earlyStop = training.early_stopping ?? {};
                const splitConfig = (localConfig.split ?? {}) as Record<string, any>;

                // Derive inner_valid options from split config
                const innerValidOpts: string[] = [];
                if (splitConfig.method === "kfold") {
                  const n = splitConfig.n_splits ?? 5;
                  for (let i = 0; i < n; i++) innerValidOpts.push(`fold_${i}`);
                } else if (splitConfig.method === "holdout") {
                  innerValidOpts.push("holdout");
                }

                return (
                  <Accordion key="training" title="Training">
                    <div class="lzw-form-row">
                      <label class="lzw-label">Seed</label>
                      <input
                        class="lzw-input lzw-input--sm"
                        type="number"
                        step={1}
                        value={training.seed ?? 42}
                        onChange={(e) =>
                          handleSectionChange("training", {
                            ...training,
                            seed: parseInt((e.target as HTMLInputElement).value),
                          })
                        }
                      />
                    </div>
                    <div class="lzw-form-row">
                      <label class="lzw-label">Early Stopping</label>
                      <label class="lzw-toggle">
                        <input
                          type="checkbox"
                          checked={earlyStop.enabled ?? true}
                          onChange={(e) =>
                            handleSectionChange("training", {
                              ...training,
                              early_stopping: {
                                ...earlyStop,
                                enabled: (e.target as HTMLInputElement).checked,
                              },
                            })
                          }
                        />
                        <span class="lzw-toggle__slider" />
                      </label>
                    </div>
                    {(earlyStop.enabled ?? true) && (
                      <>
                        <div class="lzw-form-row">
                          <label class="lzw-label">Rounds</label>
                          <input
                            class="lzw-input lzw-input--sm"
                            type="number"
                            step={50}
                            min={1}
                            value={earlyStop.rounds ?? 150}
                            onChange={(e) =>
                              handleSectionChange("training", {
                                ...training,
                                early_stopping: {
                                  ...earlyStop,
                                  rounds: parseInt((e.target as HTMLInputElement).value),
                                },
                              })
                            }
                          />
                        </div>
                        <div class="lzw-form-row">
                          <label class="lzw-label">Validation Ratio</label>
                          <input
                            class="lzw-input lzw-input--sm"
                            type="number"
                            step={0.05}
                            min={0}
                            max={1}
                            value={earlyStop.validation_ratio ?? 0.1}
                            onChange={(e) =>
                              handleSectionChange("training", {
                                ...training,
                                early_stopping: {
                                  ...earlyStop,
                                  validation_ratio: parseFloat((e.target as HTMLInputElement).value),
                                },
                              })
                            }
                          />
                        </div>
                        <div class="lzw-form-row">
                          <label class="lzw-label">Inner Validation</label>
                          <select
                            class="lzw-select"
                            value={earlyStop.inner_valid ?? ""}
                            onChange={(e) => {
                              const v = (e.target as HTMLSelectElement).value;
                              handleSectionChange("training", {
                                ...training,
                                early_stopping: {
                                  ...earlyStop,
                                  inner_valid: v || null,
                                },
                              });
                            }}
                          >
                            <option value="">auto</option>
                            {innerValidOpts.map((opt) => (
                              <option key={opt} value={opt}>{opt}</option>
                            ))}
                          </select>
                        </div>
                      </>
                    )}
                  </Accordion>
                );
              }

              const calibrationToggle =
                key === "calibration" ? (
                  <label class="lzw-toggle">
                    <input
                      type="checkbox"
                      checked={calibrationEnabled}
                      onChange={(e) =>
                        handleCalibrationToggle(
                          (e.target as HTMLInputElement).checked,
                        )
                      }
                    />
                    <span class="lzw-toggle__slider" />
                  </label>
                ) : undefined;

              return (
                <Accordion
                  key={key}
                  title={title}
                  headerRight={calibrationToggle}
                >
                  {key === "calibration" && !calibrationEnabled ? (
                    <p class="lzw-muted">Calibration is disabled.</p>
                  ) : (
                    <DynForm
                      schema={sectionSchema}
                      rootSchema={configSchema}
                      value={localConfig[key] ?? {}}
                      onChange={(v) => handleSectionChange(key, v)}
                    />
                  )}
                </Accordion>
              );
            })}

            {/* Render unknown sections as fallback */}
            {unknownKeys.map((key) => {
              const sectionSchema = getSectionSchema(configSchema, key);
              if (!sectionSchema) return null;

              // Non-object scalar fields (e.g. output_dir: nullable string) — plain text input
              const isObject = sectionSchema.type === "object" || !!sectionSchema.properties;
              if (!isObject) {
                return (
                  <Accordion key={key} title={sectionSchema.title ?? key}>
                    <div class="lzw-form-row">
                      <label class="lzw-label">{sectionSchema.title ?? key}</label>
                      <input
                        class="lzw-input"
                        type="text"
                        placeholder="null"
                        value={localConfig[key] ?? ""}
                        onChange={(e) => {
                          const v = (e.target as HTMLInputElement).value;
                          handleChange({ ...localConfig, [key]: v || null });
                        }}
                      />
                    </div>
                  </Accordion>
                );
              }

              return (
                <Accordion key={key} title={key}>
                  <DynForm
                    schema={sectionSchema}
                    rootSchema={configSchema}
                    value={localConfig[key] ?? {}}
                    onChange={(v) => handleSectionChange(key, v)}
                  />
                </Accordion>
              );
            })}

            <div class="lzw-config-tab__footer">
              <input
                ref={fileInputRef}
                type="file"
                accept=".yaml,.yml,.json"
                style="display:none"
                onChange={handleImportFile}
              />
              <button class="lzw-btn" type="button" onClick={() => fileInputRef.current?.click()}>
                Import YAML
              </button>
              <button class="lzw-btn" type="button" onClick={() => sendAction("export_yaml")}>
                Export YAML
              </button>
              <button class="lzw-btn" type="button" onClick={() => sendAction("raw_config")}>
                Raw Config
              </button>
            </div>
            {rawYaml !== null && (
              <div class="lzw-modal-overlay" onClick={() => setRawYaml(null)}>
                <div class="lzw-modal" onClick={(e) => e.stopPropagation()}>
                  <div class="lzw-modal__header">
                    <span>Raw Config</span>
                    <button class="lzw-btn" type="button" onClick={() => setRawYaml(null)}>Close</button>
                  </div>
                  <pre class="lzw-pre">{rawYaml}</pre>
                </div>
              </div>
            )}
          </>
        )}

        {subTab === "tune" && (
          <div class="lzw-tune-tab">
            <Accordion title="Settings">
              <div class="lzw-form-row">
                <label class="lzw-label">n_trials</label>
                <input
                  class="lzw-input lzw-input--sm"
                  type="number"
                  min={1}
                  step={1}
                  value={localConfig.tuning?.optuna?.params?.n_trials ?? 50}
                  onChange={(e) => {
                    const v = parseInt((e.target as HTMLInputElement).value) || 50;
                    handleTuneParam("n_trials", v);
                  }}
                />
              </div>
              <div class="lzw-form-row">
                <label class="lzw-label">metric</label>
                <select
                  class="lzw-select"
                  value={localConfig.tuning?.optuna?.params?.metric ?? ""}
                  onChange={(e) => {
                    const v = (e.target as HTMLSelectElement).value;
                    handleTuneParam("metric", v || null);
                  }}
                >
                  <option value="">default</option>
                  {(METRIC_OPTIONS[task] ?? []).map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>
            </Accordion>

            <Accordion title="Search Space">
              <SearchSpace
                schema={getSectionSchema(configSchema, "model") ?? {}}
                rootSchema={configSchema}
                value={tuneSpace}
                modelConfig={localConfig.model}
                task={task}
                onChange={(space) => {
                  const tuning = localConfig.tuning ?? {};
                  const optuna = tuning.optuna ?? {};
                  handleChange({
                    ...localConfig,
                    tuning: { ...tuning, optuna: { ...optuna, space } },
                  });
                }}
              />
            </Accordion>

            {/* Log Output (verbose) — same treatment as Fit, BLUEPRINT §5.3 */}
            <div class="lzw-form-row" style="padding: 4px 12px;">
              <label class="lzw-label">Log Output</label>
              <input
                class="lzw-input lzw-input--sm"
                type="number"
                value={(localConfig.model?.params as any)?.verbose ?? -1}
                onChange={(e) => {
                  const model = localConfig.model ?? {};
                  const params = (model.params as any) ?? {};
                  handleChange({
                    ...localConfig,
                    model: { ...model, params: { ...params, verbose: parseInt((e.target as HTMLInputElement).value) } },
                  });
                }}
              />
            </div>

            <div class="lzw-config-tab__footer">
              <input
                ref={fileInputRef}
                type="file"
                accept=".yaml,.yml,.json"
                style="display:none"
                onChange={handleImportFile}
              />
              <button class="lzw-btn" type="button" onClick={() => fileInputRef.current?.click()}>
                Import YAML
              </button>
              <button class="lzw-btn" type="button" onClick={() => sendAction("export_yaml")}>
                Export YAML
              </button>
              <button class="lzw-btn" type="button" onClick={() => sendAction("raw_config")}>
                Raw Config
              </button>
            </div>
            {rawYaml !== null && (
              <div class="lzw-modal-overlay" onClick={() => setRawYaml(null)}>
                <div class="lzw-modal" onClick={(e) => e.stopPropagation()}>
                  <div class="lzw-modal__header">
                    <span>Raw Config</span>
                    <button class="lzw-btn" type="button" onClick={() => setRawYaml(null)}>Close</button>
                  </div>
                  <pre class="lzw-pre">{rawYaml}</pre>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
