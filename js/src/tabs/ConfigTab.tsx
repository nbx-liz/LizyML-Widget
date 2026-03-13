/**
 * ConfigTab — Fit sub-tab with sectioned Accordion layout + Tune sub-tab with SearchSpace.
 * Form changes are debounced (300ms) before syncing to Python via patch_config.
 * All backend-specific constants (option_sets, parameter_hints, search_space_catalog)
 * are read from backendContract.ui_schema.
 */
import { useState, useRef, useCallback, useEffect } from "preact/hooks";
import { useCustomMsg } from "../hooks/useModel";
import { Accordion } from "../components/Accordion";
import { DynForm } from "../components/DynForm";
import { SearchSpace } from "../components/SearchSpace";
import { NumericStepper } from "../components/NumericStepper";

type SubTab = "fit" | "tune";

/** Schema top-level keys managed by Data Tab — hidden from Config Tab. */
const DATA_TAB_KEYS = ["data", "features", "split"];

/** Model section fields handled by custom ModelSection (not delegated to DynForm). */
const HANDLED_MODEL_FIELDS = new Set(["name", "auto_num_leaves", "num_leaves_ratio", "params"]);

interface ConfigTabProps {
  backendContract: Record<string, any>;
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

type TypedParamKind = "objective" | "metric" | "integer" | "number" | "boolean";
interface TypedParamMeta { key: string; label: string; kind: TypedParamKind; step?: number; }

function TypedParamsEditor({
  task,
  autoNumLeaves,
  value,
  onChange,
  parameterHints,
  optionSets,
  stepMap,
}: {
  task: string;
  autoNumLeaves: boolean;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
  parameterHints: TypedParamMeta[];
  optionSets: Record<string, Record<string, string[]>>;
  stepMap: Record<string, number>;
}) {
  const set = (k: string, v: any) => onChange({ ...value, [k]: v });

  return (
    <div>
      {parameterHints.map(({ key, label, kind }) => {
        const current = value[key];

        if (kind === "objective") {
          const opts = optionSets.objective?.[task] ?? [];
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
          const opts = optionSets.metric?.[task] ?? [];
          const selected: string[] = Array.isArray(current) ? current : [];
          return (
            <div key={key} class="lzw-form-row" style="align-items:flex-start">
              <label class="lzw-label">{label}</label>
              <div class="lzw-chip-group">
                {opts.map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    class={`lzw-chip ${selected.includes(opt) ? "lzw-chip--active" : ""}`}
                    onClick={() => {
                      const next = selected.includes(opt)
                        ? selected.filter((v) => v !== opt)
                        : [...selected, opt];
                      set(key, next);
                    }}
                  >
                    {opt}
                  </button>
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
            <NumericStepper
              value={current}
              step={stepMap[key] ?? (kind === "integer" ? 1 : "any")}
              onChange={(v) => set(key, v)}
            />
          </div>
        );
      })}

      {!autoNumLeaves && (
        <div class="lzw-form-row">
          <label class="lzw-label">Num Leaves</label>
          <NumericStepper
            value={value.num_leaves ?? 256}
            min={2}
            step={1}
            onChange={(v) => set("num_leaves", v ?? 256)}
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
  parameterHints,
  optionSets,
  stepMap,
}: {
  schema: Record<string, any>;
  rootSchema: Record<string, any>;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
  task: string;
  parameterHints: TypedParamMeta[];
  optionSets: Record<string, Record<string, string[]>>;
  stepMap: Record<string, number>;
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
        {value.name ? (
          <span class="lzw-tag lzw-tag--muted">{value.name}</span>
        ) : (
          <span class="lzw-tag lzw-tag--warning">model.name missing</span>
        )}
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
          <NumericStepper
            value={value.num_leaves_ratio ?? 1.0}
            step={0.05}
            min={0.01}
            max={1}
            onChange={(v) => setField("num_leaves_ratio", v ?? 1.0)}
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
        parameterHints={parameterHints}
        optionSets={optionSets}
        stepMap={stepMap}
      />

      {/* Log Output (verbose) */}
      <div class="lzw-form-row">
        <label class="lzw-label">Log Output</label>
        <NumericStepper
          value={params.verbose ?? -1}
          step={1}
          onChange={(v) => setParam("verbose", v ?? -1)}
        />
      </div>
    </div>
  );
}

/** Get schema keys not in known sections or data tab keys. */
function getUnknownKeys(rootSchema: Record<string, any>, sectionKeys: string[]): string[] {
  const resolved = resolveSchema(rootSchema, rootSchema);
  const props = resolved.properties ?? {};
  const knownKeys = new Set([
    ...sectionKeys,
    ...DATA_TAB_KEYS,
    "tuning",
    "config_version",  // Read-only, shown separately
    "task",            // Managed by Data tab
  ]);
  return Object.keys(props).filter((k) => !knownKeys.has(k));
}

/** Build patch ops from old and new config objects. */
function diffToPatchOps(
  oldConfig: Record<string, any>,
  newConfig: Record<string, any>,
  prefix = "",
): Array<{ op: string; path: string; value: any }> {
  const ops: Array<{ op: string; path: string; value: any }> = [];

  for (const key of Object.keys(newConfig)) {
    const path = prefix ? `${prefix}.${key}` : key;
    const oldVal = oldConfig[key];
    const newVal = newConfig[key];

    if (JSON.stringify(oldVal) !== JSON.stringify(newVal)) {
      ops.push({ op: "set", path, value: newVal });
    }
  }

  // Check for removed keys
  for (const key of Object.keys(oldConfig)) {
    if (!(key in newConfig)) {
      const path = prefix ? `${prefix}.${key}` : key;
      ops.push({ op: "unset", path, value: null });
    }
  }

  return ops;
}

export function ConfigTab({
  backendContract,
  config,
  dfInfo,
  status,
  sendAction,
  model,
}: ConfigTabProps) {
  const [subTab, setSubTab] = useState<SubTab>("fit");
  const [localConfig, setLocalConfig] = useState<Record<string, any>>(config);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSentRef = useRef<Record<string, any>>(config);

  // Extract ui_schema and config_schema from backend contract
  const uiSchema = backendContract?.ui_schema ?? {};
  const configSchema = backendContract?.config_schema ?? {};
  const sections: Array<{ key: string; title: string }> = uiSchema.sections ?? [];
  const optionSets: Record<string, Record<string, string[]>> = uiSchema.option_sets ?? {};
  const parameterHints: TypedParamMeta[] = uiSchema.parameter_hints ?? [];
  const stepMap: Record<string, number> = uiSchema.step_map ?? {};
  const conditionalVisibility: Record<string, any> = uiSchema.conditional_visibility ?? {};
  const defaults: Record<string, any> = uiSchema.defaults ?? {};
  const sectionKeys = sections.map((s) => s.key);

  // Sync from Python → local when Python config changes externally
  useEffect(() => {
    setLocalConfig(config);
    lastSentRef.current = config;
  }, [config]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Auto-set tune metric when task changes or metric is unset
  const task = dfInfo?.task ?? "";
  const metricOpts = optionSets.metric?.[task] ?? [];
  useEffect(() => {
    if (metricOpts.length === 0) return;
    const current = localConfig.tuning?.optuna?.params?.metric;
    if (!current || !metricOpts.includes(current)) {
      const tuning = localConfig.tuning ?? {};
      const optuna = tuning.optuna ?? {};
      const params = optuna.params ?? {};
      const updated = {
        ...localConfig,
        tuning: { ...tuning, optuna: { ...optuna, params: { ...params, metric: metricOpts[0] } } },
      };
      setLocalConfig(updated);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        const ops = diffToPatchOps(lastSentRef.current, updated);
        if (ops.length > 0) {
          sendAction("patch_config", { ops });
          lastSentRef.current = updated;
        }
      }, 300);
    }
  }, [task]); // Intentionally depends only on task

  // Debounced send to Python via patch_config
  const handleChange = useCallback(
    (newConfig: Record<string, any>) => {
      setLocalConfig(newConfig);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        const ops = diffToPatchOps(lastSentRef.current, newConfig);
        if (ops.length > 0) {
          sendAction("patch_config", { ops });
          lastSentRef.current = newConfig;
        }
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

  // Tune requires at least one range/choice param, unless backend allows empty space
  const tuneSpace = localConfig.tuning?.optuna?.space ?? {};
  const capabilities = backendContract?.capabilities ?? {};
  const allowEmptySpace = capabilities?.tune?.allow_empty_space ?? false;
  const hasSearchParam = allowEmptySpace || Object.values(tuneSpace).some(
    (p: any) => p.type === "float" || p.type === "int" || p.type === "categorical",
  );

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
  const unknownKeys = getUnknownKeys(configSchema, sectionKeys);

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
  const calibrationDefaults = defaults.calibration ?? { method: "platt", n_splits: 5, params: {} };
  const handleCalibrationToggle = useCallback(
    (enabled: boolean) => {
      handleChange({
        ...localConfig,
        calibration: enabled
          ? (localConfig.calibration ?? calibrationDefaults)
          : null,
      });
    },
    [localConfig, handleChange, calibrationDefaults],
  );

  // Calibration visibility from conditional_visibility
  const calibrationVisibleTasks: string[] = conditionalVisibility.calibration?.task ?? ["binary"];

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

            {/* Sections from backend contract */}
            {sections.map(({ key, title }) => {
              // Calibration: conditional visibility from ui_schema
              if (key === "calibration" && !calibrationVisibleTasks.includes(task)) return null;

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
                      parameterHints={parameterHints}
                      optionSets={optionSets}
                      stepMap={stepMap}
                    />
                  </Accordion>
                );
              }

              // Evaluation section: custom checkbox group for metrics
              if (key === "evaluation") {
                const evalMetrics: string[] = (localConfig.evaluation as any)?.metrics ?? [];
                const evalMetricOpts = optionSets.metric?.[task] ?? [];
                return (
                  <Accordion key="evaluation" title="Evaluation">
                    <div class="lzw-form-row" style="align-items:flex-start">
                      <label class="lzw-label">Metrics</label>
                      <div class="lzw-chip-group">
                        {evalMetricOpts.map((opt) => (
                          <button
                            key={opt}
                            type="button"
                            class={`lzw-chip ${evalMetrics.includes(opt) ? "lzw-chip--active" : ""}`}
                            onClick={() => {
                              const next = evalMetrics.includes(opt)
                                ? evalMetrics.filter((v: string) => v !== opt)
                                : [...evalMetrics, opt];
                              handleSectionChange("evaluation", {
                                ...((localConfig.evaluation as any) ?? {}),
                                metrics: next,
                              });
                            }}
                          >
                            {opt}
                          </button>
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
                // Inner valid options from backend contract
                const innerValidOpts: string[] = uiSchema.inner_valid_options ?? [];

                return (
                  <Accordion key="training" title="Training">
                    <div class="lzw-form-row">
                      <label class="lzw-label">Seed</label>
                      <NumericStepper
                        value={training.seed ?? 42}
                        step={1}
                        onChange={(v) =>
                          handleSectionChange("training", {
                            ...training,
                            seed: v ?? 42,
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
                          <NumericStepper
                            value={earlyStop.rounds ?? 150}
                            step={50}
                            min={1}
                            onChange={(v) =>
                              handleSectionChange("training", {
                                ...training,
                                early_stopping: {
                                  ...earlyStop,
                                  rounds: v ?? 150,
                                },
                              })
                            }
                          />
                        </div>
                        <div class="lzw-form-row">
                          <label class="lzw-label">Validation Ratio</label>
                          <NumericStepper
                            value={earlyStop.validation_ratio ?? 0.1}
                            step={0.05}
                            min={0}
                            max={1}
                            onChange={(v) =>
                              handleSectionChange("training", {
                                ...training,
                                early_stopping: {
                                  ...earlyStop,
                                  validation_ratio: v ?? 0.1,
                                },
                              })
                            }
                          />
                        </div>
                        <div class="lzw-form-row">
                          <label class="lzw-label">Inner Validation</label>
                          <select
                            class="lzw-select"
                            value={earlyStop.inner_valid?.method ?? ""}
                            onChange={(e) => {
                              const v = (e.target as HTMLSelectElement).value;
                              handleSectionChange("training", {
                                ...training,
                                early_stopping: {
                                  ...earlyStop,
                                  inner_valid: v ? { method: v } : null,
                                },
                              });
                            }}
                          >
                            <option value="">Default</option>
                            {innerValidOpts.map((opt: string) => (
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
                <NumericStepper
                  value={localConfig.tuning?.optuna?.params?.n_trials ?? 50}
                  min={1}
                  step={1}
                  onChange={(v) => handleTuneParam("n_trials", v ?? 50)}
                />
              </div>
              {(() => {
                const tuneMetricOpts = optionSets.metric?.[task] ?? [];
                if (tuneMetricOpts.length === 0) return null;
                const currentMetric = localConfig.tuning?.optuna?.params?.metric ?? tuneMetricOpts[0];
                return (
                  <div class="lzw-form-row">
                    <label class="lzw-label">metric</label>
                    <div class="lzw-segment">
                      {tuneMetricOpts.map((opt) => (
                        <button
                          key={opt}
                          type="button"
                          class={`lzw-segment__btn ${currentMetric === opt ? "lzw-segment__btn--active" : ""}`}
                          aria-pressed={currentMetric === opt}
                          onClick={() => handleTuneParam("metric", opt)}
                        >
                          {opt}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </Accordion>

            <Accordion title="Search Space">
              <SearchSpace
                schema={getSectionSchema(configSchema, "model") ?? {}}
                rootSchema={configSchema}
                value={tuneSpace}
                modelConfig={localConfig.model}
                task={task}
                uiSchema={uiSchema}
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
              <NumericStepper
                value={(localConfig.model?.params as any)?.verbose ?? -1}
                step={1}
                onChange={(v) => {
                  const mdl = localConfig.model ?? {};
                  const prm = (mdl.params as any) ?? {};
                  handleChange({
                    ...localConfig,
                    model: { ...mdl, params: { ...prm, verbose: v ?? -1 } },
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
