/**
 * FitSubTab — Model/Evaluation/Calibration/Training sections with Accordion layout.
 * Renders backend-contract-driven sections for the Fit workflow.
 */
import { useCallback, useEffect, useRef } from "preact/hooks";
import { Accordion } from "../components/Accordion";
import { DynForm } from "../components/DynForm";
import { ModelSection } from "../components/ModelEditors";
import type { TypedParamMeta } from "../components/ModelEditors";
import { NumericStepper } from "../components/NumericStepper";
import { ConfigFooter } from "../components/ConfigFooter";
import { getSectionSchema, getUnknownKeys } from "./configHelpers";

interface FitSubTabProps {
  localConfig: Record<string, any>;
  configSchema: Record<string, any>;
  uiSchema: Record<string, any>;
  task: string;
  dfInfo: Record<string, any>;
  handleChange: (newConfig: Record<string, any>) => void;
  handleSectionChange: (sectionKey: string, sectionValue: Record<string, any>) => void;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  rawYaml: string | null;
  setRawYaml: (value: string | null) => void;
  yamlExportCount?: number;
}

/** Strategies that use a group column. */
const GROUP_STRATEGIES = new Set([
  "group_kfold", "stratified_group_kfold", "group_time_series", "blocked_group_kfold",
]);

/** Strategies that use a time column. */
const TIME_STRATEGIES = new Set([
  "time_series", "purged_time_series", "group_time_series",
]);

/** Filter inner validation options based on CV strategy. */
function filterInnerValidOptions(
  options: string[],
  cv: Record<string, any> | undefined,
): string[] {
  const strategy = cv?.strategy ?? "kfold";
  const hasGroup = GROUP_STRATEGIES.has(strategy);
  const hasTime = TIME_STRATEGIES.has(strategy);
  return options.filter((opt) => {
    if (opt === "group_holdout") return hasGroup;
    if (opt === "time_holdout") return hasTime;
    return true;
  });
}

export function FitSubTab({
  localConfig,
  configSchema,
  uiSchema,
  task,
  dfInfo,
  handleChange,
  handleSectionChange,
  sendAction,
  rawYaml,
  setRawYaml,
  yamlExportCount,
}: FitSubTabProps) {
  const sections: Array<{ key: string; title: string }> = uiSchema.sections ?? [];
  const optionSets: Record<string, Record<string, string[]>> = uiSchema.option_sets ?? {};
  const parameterHints: TypedParamMeta[] = uiSchema.parameter_hints ?? [];
  const stepMap: Record<string, number> = uiSchema.step_map ?? {};
  const conditionalVisibility: Record<string, any> = uiSchema.conditional_visibility ?? {};
  const defaults: Record<string, any> = uiSchema.defaults ?? {};
  const sectionKeys = sections.map((s) => s.key);
  const unknownKeys = getUnknownKeys(configSchema, sectionKeys);

  // Auto-reset inner_valid method when CV strategy changes
  const allInnerValidOpts: string[] = uiSchema.inner_valid_options ?? [];
  const availableInnerValidOpts = filterInnerValidOptions(allInnerValidOpts, dfInfo?.cv);
  const cvStrategy = dfInfo?.cv?.strategy ?? "kfold";
  const currentInnerValid =
    localConfig.training?.early_stopping?.inner_valid?.method ?? "holdout";
  const handleSectionChangeRef = useRef(handleSectionChange);
  handleSectionChangeRef.current = handleSectionChange;
  const localConfigRef = useRef(localConfig);
  localConfigRef.current = localConfig;
  useEffect(() => {
    const available = filterInnerValidOptions(allInnerValidOpts, { strategy: cvStrategy });
    if (available.length > 0 && !available.includes(currentInnerValid)) {
      const cfg = localConfigRef.current;
      handleSectionChangeRef.current("training", {
        ...(cfg.training ?? {}),
        early_stopping: {
          ...(cfg.training?.early_stopping ?? {}),
          inner_valid: { method: "holdout" },
        },
      });
    }
  }, [cvStrategy, currentInnerValid, allInnerValidOpts]);

  // Calibration
  const calibrationEnabled = localConfig.calibration != null;
  const calibrationDefaults = defaults.calibration ?? { method: "platt", params: {} };
  const handleCalibrationToggle = useCallback(
    (enabled: boolean) => {
      handleChange({
        ...localConfig,
        calibration: enabled ? (localConfig.calibration ?? calibrationDefaults) : null,
      });
    },
    [localConfig, handleChange, calibrationDefaults],
  );
  const calibrationVisibleTasks: string[] = conditionalVisibility.calibration?.task ?? ["binary"];
  const showCalibration = calibrationVisibleTasks.includes(task);

  return (
    <>
      {/* config_version: read-only per BLUEPRINT §5.3 */}
      <div class="lzw-form-row" style="padding: 4px 12px; opacity: 0.7;">
        <span class="lzw-label">Config Version</span>
        <span>{localConfig.config_version ?? 1}</span>
      </div>

      {/* Sections from backend contract */}
      {sections.map(({ key, title }) => {
        const sectionSchema = getSectionSchema(configSchema, key);
        if (!sectionSchema) return null;

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
                columns={dfInfo?.columns ?? []}
                additionalParams={uiSchema.additional_params ?? []}
              />
            </Accordion>
          );
        }

        if (key === "evaluation") {
          const evalSection = (localConfig.evaluation as any) ?? {};
          const evalMetrics: string[] = evalSection.metrics ?? [];
          const evalMetricOpts = optionSets.metric?.[task] ?? [];
          const evalParams = evalSection.params ?? {};
          const hasPrecisionAtK = evalMetrics.includes("precision_at_k");
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
                        handleSectionChange("evaluation", { ...evalSection, metrics: next });
                      }}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              </div>
              {hasPrecisionAtK && (
                <div class="lzw-form-row">
                  <label class="lzw-label">precision_at_k: k</label>
                  <NumericStepper
                    value={evalParams.precision_at_k_k ?? 10}
                    min={1}
                    max={100}
                    step={1}
                    onChange={(v) =>
                      handleSectionChange("evaluation", {
                        ...evalSection,
                        params: { ...evalParams, precision_at_k_k: v ?? 10 },
                      })
                    }
                  />
                </div>
              )}
            </Accordion>
          );
        }

        if (key === "calibration") {
          if (!showCalibration) return null;
          const calValue = (localConfig.calibration ?? calibrationDefaults) as Record<string, any>;
          const calMethodOpts: string[] = uiSchema.calibration_methods ?? ["platt", "isotonic", "beta"];
          const calParams = (calValue.params ?? {}) as Record<string, any>;
          return (
            <Accordion key="calibration" title="Calibration">
              <div class="lzw-form-row">
                <label class="lzw-label">Enable</label>
                <label class="lzw-toggle">
                  <input
                    type="checkbox"
                    checked={calibrationEnabled}
                    aria-label="Enable calibration"
                    onChange={(e) =>
                      handleCalibrationToggle((e.target as HTMLInputElement).checked)
                    }
                  />
                  <span class="lzw-toggle__slider" />
                </label>
              </div>
              {calibrationEnabled && (
                <>
                  <div class="lzw-form-row">
                    <label class="lzw-label">Method</label>
                    <select
                      class="lzw-select"
                      value={calValue.method ?? "platt"}
                      onChange={(e) =>
                        handleSectionChange("calibration", {
                          ...calValue,
                          method: (e.target as HTMLSelectElement).value,
                        })
                      }
                    >
                      {calMethodOpts.map((opt) => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  </div>
                  <div class="lzw-dynform__section-title">Params</div>
                  {Object.entries(calParams).map(([pKey, pVal]) => (
                    <div key={pKey} class="lzw-form-row">
                      <span class="lzw-label">{pKey}</span>
                      <NumericStepper
                        value={pVal as number}
                        onChange={(v) =>
                          handleSectionChange("calibration", {
                            ...calValue,
                            params: { ...calParams, [pKey]: v },
                          })
                        }
                      />
                      <button
                        type="button"
                        class="lzw-tag__remove"
                        aria-label={`Remove ${pKey}`}
                        onClick={() => {
                          const { [pKey]: _, ...rest } = calParams;
                          handleSectionChange("calibration", { ...calValue, params: rest });
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                  {(() => {
                    const calParamsByMethod: Record<string, string[]> = uiSchema.calibration_params ?? {};
                    const calParamOpts: string[] = calParamsByMethod[calValue.method ?? "platt"] ?? [];
                    const available = calParamOpts.filter((p) => !(p in calParams));
                    if (available.length === 0) return null;
                    return (
                      <select
                        class="lzw-select"
                        value=""
                        onChange={(e) => {
                          const v = (e.target as HTMLSelectElement).value;
                          if (v) {
                            handleSectionChange("calibration", {
                              ...calValue,
                              params: { ...calParams, [v]: 0 },
                            });
                          }
                        }}
                      >
                        <option value="">+ Add</option>
                        {available.map((p) => (
                          <option key={p} value={p}>{p}</option>
                        ))}
                      </select>
                    );
                  })()}
                </>
              )}
            </Accordion>
          );
        }

        if (key === "training") {
          const training = (localConfig.training ?? {}) as Record<string, any>;
          const earlyStop = training.early_stopping ?? {};
          const innerValidOpts = availableInnerValidOpts;
          return (
            <Accordion key="training" title="Training">
              <div class="lzw-form-row">
                <label class="lzw-label">Seed</label>
                <NumericStepper
                  value={training.seed ?? 42}
                  step={1}
                  onChange={(v) =>
                    handleSectionChange("training", { ...training, seed: v ?? 42 })
                  }
                />
              </div>
              <div class="lzw-form-row">
                <label class="lzw-label">Early Stopping</label>
                <label class="lzw-toggle">
                  <input
                    type="checkbox"
                    checked={earlyStop.enabled ?? true}
                    aria-label="Enable early stopping"
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
                          early_stopping: { ...earlyStop, rounds: v ?? 150 },
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
                          early_stopping: { ...earlyStop, validation_ratio: v ?? 0.1 },
                        })
                      }
                    />
                  </div>
                  <div class="lzw-form-row">
                    <label class="lzw-label">Inner Validation</label>
                    <select
                      class="lzw-select"
                      value={innerValidOpts.includes(earlyStop.inner_valid?.method ?? "holdout")
                        ? (earlyStop.inner_valid?.method ?? "holdout")
                        : "holdout"}
                      onChange={(e) => {
                        const v = (e.target as HTMLSelectElement).value;
                        handleSectionChange("training", {
                          ...training,
                          early_stopping: { ...earlyStop, inner_valid: { method: v } },
                        });
                      }}
                    >
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

        return (
          <Accordion key={key} title={title}>
            <DynForm
              schema={sectionSchema}
              rootSchema={configSchema}
              value={localConfig[key] ?? {}}
              onChange={(v) => handleSectionChange(key, v)}
            />
          </Accordion>
        );
      })}

      {/* Render unknown sections as fallback */}
      {unknownKeys.map((key) => {
        const sectionSchema = getSectionSchema(configSchema, key);
        if (!sectionSchema) return null;
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

      <ConfigFooter sendAction={sendAction} rawYaml={rawYaml} setRawYaml={setRawYaml} yamlExportCount={yamlExportCount} />
    </>
  );
}
