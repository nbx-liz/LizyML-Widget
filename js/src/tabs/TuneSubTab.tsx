/**
 * TuneSubTab — Search Space + Tuning Settings + Evaluation sections.
 * Renders the Tune workflow with independent config from Fit.
 */
import { useCallback } from "preact/hooks";
import { Accordion } from "../components/Accordion";
import { SearchSpace } from "../components/SearchSpace";
import { NumericStepper } from "../components/NumericStepper";
import { ConfigFooter } from "../components/ConfigFooter";

interface TuneSubTabProps {
  localConfig: Record<string, any>;
  uiSchema: Record<string, any>;
  task: string;
  dfInfo: Record<string, any>;
  handleChange: (newConfig: Record<string, any>) => void;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  rawYaml: string | null;
  setRawYaml: (value: string | null) => void;
  yamlExportCount?: number;
}

export function TuneSubTab({
  localConfig,
  uiSchema,
  task,
  dfInfo,
  handleChange,
  sendAction,
  rawYaml,
  setRawYaml,
  yamlExportCount,
}: TuneSubTabProps) {
  const optionSets: Record<string, Record<string, string[]>> = uiSchema.option_sets ?? {};
  const tuning = localConfig.tuning ?? {};
  const tuneSpace = tuning.optuna?.space ?? {};
  const tuneModelParams = tuning.model_params ?? {};
  const tuneTraining = tuning.training ?? {};
  const tuneEvaluation = tuning.evaluation ?? {};
  const tuneMetrics: string[] = tuneEvaluation.metrics ?? [];
  const evalMetricOpts = optionSets.metric?.[task] ?? [];
  const optimizationMetric = tuneMetrics[0] ?? (evalMetricOpts[0] ?? "");
  const additionalMetrics = tuneMetrics.slice(1);

  const handleTuneParam = useCallback(
    (key: string, value: any) => {
      const optuna = tuning.optuna ?? {};
      const params = optuna.params ?? {};
      handleChange({
        ...localConfig,
        tuning: { ...tuning, optuna: { ...optuna, params: { ...params, [key]: value } } },
      });
    },
    [localConfig, tuning, handleChange],
  );

  const handleTuningField = useCallback(
    (field: string, value: any) => {
      handleChange({
        ...localConfig,
        tuning: { ...tuning, [field]: value },
      });
    },
    [localConfig, tuning, handleChange],
  );

  return (
    <div class="lzw-tune-tab">
      <Accordion title="Tuning Settings">
        <div class="lzw-form-row">
          <label class="lzw-label">n_trials</label>
          <NumericStepper
            value={localConfig.tuning?.optuna?.params?.n_trials ?? 50}
            min={1}
            step={1}
            onChange={(v) => handleTuneParam("n_trials", v ?? 50)}
          />
        </div>
      </Accordion>

      <Accordion title="Search Space">
        <SearchSpace
          spaceValue={tuneSpace}
          fixedModelParams={tuneModelParams}
          fixedTraining={tuneTraining}
          modelConfig={localConfig.model}
          trainingConfig={localConfig.training}
          task={task}
          uiSchema={uiSchema}
          columns={dfInfo?.columns ?? []}
          onChange={({ space, fixedModelParams: mp, fixedTraining: tr }) => {
            const optuna = tuning.optuna ?? {};
            handleChange({
              ...localConfig,
              tuning: {
                ...tuning,
                optuna: { ...optuna, space },
                model_params: mp,
                training: tr,
              },
            });
          }}
        />
      </Accordion>

      <Accordion title="Evaluation">
        {evalMetricOpts.length > 0 && (
          <div class="lzw-form-row" style="align-items:flex-start">
            <label class="lzw-label">Optimization Metric</label>
            <div class="lzw-segment">
              {evalMetricOpts.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  class={`lzw-segment__btn ${optimizationMetric === opt ? "lzw-segment__btn--active" : ""}`}
                  aria-pressed={optimizationMetric === opt}
                  onClick={() => {
                    handleTuningField("evaluation", {
                      ...tuneEvaluation,
                      metrics: [opt, ...additionalMetrics.filter((m) => m !== opt)],
                    });
                  }}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>
        )}
        <div class="lzw-form-row" style="align-items:flex-start">
          <label class="lzw-label">Additional Metrics</label>
          <div class="lzw-chip-group">
            {evalMetricOpts
              .filter((opt) => opt !== optimizationMetric)
              .map((opt) => (
                <button
                  key={opt}
                  type="button"
                  class={`lzw-chip ${additionalMetrics.includes(opt) ? "lzw-chip--active" : ""}`}
                  onClick={() => {
                    const next = additionalMetrics.includes(opt)
                      ? additionalMetrics.filter((m) => m !== opt)
                      : [...additionalMetrics, opt];
                    handleTuningField("evaluation", {
                      ...tuneEvaluation,
                      metrics: [optimizationMetric, ...next],
                    });
                  }}
                >
                  {opt}
                </button>
              ))}
          </div>
        </div>
        {tuneMetrics.includes("precision_at_k") && (
          <div class="lzw-form-row">
            <label class="lzw-label">precision_at_k: k</label>
            <NumericStepper
              value={(tuneEvaluation.params ?? {}).precision_at_k_k ?? 10}
              min={1}
              max={100}
              step={1}
              onChange={(v) =>
                handleTuningField("evaluation", {
                  ...tuneEvaluation,
                  params: { ...(tuneEvaluation.params ?? {}), precision_at_k_k: v ?? 10 },
                })
              }
            />
          </div>
        )}
      </Accordion>

      <ConfigFooter sendAction={sendAction} rawYaml={rawYaml} setRawYaml={setRawYaml} yamlExportCount={yamlExportCount} />
    </div>
  );
}
