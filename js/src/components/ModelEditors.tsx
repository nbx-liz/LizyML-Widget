/**
 * ModelEditors — Model section sub-components for Fit tab.
 *
 * Extracted from ConfigTab.tsx to keep file size under 800 lines (P-014).
 * Contains: TypedParamsEditor, ModelSection, FeatureWeightsEditor, AdditionalParamsEditor.
 */
import { NumericStepper } from "./NumericStepper";
import { DynForm } from "./DynForm";

export type TypedParamKind = "objective" | "model_metric" | "integer" | "number" | "boolean";
export interface TypedParamMeta { key: string; label: string; kind: TypedParamKind; step?: number; }

/** Model section fields handled by custom ModelSection (not delegated to DynForm). */
const HANDLED_MODEL_FIELDS = new Set([
  "name", "auto_num_leaves", "num_leaves_ratio", "params",
  "min_data_in_leaf_ratio", "min_data_in_bin_ratio", "feature_weights", "balanced",
]);

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
            <div key={key} class="lzw-form-row" style="align-items:flex-start">
              <label class="lzw-label">{label}</label>
              <div class="lzw-segment">
                {opts.map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    class={`lzw-segment__btn ${current === opt ? "lzw-segment__btn--active" : ""}`}
                    aria-pressed={current === opt}
                    onClick={() => set(key, opt)}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          );
        }

        if (kind === "model_metric") {
          const opts = optionSets.model_metric?.[task] ?? [];
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

/** Feature Weights editor: toggle + column select + stepper rows. */
function FeatureWeightsEditor({
  value,
  onChange,
  columns,
}: {
  value: Record<string, number> | null;
  onChange: (v: Record<string, number> | null) => void;
  columns: Array<{ name: string }>;
}) {
  const enabled = value != null;
  const weights = value ?? {};
  const entries = Object.entries(weights);
  const usedCols = new Set(entries.map(([k]) => k));
  const availableCols = columns.filter((c) => !usedCols.has(c.name));

  return (
    <div>
      <div class="lzw-form-row">
        <label class="lzw-label">Feature Weights</label>
        <label class="lzw-toggle">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => {
              onChange((e.target as HTMLInputElement).checked ? {} : null);
            }}
          />
          <span class="lzw-toggle__slider" />
        </label>
      </div>
      {enabled && (
        <div class="lzw-indent">
          {entries.map(([col, w]) => (
            <div key={col} class="lzw-form-row">
              <select
                class="lzw-select"
                value={col}
                onChange={(e) => {
                  const newCol = (e.target as HTMLSelectElement).value;
                  if (newCol === col) return;
                  const { [col]: _, ...rest } = weights;
                  onChange({ ...rest, [newCol]: w });
                }}
              >
                <option value={col}>{col}</option>
                {availableCols.map((c) => (
                  <option key={c.name} value={c.name}>{c.name}</option>
                ))}
              </select>
              <NumericStepper
                value={w}
                step={0.1}
                onChange={(v) => onChange({ ...weights, [col]: v ?? 1.0 })}
              />
              <button
                type="button"
                class="lzw-tag__remove"
                aria-label={`Remove ${col}`}
                onClick={() => {
                  const { [col]: _, ...rest } = weights;
                  onChange(rest);
                }}
              >
                ×
              </button>
            </div>
          ))}
          {availableCols.length > 0 && (
            <button
              type="button"
              class="lzw-btn"
              onClick={() => onChange({ ...weights, [availableCols[0].name]: 1.0 })}
            >
              + Add
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** Additional Params: param name select + stepper + delete. */
function AdditionalParamsEditor({
  value,
  onChange,
  additionalParams,
  excludeKeys,
  stepMap,
}: {
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
  additionalParams: string[];
  excludeKeys: Set<string>;
  stepMap: Record<string, number>;
}) {
  const entries = Object.entries(value).filter(([k]) => !excludeKeys.has(k));
  const usedKeys = new Set(entries.map(([k]) => k));
  const availableKeys = additionalParams.filter(
    (p) => !usedKeys.has(p) && !excludeKeys.has(p),
  );

  return (
    <div>
      <div class="lzw-dynform__section-title">Additional Params</div>
      {entries.map(([key, val]) => (
        <div key={key} class="lzw-form-row">
          <select
            class="lzw-select"
            value={key}
            onChange={(e) => {
              const newKey = (e.target as HTMLSelectElement).value;
              if (newKey === key) return;
              const { [key]: _, ...rest } = value;
              onChange({ ...rest, [newKey]: val });
            }}
          >
            <option value={key}>{key}</option>
            {availableKeys.map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
          <NumericStepper
            value={val}
            step={stepMap[key] ?? "any"}
            onChange={(v) => onChange({ ...value, [key]: v })}
          />
          <button
            type="button"
            class="lzw-tag__remove"
            aria-label={`Remove ${key}`}
            onClick={() => {
              const { [key]: _, ...rest } = value;
              onChange(rest);
            }}
          >
            ×
          </button>
        </div>
      ))}
      {availableKeys.length > 0 && (
        <button
          type="button"
          class="lzw-btn"
          onClick={() => onChange({ ...value, [availableKeys[0]]: 0 })}
        >
          + Add
        </button>
      )}
    </div>
  );
}

/** Custom Model section with Smart Params, typed params, Additional Params. */
export function ModelSection({
  schema,
  rootSchema,
  value,
  onChange,
  task,
  parameterHints,
  optionSets,
  stepMap,
  columns,
  additionalParams,
}: {
  schema: Record<string, any>;
  rootSchema: Record<string, any>;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
  task: string;
  parameterHints: TypedParamMeta[];
  optionSets: Record<string, Record<string, string[]>>;
  stepMap: Record<string, number>;
  columns: Array<{ name: string }>;
  additionalParams: string[];
}) {
  const params = (value.params ?? {}) as Record<string, any>;
  const autoNumLeaves = value.auto_num_leaves ?? true;

  const setField = (k: string, v: any) => onChange({ ...value, [k]: v });
  const setParam = (k: string, v: any) =>
    onChange({ ...value, params: { ...params, [k]: v } });
  const setParams = (newParams: Record<string, any>) =>
    onChange({ ...value, params: newParams });

  const hintKeys = new Set(parameterHints.map((h) => h.key));
  const excludeFromAdditional = new Set([
    ...hintKeys, "verbose", "num_leaves", "num_threads",
  ]);

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
      <div class="lzw-form-row">
        <label class="lzw-label">Model Type</label>
        {value.name ? (
          <span class="lzw-tag lzw-tag--muted">{value.name}</span>
        ) : (
          <span class="lzw-tag lzw-tag--warning">model.name missing</span>
        )}
      </div>

      <div class="lzw-dynform__section-title">Smart Params</div>

      <div class="lzw-form-row">
        <label class="lzw-label">Auto Num Leaves</label>
        <label class="lzw-toggle">
          <input
            type="checkbox"
            checked={autoNumLeaves}
            onChange={(e) => {
              const v = (e.target as HTMLInputElement).checked;
              const updated = v
                ? (() => { const { num_leaves: _, ...rest } = params; return rest; })()
                : { ...params, num_leaves: params.num_leaves ?? 256 };
              onChange({ ...value, auto_num_leaves: v, params: updated });
            }}
          />
          <span class="lzw-toggle__slider" />
        </label>
      </div>

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

      <div class="lzw-form-row">
        <label class="lzw-label">Min Data In Leaf Ratio</label>
        <NumericStepper
          value={value.min_data_in_leaf_ratio ?? 0.01}
          step={0.01}
          min={0}
          onChange={(v) => setField("min_data_in_leaf_ratio", v ?? 0.01)}
        />
      </div>

      <div class="lzw-form-row">
        <label class="lzw-label">Min Data In Bin Ratio</label>
        <NumericStepper
          value={value.min_data_in_bin_ratio ?? 0.01}
          step={0.01}
          min={0}
          onChange={(v) => setField("min_data_in_bin_ratio", v ?? 0.01)}
        />
      </div>

      <FeatureWeightsEditor
        value={value.feature_weights ?? null}
        onChange={(v) => setField("feature_weights", v)}
        columns={columns}
      />

      <div class="lzw-form-row">
        <label class="lzw-label">Balanced</label>
        <label class="lzw-toggle">
          <input
            type="checkbox"
            checked={value.balanced ?? false}
            onChange={(e) => {
              const checked = (e.target as HTMLInputElement).checked;
              setField("balanced", checked ? true : null);
            }}
          />
          <span class="lzw-toggle__slider" />
        </label>
      </div>

      {Object.keys(filteredSchema.properties ?? {}).length > 0 && (
        <DynForm
          schema={filteredSchema}
          rootSchema={rootSchema}
          value={value}
          onChange={onChange}
        />
      )}

      <div class="lzw-dynform__section-title">Model Params</div>

      <TypedParamsEditor
        task={task}
        autoNumLeaves={autoNumLeaves}
        value={params}
        onChange={setParams}
        parameterHints={parameterHints}
        optionSets={optionSets}
        stepMap={stepMap}
      />

      <div class="lzw-form-row">
        <label class="lzw-label">Log Output</label>
        <NumericStepper
          value={params.verbose ?? -1}
          step={1}
          onChange={(v) => setParam("verbose", v ?? -1)}
        />
      </div>

      <AdditionalParamsEditor
        value={params}
        onChange={setParams}
        additionalParams={additionalParams}
        excludeKeys={excludeFromAdditional}
        stepMap={stepMap}
      />
    </div>
  );
}
