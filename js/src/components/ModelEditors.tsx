/**
 * ModelEditors — Model section sub-components for Fit tab.
 *
 * Extracted from ConfigTab.tsx to keep file size under 800 lines (P-014).
 * Contains: TypedParamsEditor, ModelSection, FeatureWeightsEditor, AdditionalParamsEditor.
 *
 * #119: backend-specific keys (smart_params group, hidden additional keys,
 * num_leaves default) are derived from the backend contract — no LightGBM
 * key names are hardcoded in this file.
 */
import { NumericStepper } from "./NumericStepper";
import { DynForm } from "./DynForm";

type TypedParamKind = "objective" | "model_metric" | "integer" | "number" | "boolean";
export interface TypedParamMeta { key: string; label: string; kind: TypedParamKind; step?: number; }

interface SmartParamCatalogEntry {
  key: string;
  title?: string;
  paramType?: string;
  group?: string;
  default?: any;
}

/** Structural keys filtered from DynForm regardless of backend. */
const STRUCTURAL_MODEL_FIELDS: ReadonlySet<string> = new Set(["name", "params"]);

/** Read the smart_params group keys from search_space_catalog. */
function getSmartParamsKeys(catalog: SmartParamCatalogEntry[]): Set<string> {
  return new Set(
    catalog
      .filter((entry) => entry.group === "smart_params")
      .map((entry) => entry.key),
  );
}

/** Look up a default value for a smart_params catalog entry. */
function getSmartParamDefault(
  catalog: SmartParamCatalogEntry[],
  key: string,
): any {
  return catalog.find((entry) => entry.key === key)?.default;
}

function TypedParamsEditor({
  task,
  autoNumLeaves,
  value,
  onChange,
  parameterHints,
  optionSets,
  stepMap,
  manualNumLeavesKey,
  manualNumLeavesDefault,
}: {
  task: string;
  autoNumLeaves: boolean;
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
  parameterHints: TypedParamMeta[];
  optionSets: Record<string, Record<string, string[]>>;
  stepMap: Record<string, number>;
  /** Key for the manual leaf-count override (e.g. "num_leaves"); only rendered when defined. */
  manualNumLeavesKey?: string;
  /** Default leaf count used when toggle is off and value is unset. */
  manualNumLeavesDefault?: number;
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
          const hasPrecisionAtK = selected.includes("precision_at_k");
          return (
            <>
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
              {hasPrecisionAtK && (
                <div class="lzw-form-row">
                  <label class="lzw-label">precision_at_k: k</label>
                  <NumericStepper
                    value={value._precision_at_k_k ?? 10}
                    min={1}
                    max={100}
                    step={1}
                    onChange={(v) => set("_precision_at_k_k", v ?? 10)}
                  />
                </div>
              )}
            </>
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

      {!autoNumLeaves && manualNumLeavesKey && (
        <div class="lzw-form-row">
          <label class="lzw-label">Num Leaves</label>
          <NumericStepper
            value={value[manualNumLeavesKey] ?? manualNumLeavesDefault}
            min={2}
            step={1}
            onChange={(v) => set(manualNumLeavesKey, v ?? manualNumLeavesDefault)}
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
  searchSpaceCatalog,
  additionalParamsHiddenKeys,
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
  /** search_space_catalog from backend ui_schema (optional; falls back to legacy literal set when absent). */
  searchSpaceCatalog?: SmartParamCatalogEntry[];
  /** Param keys hidden from Additional Params (e.g. verbose, num_threads). Sourced from contract capabilities. */
  additionalParamsHiddenKeys?: string[];
}) {
  const params = (value.params ?? {}) as Record<string, any>;
  // Smart Params section is currently LightGBM-aware (the auto-num-leaves
  // toggle pattern), but the *set* of smart-params keys comes from the
  // contract — the JS no longer ships its own list (#119).
  const catalog: SmartParamCatalogEntry[] = searchSpaceCatalog ?? [];
  const smartParamsKeys =
    catalog.length > 0
      ? getSmartParamsKeys(catalog)
      : new Set([
          // Fallback for unit-test fixtures that omit the contract; production
          // always supplies the catalog. Adding/removing smart params in the
          // backend automatically propagates without touching this set.
          "auto_num_leaves",
          "num_leaves_ratio",
          "num_leaves",
          "min_data_in_leaf_ratio",
          "min_data_in_bin_ratio",
          "feature_weights",
          "balanced",
        ]);
  const handledModelFields = new Set<string>([
    ...STRUCTURAL_MODEL_FIELDS,
    ...smartParamsKeys,
  ]);

  const autoNumLeavesKey = "auto_num_leaves";
  const manualNumLeavesKey = smartParamsKeys.has("num_leaves") ? "num_leaves" : undefined;
  // Default flows from the backend search_space_catalog — JS does not own a literal
  // (#119). When the catalog is absent (test fixtures), undefined falls through to
  // NumericStepper, which renders an empty stepper for the user to set.
  const manualNumLeavesDefault: number | undefined = manualNumLeavesKey
    ? (getSmartParamDefault(catalog, manualNumLeavesKey) as number | undefined)
    : undefined;
  const autoNumLeaves = value[autoNumLeavesKey] ?? true;

  const setField = (k: string, v: any) => onChange({ ...value, [k]: v });
  const setParam = (k: string, v: any) =>
    onChange({ ...value, params: { ...params, [k]: v } });
  const setParams = (newParams: Record<string, any>) =>
    onChange({ ...value, params: newParams });

  const hintKeys = new Set(parameterHints.map((h) => h.key));
  const hiddenKeys = additionalParamsHiddenKeys ?? [];
  // Exclude from Additional Params: parameter_hints, smart_params keys, and
  // backend-declared hidden keys (e.g. verbose, num_threads).
  const excludeFromAdditional = new Set<string>([
    ...hintKeys,
    ...smartParamsKeys,
    ...hiddenKeys,
  ]);

  const filteredSchema = {
    ...schema,
    properties: Object.fromEntries(
      Object.entries((schema.properties ?? {}) as Record<string, any>).filter(
        ([k]) => !handledModelFields.has(k),
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
              let updated: Record<string, any> = params;
              if (manualNumLeavesKey) {
                if (v) {
                  const { [manualNumLeavesKey]: _drop, ...rest } = params;
                  updated = rest;
                } else {
                  updated = {
                    ...params,
                    [manualNumLeavesKey]: params[manualNumLeavesKey] ?? manualNumLeavesDefault,
                  };
                }
              }
              onChange({ ...value, [autoNumLeavesKey]: v, params: updated });
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
            checked={value.balanced ?? true}
            onChange={(e) => {
              const checked = (e.target as HTMLInputElement).checked;
              setField("balanced", checked);
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
        manualNumLeavesKey={manualNumLeavesKey}
        manualNumLeavesDefault={manualNumLeavesDefault}
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
