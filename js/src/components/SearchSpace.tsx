/**
 * SearchSpace — per-parameter mode selection for tuning (Fixed/Range/Choice).
 *
 * For numeric params: Fixed | Range
 * For enum/string/boolean params: Fixed | Choice
 */
import { useState } from "preact/hooks";

interface SearchSpaceProps {
  schema: Record<string, any>;
  value: Record<string, any>;
  onChange: (value: Record<string, any>) => void;
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

export function SearchSpace({ schema, value, onChange }: SearchSpaceProps) {
  const root = schema;
  const resolved = resolve(schema, root);
  const properties = resolved.properties ?? {};
  const keys = Object.keys(properties);

  // Flatten to tunable params (skip nested objects, only leaf fields)
  const tunableKeys = keys.filter((k) => {
    const s = resolve(properties[k], root);
    return s.type !== "object";
  });

  if (!tunableKeys.length) {
    return <p class="lzw-muted">No tunable parameters found.</p>;
  }

  const handleUpdate = (key: string, config: ParamConfig) => {
    onChange({ ...value, [key]: config });
  };

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
          {tunableKeys.map((key) => {
            const fieldSchema = resolve(properties[key], root);
            const config: ParamConfig = value[key] ?? {
              mode: "fixed",
              fixed: fieldSchema.default,
            };
            const modes = availableModes(fieldSchema);
            return (
              <SearchSpaceRow
                key={key}
                name={key}
                fieldSchema={fieldSchema}
                config={config}
                modes={modes}
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
        {config.mode === "fixed" && (
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
        )}
        {config.mode === "range" && (
          <div class="lzw-search-space__range">
            <input
              class="lzw-input lzw-input--sm"
              type="number"
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
        )}
        {config.mode === "choice" && (
          <span class="lzw-muted">
            {(config.choices ?? []).join(", ")}
          </span>
        )}
      </td>
    </tr>
  );
}
