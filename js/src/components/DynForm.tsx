/**
 * DynForm — renders form fields from a JSON Schema.
 *
 * Handles: number, integer, boolean, string+enum, string, array (tag input), object (nested).
 * Resolves $defs references from Pydantic v2 schemas.
 */
import { useState } from "preact/hooks";
import { NumericStepper } from "./NumericStepper";
interface DynFormProps {
  schema: Record<string, any>;
  value: Record<string, any>;
  onChange: (value: Record<string, any>) => void;
  rootSchema?: Record<string, any>;
  path?: string;
}

/** Tag-style multi-value input for array fields without enum items. */
function TagInput({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const [input, setInput] = useState("");

  const commit = (raw: string) => {
    const tag = raw.trim().replace(/,$/, "");
    if (tag && !value.includes(tag)) onChange([...value, tag]);
    setInput("");
  };

  return (
    <div class="lzw-tag-input">
      {value.map((tag) => (
        <span key={tag} class="lzw-tag lzw-tag--removable">
          {tag}
          <button
            type="button"
            class="lzw-tag__remove"
            onClick={() => onChange(value.filter((t) => t !== tag))}
          >
            ×
          </button>
        </span>
      ))}
      <input
        class="lzw-tag-input__field"
        type="text"
        value={input}
        placeholder={value.length === 0 ? "Add item..." : ""}
        onInput={(e) => setInput((e.target as HTMLInputElement).value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            commit((e.target as HTMLInputElement).value);
          } else if (e.key === "Backspace" && input === "" && value.length > 0) {
            onChange(value.slice(0, -1));
          }
        }}
        onBlur={(e) => {
          const v = (e.target as HTMLInputElement).value;
          if (v.trim()) commit(v);
        }}
      />
    </div>
  );
}

/** Key-value editor for object fields with additionalProperties (e.g. model.params). */
export function KVEditor({
  value,
  onChange,
}: {
  value: Record<string, any>;
  onChange: (v: Record<string, any>) => void;
}) {
  const entries = Object.entries(value ?? {});

  const setKey = (oldKey: string, newKey: string) => {
    if (!newKey || newKey === oldKey) return;
    const { [oldKey]: v, ...rest } = value ?? {};
    onChange({ ...rest, [newKey]: v });
  };

  const setValue = (key: string, raw: string) => {
    let parsed: any;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = raw;
    }
    onChange({ ...(value ?? {}), [key]: parsed });
  };

  const remove = (key: string) => {
    const { [key]: _, ...rest } = value ?? {};
    onChange(rest);
  };

  return (
    <div class="lzw-kv-editor">
      {entries.map(([k, v]) => (
        <div key={k} class="lzw-kv-editor__row">
          <input
            class="lzw-input lzw-input--sm"
            type="text"
            value={k}
            placeholder="key"
            onChange={(e) => setKey(k, (e.target as HTMLInputElement).value)}
          />
          <input
            class="lzw-input lzw-input--sm"
            type="text"
            value={v === null ? "null" : String(v)}
            placeholder="value"
            onChange={(e) => setValue(k, (e.target as HTMLInputElement).value)}
          />
          <button type="button" class="lzw-tag__remove" onClick={() => remove(k)}>
            ×
          </button>
        </div>
      ))}
      <button
        type="button"
        class="lzw-btn"
        onClick={() => onChange({ ...(value ?? {}), "": "" })}
      >
        + Add
      </button>
    </div>
  );
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
function resolve(
  schema: Record<string, any>,
  root: Record<string, any>,
): Record<string, any> {
  if (schema.$ref) return resolveRef(schema.$ref, root);
  if (schema.allOf?.length === 1 && schema.allOf[0].$ref) {
    const resolved = resolveRef(schema.allOf[0].$ref, root);
    const { allOf: _, ...rest } = schema;
    return { ...resolved, ...rest };
  }
  // Unwrap oneOf with a single variant (discriminated union with one model type)
  if (schema.oneOf?.length === 1) {
    const { oneOf: _, discriminator: __, ...rest } = schema;
    return { ...resolve(schema.oneOf[0], root), ...rest };
  }
  // Unwrap anyOf with one non-null variant (Pydantic Optional[T] → anyOf: [$ref, {type:null}])
  if (schema.anyOf) {
    const nonNull = (schema.anyOf as any[]).filter((v: any) => v.type !== "null");
    if (nonNull.length === 1) {
      const { anyOf: _, ...rest } = schema;
      return { ...resolve(nonNull[0], root), ...rest };
    }
  }
  return schema;
}

export function DynForm({ schema, value, onChange, rootSchema, path = "" }: DynFormProps) {
  const root = rootSchema ?? schema;
  const resolved = resolve(schema, root);
  const properties = resolved.properties ?? {};
  const keys = Object.keys(properties);

  if (!keys.length) {
    return <p class="lzw-muted">No configuration options.</p>;
  }

  return (
    <div class="lzw-dynform">
      {keys.map((key) => {
        const fieldSchema = resolve(properties[key], root);
        const fieldPath = path ? `${path}.${key}` : key;
        return (
          <DynField
            key={fieldPath}
            name={key}
            schema={fieldSchema}
            rootSchema={root}
            value={value[key]}
            onChange={(v) => onChange({ ...value, [key]: v })}
          />
        );
      })}
    </div>
  );
}

interface DynFieldProps {
  name: string;
  schema: Record<string, any>;
  rootSchema: Record<string, any>;
  value: any;
  onChange: (value: any) => void;
}

function DynField({ name, schema, rootSchema, value, onChange }: DynFieldProps) {
  const resolved = resolve(schema, rootSchema);
  const type = resolved.type;
  const title = resolved.title ?? name;
  const description = resolved.description;
  const defaultVal = resolved.default;
  const currentVal = value ?? defaultVal;

  // Const field → read-only badge (fixed by schema, not user-editable)
  if (resolved.const !== undefined) {
    return (
      <div class="lzw-form-row" title={description}>
        <label class="lzw-label">{title}</label>
        <span class="lzw-tag lzw-tag--muted">{String(resolved.const)}</span>
      </div>
    );
  }

  // Nested object
  if (type === "object" && resolved.properties) {
    return (
      <div class="lzw-dynform__section">
        <div class="lzw-dynform__section-title">{title}</div>
        <div class="lzw-dynform__section-body">
          {Object.keys(resolved.properties).map((k) => {
            const childSchema = resolve(resolved.properties[k], rootSchema);
            return (
              <DynField
                key={k}
                name={k}
                schema={childSchema}
                rootSchema={rootSchema}
                value={(currentVal ?? {})[k]}
                onChange={(v) =>
                  onChange({ ...(currentVal ?? {}), [k]: v })
                }
              />
            );
          })}
        </div>
      </div>
    );
  }

  // Enum (string + enum)
  if (resolved.enum) {
    return (
      <div class="lzw-form-row" title={description}>
        <label class="lzw-label">{title}</label>
        <select
          class="lzw-select"
          value={currentVal ?? ""}
          onChange={(e) => onChange((e.target as HTMLSelectElement).value)}
        >
          {resolved.enum.map((opt: string) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  // Boolean
  if (type === "boolean") {
    return (
      <div class="lzw-form-row" title={description}>
        <label class="lzw-label">{title}</label>
        <label class="lzw-toggle">
          <input
            type="checkbox"
            checked={currentVal ?? false}
            onChange={(e) =>
              onChange((e.target as HTMLInputElement).checked)
            }
          />
          <span class="lzw-toggle__slider" />
        </label>
      </div>
    );
  }

  // Number / Integer
  if (type === "number" || type === "integer") {
    return (
      <div class="lzw-form-row" title={description}>
        <label class="lzw-label">{title}</label>
        <NumericStepper
          value={currentVal ?? undefined}
          step={type === "integer" ? 1 : undefined}
          onChange={(v) => onChange(v)}
        />
      </div>
    );
  }

  // Free-form object with additionalProperties (e.g. model.params)
  if (type === "object" && resolved.additionalProperties) {
    return (
      <div class="lzw-form-row" title={description} style="align-items:flex-start">
        <label class="lzw-label">{title}</label>
        <KVEditor value={currentVal ?? {}} onChange={onChange} />
      </div>
    );
  }

  // Array fields
  if (type === "array" && resolved.items) {
    const itemSchema = resolve(resolved.items, rootSchema);
    if (itemSchema.enum) {
      // Array with enum items → checkbox group
      const selected: string[] = currentVal ?? [];
      return (
        <div class="lzw-form-row" title={description} style="align-items:flex-start">
          <label class="lzw-label">{title}</label>
          <div class="lzw-checkbox-group">
            {itemSchema.enum.map((opt: string) => (
              <label key={opt}>
                <input
                  type="checkbox"
                  checked={selected.includes(opt)}
                  onChange={(e) => {
                    const checked = (e.target as HTMLInputElement).checked;
                    onChange(
                      checked
                        ? [...selected, opt]
                        : selected.filter((v: string) => v !== opt),
                    );
                  }}
                />
                {opt}
              </label>
            ))}
          </div>
        </div>
      );
    }
    // Array without enum → tag input
    return (
      <div class="lzw-form-row" title={description} style="align-items:flex-start">
        <label class="lzw-label">{title}</label>
        <TagInput value={currentVal ?? []} onChange={onChange} />
      </div>
    );
  }

  // anyOf / oneOf with null (optional field)
  if (resolved.anyOf || resolved.oneOf) {
    const variants = (resolved.anyOf ?? resolved.oneOf) as any[];
    const nonNull = variants.filter((v: any) => v.type !== "null");
    if (nonNull.length === 1) {
      return (
        <DynField
          name={name}
          schema={{ ...nonNull[0], title: resolved.title, description: resolved.description }}
          rootSchema={rootSchema}
          value={value}
          onChange={onChange}
        />
      );
    }
  }

  // Fallback: string input
  return (
    <div class="lzw-form-row" title={description}>
      <label class="lzw-label">{title}</label>
      <input
        class="lzw-input"
        type="text"
        value={currentVal ?? ""}
        onChange={(e) => onChange((e.target as HTMLInputElement).value)}
      />
    </div>
  );
}
