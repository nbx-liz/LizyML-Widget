/**
 * DynForm — renders form fields from a JSON Schema.
 *
 * Handles: number, integer, boolean, string+enum, string, object (nested).
 * Resolves $defs references from Pydantic v2 schemas.
 */
import { useCallback } from "preact/hooks";

interface DynFormProps {
  schema: Record<string, any>;
  value: Record<string, any>;
  onChange: (value: Record<string, any>) => void;
  path?: string;
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
  return schema;
}

export function DynForm({ schema, value, onChange, path = "" }: DynFormProps) {
  const root = schema;
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
        <input
          type="checkbox"
          checked={currentVal ?? false}
          onChange={(e) =>
            onChange((e.target as HTMLInputElement).checked)
          }
        />
      </div>
    );
  }

  // Number / Integer
  if (type === "number" || type === "integer") {
    return (
      <div class="lzw-form-row" title={description}>
        <label class="lzw-label">{title}</label>
        <input
          class="lzw-input"
          type="number"
          step={type === "integer" ? 1 : "any"}
          value={currentVal ?? ""}
          onChange={(e) => {
            const raw = (e.target as HTMLInputElement).value;
            if (raw === "") {
              onChange(undefined);
            } else {
              onChange(
                type === "integer" ? parseInt(raw) : parseFloat(raw),
              );
            }
          }}
        />
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
