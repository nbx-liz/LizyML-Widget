/**
 * Shared schema resolution and config diff utilities for Config sub-tabs.
 */

/** Schema top-level keys managed by Data Tab — hidden from Config Tab. */
const DATA_TAB_KEYS = ["data", "features", "split"];

/** Property names that must never be traversed during $ref resolution. */
const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);

/** Resolve a $ref like "#/$defs/FooBar" against the root schema. */
function resolveRef(
  ref: string,
  root: Record<string, any>,
): Record<string, any> {
  const parts = ref.replace("#/", "").split("/");
  let node: any = root;
  for (const p of parts) {
    if (FORBIDDEN_KEYS.has(p)) return {};
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
export function getSectionSchema(
  rootSchema: Record<string, any>,
  key: string,
): Record<string, any> | null {
  const resolved = resolveSchema(rootSchema, rootSchema);
  const props = resolved.properties ?? {};
  if (!props[key]) return null;
  return resolveSchema(props[key], rootSchema);
}

/** Get schema keys not in known sections or data tab keys. */
export function getUnknownKeys(rootSchema: Record<string, any>, sectionKeys: string[]): string[] {
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

/** Build patch ops from old and new config objects (top-level keys only). */
export function diffToPatchOps(
  oldConfig: Record<string, any>,
  newConfig: Record<string, any>,
): Array<{ op: string; path: string; value: any }> {
  const ops: Array<{ op: string; path: string; value: any }> = [];

  for (const key of Object.keys(newConfig)) {
    const oldVal = oldConfig[key];
    const newVal = newConfig[key];

    if (JSON.stringify(oldVal) !== JSON.stringify(newVal)) {
      ops.push({ op: "set", path: key, value: newVal });
    }
  }

  // Check for removed keys
  for (const key of Object.keys(oldConfig)) {
    if (!(key in newConfig)) {
      ops.push({ op: "unset", path: key, value: null });
    }
  }

  return ops;
}
