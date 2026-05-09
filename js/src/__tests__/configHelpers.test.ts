/**
 * Tests for configHelpers — pure schema utilities.
 *
 * #114 Phase A: configHelpers had 48% statement coverage; the helpers are
 * pure and trivial to exercise so they should be near-100%.
 */
import { describe, it, expect } from "vitest";
import { getSectionSchema, getUnknownKeys, diffToPatchOps } from "../tabs/configHelpers";

describe("getSectionSchema", () => {
  it("returns null when the requested section is missing", () => {
    const root = { properties: { other: { type: "object" } } };
    expect(getSectionSchema(root, "model")).toBeNull();
  });

  it("returns the section sub-schema for a top-level property", () => {
    const root = {
      properties: {
        model: { type: "object", properties: { name: { type: "string" } } },
      },
    };
    const result = getSectionSchema(root, "model");
    expect(result).not.toBeNull();
    expect(result?.type).toBe("object");
    expect(result?.properties?.name).toEqual({ type: "string" });
  });

  it("resolves a $ref pointer at the root level", () => {
    const root = {
      $ref: "#/$defs/Root",
      $defs: {
        Root: {
          properties: { model: { type: "object" } },
        },
      },
    };
    expect(getSectionSchema(root, "model")).toEqual({ type: "object" });
  });

  it("resolves a $ref on a section property", () => {
    const root = {
      properties: { model: { $ref: "#/$defs/Model" } },
      $defs: { Model: { type: "object", title: "Model" } },
    };
    const result = getSectionSchema(root, "model");
    expect(result?.title).toBe("Model");
  });

  it("collapses a single-element allOf wrapping a $ref", () => {
    const root = {
      properties: {
        model: { allOf: [{ $ref: "#/$defs/Model" }], description: "wrapped" },
      },
      $defs: { Model: { type: "object", title: "Model" } },
    };
    const result = getSectionSchema(root, "model");
    // Outer description must override inner schema (rest spread last)
    expect(result?.description).toBe("wrapped");
    expect(result?.title).toBe("Model");
  });

  it("ignores forbidden $ref segments (__proto__, constructor, prototype)", () => {
    const root = {
      properties: { model: { $ref: "#/__proto__/polluted" } },
    };
    expect(getSectionSchema(root, "model")).toEqual({});
  });
});

describe("getUnknownKeys", () => {
  it("excludes known sections, data tab keys, tuning, config_version, and task", () => {
    const root = {
      properties: {
        model: {},
        training: {},
        data: {},
        features: {},
        split: {},
        tuning: {},
        config_version: {},
        task: {},
        custom_extra: {},
      },
    };
    expect(getUnknownKeys(root, ["model", "training"])).toEqual(["custom_extra"]);
  });

  it("returns an empty list when nothing is unknown", () => {
    const root = { properties: { model: {}, data: {} } };
    expect(getUnknownKeys(root, ["model"])).toEqual([]);
  });

  it("handles a missing properties block gracefully", () => {
    expect(getUnknownKeys({}, [])).toEqual([]);
  });
});

describe("diffToPatchOps", () => {
  it("emits a 'set' op when a value changes", () => {
    const ops = diffToPatchOps({ a: 1 }, { a: 2 });
    expect(ops).toEqual([{ op: "set", path: "a", value: 2 }]);
  });

  it("emits 'set' for a newly added key", () => {
    const ops = diffToPatchOps({}, { a: 1 });
    expect(ops).toEqual([{ op: "set", path: "a", value: 1 }]);
  });

  it("emits 'unset' for a removed key", () => {
    const ops = diffToPatchOps({ a: 1, b: 2 }, { a: 1 });
    expect(ops).toEqual([{ op: "unset", path: "b", value: null }]);
  });

  it("returns no ops when configs are deeply equal", () => {
    const a = { a: 1, b: { c: [1, 2] } };
    const b = { a: 1, b: { c: [1, 2] } };
    expect(diffToPatchOps(a, b)).toEqual([]);
  });

  it("treats deep object differences as a single top-level set", () => {
    const ops = diffToPatchOps({ b: { c: 1 } }, { b: { c: 2 } });
    expect(ops).toEqual([{ op: "set", path: "b", value: { c: 2 } }]);
  });
});
