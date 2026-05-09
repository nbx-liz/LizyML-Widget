/**
 * Tests for DynForm — dynamic form rendering from JSON Schema.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { DynForm } from "../components/DynForm";

describe("DynForm — enum (select dropdown)", () => {
  it("renders a select with options from schema enum", () => {
    const schema = {
      type: "object",
      properties: {
        color: { title: "Color", type: "string", enum: ["red", "green", "blue"] },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm schema={schema} value={{ color: "green" }} onChange={onChange} />,
    );
    const select = container.querySelector(".lzw-select") as HTMLSelectElement;
    expect(select).not.toBeNull();
    expect(select.value).toBe("green");
    const options = select.querySelectorAll("option");
    expect(options.length).toBe(3);
  });

  it("fires onChange when select value changes", () => {
    const schema = {
      type: "object",
      properties: {
        color: { title: "Color", type: "string", enum: ["red", "green", "blue"] },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm schema={schema} value={{ color: "red" }} onChange={onChange} />,
    );
    const select = container.querySelector(".lzw-select") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "blue" } });
    expect(onChange).toHaveBeenCalledWith({ color: "blue" });
  });
});

describe("DynForm — boolean (checkbox)", () => {
  it("renders a checkbox in checked state", () => {
    const schema = {
      type: "object",
      properties: {
        enabled: { title: "Enabled", type: "boolean" },
      },
    };
    const { container } = render(
      <DynForm schema={schema} value={{ enabled: true }} onChange={vi.fn()} />,
    );
    const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
    expect(checkbox).not.toBeNull();
    expect(checkbox.checked).toBe(true);
  });

  it("fires onChange with toggled value", () => {
    const schema = {
      type: "object",
      properties: {
        enabled: { title: "Enabled", type: "boolean" },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm schema={schema} value={{ enabled: true }} onChange={onChange} />,
    );
    const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith({ enabled: false });
  });
});

describe("DynForm — numeric (NumericStepper)", () => {
  it("renders a numeric stepper with current value", () => {
    const schema = {
      type: "object",
      properties: {
        count: { title: "Count", type: "integer" },
      },
    };
    const { container } = render(
      <DynForm schema={schema} value={{ count: 5 }} onChange={vi.fn()} />,
    );
    const input = container.querySelector('input[type="number"]') as HTMLInputElement;
    expect(input).not.toBeNull();
    expect(input.value).toBe("5");
  });

  it("fires onChange with parsed number on stepper increment", () => {
    const schema = {
      type: "object",
      properties: {
        count: { title: "Count", type: "integer" },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm schema={schema} value={{ count: 5 }} onChange={onChange} />,
    );
    // Click the increment button (+)
    const buttons = container.querySelectorAll(".lzw-stepper__btn");
    const incrementBtn = buttons[1]; // second button is +
    fireEvent.click(incrementBtn);
    expect(onChange).toHaveBeenCalledWith({ count: 6 });
  });

  it("fires onChange with parsed number on stepper decrement", () => {
    const schema = {
      type: "object",
      properties: {
        count: { title: "Count", type: "integer" },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm schema={schema} value={{ count: 5 }} onChange={onChange} />,
    );
    const buttons = container.querySelectorAll(".lzw-stepper__btn");
    const decrementBtn = buttons[0]; // first button is -
    fireEvent.click(decrementBtn);
    expect(onChange).toHaveBeenCalledWith({ count: 4 });
  });
});

describe("DynForm — nested object", () => {
  it("renders nested object fields", () => {
    const schema = {
      type: "object",
      properties: {
        settings: {
          title: "Settings",
          type: "object",
          properties: {
            verbose: { title: "Verbose", type: "boolean" },
          },
        },
      },
    };
    const { container } = render(
      <DynForm schema={schema} value={{ settings: { verbose: true } }} onChange={vi.fn()} />,
    );
    expect(screen.getByText("Settings")).toBeDefined();
    const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
  });
});

describe("DynForm — string (text input)", () => {
  it("renders a text input for string fields", () => {
    const schema = {
      type: "object",
      properties: {
        name: { title: "Name", type: "string" },
      },
    };
    const { container } = render(
      <DynForm schema={schema} value={{ name: "hello" }} onChange={vi.fn()} />,
    );
    const input = container.querySelector('input[type="text"]') as HTMLInputElement;
    expect(input).not.toBeNull();
    expect(input.value).toBe("hello");
  });

  it("fires onChange on text input change", () => {
    const schema = {
      type: "object",
      properties: {
        name: { title: "Name", type: "string" },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm schema={schema} value={{ name: "hello" }} onChange={onChange} />,
    );
    const input = container.querySelector('input[type="text"]') as HTMLInputElement;
    fireEvent.change(input, { target: { value: "world" } });
    expect(onChange).toHaveBeenCalledWith({ name: "world" });
  });
});

describe("DynForm — const field", () => {
  it("renders const value as read-only badge", () => {
    const schema = {
      type: "object",
      properties: {
        version: { title: "Version", const: "1.0" },
      },
    };
    render(
      <DynForm schema={schema} value={{}} onChange={vi.fn()} />,
    );
    expect(screen.getByText("1.0")).toBeDefined();
  });
});

describe("DynForm — empty schema", () => {
  it("shows 'No configuration options' when no properties", () => {
    const schema = { type: "object", properties: {} };
    render(<DynForm schema={schema} value={{}} onChange={vi.fn()} />);
    expect(screen.getByText("No configuration options.")).toBeDefined();
  });
});

describe("DynForm — $ref resolution", () => {
  it("resolves $defs references correctly", () => {
    const schema = {
      type: "object",
      properties: {
        mode: { $ref: "#/$defs/ModeEnum" },
      },
      $defs: {
        ModeEnum: { title: "Mode", type: "string", enum: ["fast", "slow"] },
      },
    };
    const { container } = render(
      <DynForm schema={schema} value={{ mode: "fast" }} onChange={vi.fn()} />,
    );
    const select = container.querySelector(".lzw-select") as HTMLSelectElement;
    expect(select).not.toBeNull();
    expect(select.value).toBe("fast");
  });
});

describe("DynForm — array with enum (checkbox group)", () => {
  it("renders one checkbox per enum option, selected matches value", () => {
    const schema = {
      type: "object",
      properties: {
        metrics: {
          title: "Metrics",
          type: "array",
          items: { type: "string", enum: ["auc", "loss", "f1"] },
        },
      },
    };
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ metrics: ["auc", "f1"] }}
        onChange={vi.fn()}
      />,
    );
    const checks = container.querySelectorAll(
      ".lzw-checkbox-group input[type='checkbox']",
    );
    expect(checks.length).toBe(3);
    expect((checks[0] as HTMLInputElement).checked).toBe(true);   // auc
    expect((checks[1] as HTMLInputElement).checked).toBe(false);  // loss
    expect((checks[2] as HTMLInputElement).checked).toBe(true);   // f1
  });

  it("adds an enum entry when an unchecked checkbox is clicked", () => {
    const schema = {
      type: "object",
      properties: {
        metrics: {
          title: "Metrics",
          type: "array",
          items: { type: "string", enum: ["auc", "loss"] },
        },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ metrics: ["auc"] }}
        onChange={onChange}
      />,
    );
    const checks = container.querySelectorAll(
      ".lzw-checkbox-group input[type='checkbox']",
    );
    fireEvent.click(checks[1]); // loss
    expect(onChange).toHaveBeenCalledWith({ metrics: ["auc", "loss"] });
  });

  it("removes an enum entry when a checked checkbox is clicked", () => {
    const schema = {
      type: "object",
      properties: {
        metrics: {
          title: "Metrics",
          type: "array",
          items: { type: "string", enum: ["auc", "loss"] },
        },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ metrics: ["auc", "loss"] }}
        onChange={onChange}
      />,
    );
    const checks = container.querySelectorAll(
      ".lzw-checkbox-group input[type='checkbox']",
    );
    fireEvent.click(checks[0]); // uncheck auc
    expect(onChange).toHaveBeenCalledWith({ metrics: ["loss"] });
  });
});

describe("DynForm — array without enum (TagInput)", () => {
  it("renders existing tags and a placeholder when empty", () => {
    const schema = {
      type: "object",
      properties: {
        tags: { title: "Tags", type: "array", items: { type: "string" } },
      },
    };
    const { container } = render(
      <DynForm schema={schema} value={{ tags: ["a", "b"] }} onChange={vi.fn()} />,
    );
    const tags = container.querySelectorAll(".lzw-tag");
    expect(tags.length).toBe(2);
    expect((tags[0] as HTMLElement).textContent).toContain("a");
  });

  it("commits a new tag on Enter and clears the input", () => {
    const schema = {
      type: "object",
      properties: {
        tags: { title: "Tags", type: "array", items: { type: "string" } },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm schema={schema} value={{ tags: ["a"] }} onChange={onChange} />,
    );
    const input = container.querySelector(
      ".lzw-tag-input__field",
    ) as HTMLInputElement;
    fireEvent.input(input, { target: { value: "newTag" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith({ tags: ["a", "newTag"] });
  });

  it("removes a tag when its × button is clicked", () => {
    const schema = {
      type: "object",
      properties: {
        tags: { title: "Tags", type: "array", items: { type: "string" } },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ tags: ["a", "b"] }}
        onChange={onChange}
      />,
    );
    const removes = container.querySelectorAll(".lzw-tag__remove");
    fireEvent.click(removes[0]);
    expect(onChange).toHaveBeenCalledWith({ tags: ["b"] });
  });

  it("removes the last tag on Backspace when input is empty", () => {
    const schema = {
      type: "object",
      properties: {
        tags: { title: "Tags", type: "array", items: { type: "string" } },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ tags: ["a", "b"] }}
        onChange={onChange}
      />,
    );
    const input = container.querySelector(
      ".lzw-tag-input__field",
    ) as HTMLInputElement;
    fireEvent.keyDown(input, { key: "Backspace" });
    expect(onChange).toHaveBeenCalledWith({ tags: ["a"] });
  });
});

describe("DynForm — additionalProperties (KVEditor)", () => {
  it("renders one row per existing key/value pair", () => {
    const schema = {
      type: "object",
      properties: {
        params: {
          title: "Params",
          type: "object",
          additionalProperties: true,
        },
      },
    };
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ params: { lr: 0.01, max_depth: 6 } }}
        onChange={vi.fn()}
      />,
    );
    const rows = container.querySelectorAll(".lzw-kv-editor__row");
    expect(rows.length).toBe(2);
  });

  it("appends a new empty row when + Add is clicked", () => {
    const schema = {
      type: "object",
      properties: {
        params: {
          title: "Params",
          type: "object",
          additionalProperties: true,
        },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ params: { lr: 0.01 } }}
        onChange={onChange}
      />,
    );
    const addBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("+ Add"),
    ) as HTMLButtonElement;
    fireEvent.click(addBtn);
    expect(onChange).toHaveBeenCalledWith({ params: { lr: 0.01, "": "" } });
  });

  it("parses numeric strings via JSON when value changes", () => {
    const schema = {
      type: "object",
      properties: {
        params: {
          title: "Params",
          type: "object",
          additionalProperties: true,
        },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ params: { lr: 0.01 } }}
        onChange={onChange}
      />,
    );
    const inputs = container.querySelectorAll(
      ".lzw-kv-editor__row .lzw-input--sm",
    );
    // second input is the value field
    fireEvent.change(inputs[1], { target: { value: "0.05" } });
    expect(onChange).toHaveBeenCalledWith({ params: { lr: 0.05 } });
  });

  it("removes a row when × is clicked", () => {
    const schema = {
      type: "object",
      properties: {
        params: {
          title: "Params",
          type: "object",
          additionalProperties: true,
        },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ params: { a: 1, b: 2 } }}
        onChange={onChange}
      />,
    );
    const removes = container.querySelectorAll(".lzw-tag__remove");
    fireEvent.click(removes[0]);
    expect(onChange).toHaveBeenCalledWith({ params: { b: 2 } });
  });
});

describe("DynForm — nested object onChange", () => {
  it("propagates nested edits up to the top-level onChange", () => {
    const schema = {
      type: "object",
      properties: {
        settings: {
          title: "Settings",
          type: "object",
          properties: {
            verbose: { title: "Verbose", type: "boolean" },
            level: { title: "Level", type: "integer" },
          },
        },
      },
    };
    const onChange = vi.fn();
    const { container } = render(
      <DynForm
        schema={schema}
        value={{ settings: { verbose: false, level: 1 } }}
        onChange={onChange}
      />,
    );
    const checkbox = container.querySelector(
      'input[type="checkbox"]',
    ) as HTMLInputElement;
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith({
      settings: { verbose: true, level: 1 },
    });
  });
});

describe("DynForm — anyOf with null (Pydantic Optional)", () => {
  it("unwraps anyOf:[non-null, null] and renders the non-null variant", () => {
    const schema = {
      type: "object",
      properties: {
        seed: {
          title: "Seed",
          anyOf: [{ type: "integer" }, { type: "null" }],
        },
      },
    };
    const { container } = render(
      <DynForm schema={schema} value={{ seed: 42 }} onChange={vi.fn()} />,
    );
    const input = container.querySelector(
      'input[type="number"]',
    ) as HTMLInputElement;
    expect(input).not.toBeNull();
    expect(input.value).toBe("42");
  });
});
