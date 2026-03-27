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
