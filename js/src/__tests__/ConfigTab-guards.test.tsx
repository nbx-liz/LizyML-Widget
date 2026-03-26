/**
 * Tests for ConfigTab UI guards:
 * - B-5: Config form disabled during job execution
 * - Fit/Tune button disabled during running
 */
import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/preact";
import { ConfigTab } from "../tabs/ConfigTab";
import { createMockModel } from "./mock-model";

const minimalContract = {
  config_schema: { type: "object", properties: {} },
  ui_schema: { sections: [], option_sets: {}, parameter_hints: [], step_map: {}, conditional_visibility: {} },
  capabilities: {},
};

const defaultProps = {
  backendContract: minimalContract,
  config: { model: { name: "lgbm", params: {} } },
  dfInfo: { target: "y", task: "binary", shape: [100, 5] },
  sendAction: vi.fn(),
  model: createMockModel(),
};

describe("ConfigTab — running state guard (B-5)", () => {
  it("applies visual disabled style when status is 'running'", () => {
    const { container } = render(
      <ConfigTab {...defaultProps} status="running" />,
    );

    // The config body should have pointer-events: none and opacity: 0.6
    const body = container.querySelector(".lzw-config-tab__body") as HTMLElement;
    expect(body).not.toBeNull();
    expect(body.style.pointerEvents).toBe("none");
    expect(body.style.opacity).toBe("0.6");
  });

  it("does not apply disabled style when status is 'completed'", () => {
    const { container } = render(
      <ConfigTab {...defaultProps} status="completed" />,
    );

    const body = container.querySelector(".lzw-config-tab__body") as HTMLElement;
    expect(body).not.toBeNull();
    // Should NOT have disabled styling
    expect(body.style.pointerEvents).not.toBe("none");
  });

  it("disables Fit button when status is 'running'", () => {
    const { container } = render(<ConfigTab {...defaultProps} status="running" />);

    // The primary action button (not the sub-tab selector)
    const fitBtn = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    expect(fitBtn).not.toBeNull();
    expect(fitBtn.disabled).toBe(true);
    expect(fitBtn.textContent).toContain("Running...");
  });

  it("enables Fit button when status is 'completed'", () => {
    const { container } = render(<ConfigTab {...defaultProps} status="completed" />);

    const fitBtn = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    expect(fitBtn).not.toBeNull();
    expect(fitBtn.disabled).toBe(false);
    expect(fitBtn.textContent).toBe("Fit");
  });

  it("disables Fit button when status is 'idle' (no data)", () => {
    const { container } = render(
      <ConfigTab {...defaultProps} status="idle" dfInfo={{}} />,
    );

    const fitBtn = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    expect(fitBtn).not.toBeNull();
    expect(fitBtn.disabled).toBe(true);
  });
});
