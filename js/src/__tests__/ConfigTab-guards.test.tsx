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
    expect(fitBtn.textContent).toContain("Fit");
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

import { fireEvent } from "@testing-library/preact";

describe("ConfigTab — Fit/Tune subtab switching (#134)", () => {
  it("renders only the Fit primary button on initial mount (Fit subtab default)", () => {
    const { container } = render(<ConfigTab {...defaultProps} status="completed" />);
    const primaries = container.querySelectorAll(".lzw-btn--primary");
    expect(primaries.length).toBe(1);
    expect((primaries[0] as HTMLButtonElement).textContent).toContain("Fit");
  });

  it("clicking the Tune subtab swaps the primary button to Tune", () => {
    const { container } = render(<ConfigTab {...defaultProps} status="completed" />);
    const tuneSubtabBtn = Array.from(
      container.querySelectorAll(".lzw-subtabs__btn"),
    ).find((b) => b.textContent === "Tune") as HTMLButtonElement;
    expect(tuneSubtabBtn).toBeDefined();
    fireEvent.click(tuneSubtabBtn);
    const primary = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    expect(primary.textContent).toContain("Tune");
  });

  it("Tune primary button is disabled while a job is running (B-5 parity)", () => {
    const { container } = render(
      <ConfigTab
        {...defaultProps}
        status="running"
        config={{
          model: { name: "lgbm", params: {} },
          tuning: { optuna: { space: { learning_rate: { type: "float", low: 0.01, high: 0.1 } } } },
        }}
      />,
    );
    // Switch to Tune subtab
    const tuneBtn = Array.from(
      container.querySelectorAll(".lzw-subtabs__btn"),
    ).find((b) => b.textContent === "Tune") as HTMLButtonElement;
    fireEvent.click(tuneBtn);
    const primary = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    expect(primary.disabled).toBe(true);
    expect(primary.textContent).toContain("Running...");
  });

  it("Tune primary button is disabled when no search-space param is configured", () => {
    const { container } = render(<ConfigTab {...defaultProps} status="completed" />);
    const tuneBtn = Array.from(
      container.querySelectorAll(".lzw-subtabs__btn"),
    ).find((b) => b.textContent === "Tune") as HTMLButtonElement;
    fireEvent.click(tuneBtn);
    const primary = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    // No tuning.optuna.space populated -> Tune is disabled
    expect(primary.disabled).toBe(true);
  });

  it("Tune primary button is enabled when both data and a search param exist", () => {
    const { container } = render(
      <ConfigTab
        {...defaultProps}
        status="completed"
        config={{
          model: { name: "lgbm", params: {} },
          tuning: { optuna: { space: { learning_rate: { type: "float", low: 0.01, high: 0.1 } } } },
        }}
      />,
    );
    const tuneBtn = Array.from(
      container.querySelectorAll(".lzw-subtabs__btn"),
    ).find((b) => b.textContent === "Tune") as HTMLButtonElement;
    fireEvent.click(tuneBtn);
    const primary = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    expect(primary.disabled).toBe(false);
    expect(primary.textContent).toContain("Tune");
  });

  it("clicking the Fit primary button dispatches the 'fit' action", () => {
    const sendAction = vi.fn();
    const { container } = render(
      <ConfigTab {...defaultProps} status="completed" sendAction={sendAction} />,
    );
    const primary = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    fireEvent.click(primary);
    expect(sendAction).toHaveBeenCalledWith("fit");
  });

  it("clicking the Tune primary button dispatches the 'tune' action", () => {
    const sendAction = vi.fn();
    const { container } = render(
      <ConfigTab
        {...defaultProps}
        status="completed"
        sendAction={sendAction}
        config={{
          model: { name: "lgbm", params: {} },
          tuning: { optuna: { space: { learning_rate: { type: "float", low: 0.01, high: 0.1 } } } },
        }}
      />,
    );
    const tuneBtn = Array.from(
      container.querySelectorAll(".lzw-subtabs__btn"),
    ).find((b) => b.textContent === "Tune") as HTMLButtonElement;
    fireEvent.click(tuneBtn);
    const primary = container.querySelector(".lzw-btn--primary") as HTMLButtonElement;
    fireEvent.click(primary);
    expect(sendAction).toHaveBeenCalledWith("tune");
  });
});
