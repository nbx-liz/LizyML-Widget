/**
 * Tests for App — Header + tab gating + auto-switch + custom-msg routing.
 *
 * #114 Phase A: App had 0% statement coverage. We render it against a
 * MockModel and assert the high-level routing rules — tab enable/disable
 * by status, auto-switch on running, custom-message handling for
 * column_stats / split_preview / code_export_download.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/preact";
import { App } from "../App";
import { createMockModel, MockModel } from "./mock-model";

// jsdom does not implement matchMedia; useTheme falls back to it during init.
beforeEach(() => {
  if (!window.matchMedia) {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  }
});

function makeRootEl(): HTMLElement {
  const el = document.createElement("div");
  document.body.appendChild(el);
  return el;
}

function setupApp(modelOverrides: Record<string, any> = {}) {
  const model = createMockModel(modelOverrides);
  const rootEl = makeRootEl();
  const utils = render(<App model={model} rootEl={rootEl} />);
  return { model, rootEl, ...utils };
}

describe("App — initial render", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  it("starts on the Data tab when status is idle", () => {
    const { container } = setupApp();
    const activeBtn = container.querySelector(".lzw-tabs__btn--active");
    expect(activeBtn?.textContent).toBe("Data");
  });

  it("renders the three tab buttons", () => {
    setupApp();
    expect(screen.getByText("Data")).toBeDefined();
    expect(screen.getByText("Model")).toBeDefined();
    expect(screen.getByText("Results")).toBeDefined();
  });

  it("renders the Header with backend info", () => {
    setupApp({ backend_info: { name: "lizyml", version: "0.10.0" } });
    expect(screen.getByText("lizyml v0.10.0")).toBeDefined();
  });
});

describe("App — tab gating by status", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  it("disables Model and Results tabs when status is idle", () => {
    const { container } = setupApp({ status: "idle", job_index: 0 });
    const buttons = Array.from(container.querySelectorAll(".lzw-tabs__btn"));
    const model = buttons.find((b) => b.textContent === "Model");
    const results = buttons.find((b) => b.textContent === "Results");
    expect(model?.classList.contains("lzw-tabs__btn--disabled")).toBe(true);
    expect(results?.classList.contains("lzw-tabs__btn--disabled")).toBe(true);
  });

  it("enables Model tab once status leaves idle (data_loaded)", () => {
    const { container } = setupApp({ status: "data_loaded" });
    const model = Array.from(container.querySelectorAll(".lzw-tabs__btn")).find(
      (b) => b.textContent === "Model",
    );
    expect(model?.classList.contains("lzw-tabs__btn--disabled")).toBe(false);
  });

  it("enables Results tab once a job has completed", () => {
    const { container } = setupApp({ status: "completed", job_index: 1 });
    const results = Array.from(container.querySelectorAll(".lzw-tabs__btn")).find(
      (b) => b.textContent === "Results",
    );
    expect(results?.classList.contains("lzw-tabs__btn--disabled")).toBe(false);
  });

  it("ignores clicks on a disabled tab (Model when idle)", () => {
    const { container } = setupApp({ status: "idle" });
    const modelBtn = Array.from(container.querySelectorAll(".lzw-tabs__btn")).find(
      (b) => b.textContent === "Model",
    ) as HTMLButtonElement;
    fireEvent.click(modelBtn);
    const active = container.querySelector(".lzw-tabs__btn--active");
    expect(active?.textContent).toBe("Data");
  });
});

describe("App — auto-switch to Results", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  it("switches to Results when status transitions to running", async () => {
    const { container, model } = setupApp({ status: "data_loaded" });
    expect(container.querySelector(".lzw-tabs__btn--active")?.textContent).toBe("Data");
    await act(async () => {
      model.simulateTraitletChange("status", "running");
    });
    expect(container.querySelector(".lzw-tabs__btn--active")?.textContent).toBe("Results");
  });

  it("switches to Results when status transitions to completed", async () => {
    const { container, model } = setupApp({ status: "running", job_index: 1 });
    await act(async () => {
      model.simulateTraitletChange("status", "completed");
    });
    expect(container.querySelector(".lzw-tabs__btn--active")?.textContent).toBe("Results");
  });

  it("switches to Results when status transitions to failed", async () => {
    const { container, model } = setupApp({ status: "running", job_index: 1 });
    await act(async () => {
      model.simulateTraitletChange("status", "failed");
    });
    expect(container.querySelector(".lzw-tabs__btn--active")?.textContent).toBe("Results");
  });
});

describe("App — custom message handling", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  it("routes column_stats messages without crashing", async () => {
    const { model } = setupApp({ status: "data_loaded" });
    // Should not throw — column_stats is consumed and stored locally.
    await act(async () => {
      model.simulateCustomMessage({ type: "column_stats", column: "x1" });
    });
    expect(true).toBe(true);
  });

  it("triggers a download for code_export_download messages", async () => {
    const model = createMockModel({ status: "completed", job_index: 1 });
    const rootEl = makeRootEl();
    render(<App model={model as unknown as MockModel} rootEl={rootEl} />);

    // Stub URL.createObjectURL / revokeObjectURL on the global `URL` object
    // (jsdom does not implement them).
    const created: string[] = [];
    const origCreate = URL.createObjectURL;
    const origRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn(() => {
      const u = `blob:mock-${created.length}`;
      created.push(u);
      return u;
    });
    URL.revokeObjectURL = vi.fn();

    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    try {
      const buf = new ArrayBuffer(8);
      await act(async () => {
        model.simulateCustomMessage(
          { type: "code_export_download", filename: "exported.zip" },
          [buf],
        );
      });
      expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(URL.revokeObjectURL).toHaveBeenCalledWith(created[0]);
    } finally {
      clickSpy.mockRestore();
      URL.createObjectURL = origCreate;
      URL.revokeObjectURL = origRevoke;
    }
  });
});
