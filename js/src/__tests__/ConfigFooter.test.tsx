/**
 * Tests for ConfigFooter — Import/Export YAML buttons + Raw Config modal.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { ConfigFooter } from "../components/ConfigFooter";

const defaultProps = {
  sendAction: vi.fn(),
  rawYaml: null as string | null,
  setRawYaml: vi.fn(),
  yamlExportCount: 0,
};

describe("ConfigFooter — Export YAML button", () => {
  it("fires sendAction('export_yaml') on click and shows 'Exporting...'", () => {
    const sendAction = vi.fn();
    render(<ConfigFooter {...defaultProps} sendAction={sendAction} />);
    const btn = screen.getByText("Export YAML");
    fireEvent.click(btn);
    expect(sendAction).toHaveBeenCalledWith("export_yaml");
    // After click, button text changes to "Exporting..."
    expect(screen.getByText("Exporting...")).toBeDefined();
    expect((screen.getByText("Exporting...") as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("ConfigFooter — Raw Config button", () => {
  it("fires sendAction('raw_config') on click", () => {
    const sendAction = vi.fn();
    render(<ConfigFooter {...defaultProps} sendAction={sendAction} />);
    const btn = screen.getByText("Raw Config");
    fireEvent.click(btn);
    expect(sendAction).toHaveBeenCalledWith("raw_config");
  });
});

describe("ConfigFooter — Raw Config modal", () => {
  it("shows modal with YAML content when rawYaml is set", () => {
    const yaml = "model:\n  name: lgbm";
    render(<ConfigFooter {...defaultProps} rawYaml={yaml} />);
    // The YAML is rendered inside a <pre> element
    const pre = document.querySelector(".lzw-pre") as HTMLElement;
    expect(pre).not.toBeNull();
    expect(pre.textContent).toContain("model:");
    expect(screen.getByText("Close")).toBeDefined();
  });

  it("calls setRawYaml(null) when Close button is clicked", () => {
    const setRawYaml = vi.fn();
    render(<ConfigFooter {...defaultProps} rawYaml="test: true" setRawYaml={setRawYaml} />);
    fireEvent.click(screen.getByText("Close"));
    expect(setRawYaml).toHaveBeenCalledWith(null);
  });

  it("calls setRawYaml(null) when overlay is clicked", () => {
    const setRawYaml = vi.fn();
    const { container } = render(
      <ConfigFooter {...defaultProps} rawYaml="test: true" setRawYaml={setRawYaml} />,
    );
    const overlay = container.querySelector(".lzw-modal-overlay") as HTMLElement;
    fireEvent.click(overlay);
    expect(setRawYaml).toHaveBeenCalledWith(null);
  });

  it("does not show modal when rawYaml is null", () => {
    const { container } = render(<ConfigFooter {...defaultProps} rawYaml={null} />);
    expect(container.querySelector(".lzw-modal-overlay")).toBeNull();
  });
});

describe("ConfigFooter — Import YAML button", () => {
  it("renders Import YAML button", () => {
    render(<ConfigFooter {...defaultProps} />);
    expect(screen.getByText("Import YAML")).toBeDefined();
  });

  // Note: Testing file input behavior with FileReader is complex in jsdom.
  // The Import YAML button triggers a hidden file input click, and the
  // actual file reading is handled by FileReader.onload → sendAction("import_yaml").
  // This integration requires mocking FileReader which adds complexity
  // beyond unit-level testing.
});
