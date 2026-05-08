/**
 * Tests for Header — backend badge + status indicator + theme toggle.
 *
 * #114 Phase A: Header had 0% coverage but is small and pure.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { Header } from "../components/Header";

const baseProps = {
  backendInfo: { name: "lizyml", version: "0.10.0" },
  status: "idle",
  theme: "light" as const,
  onToggleTheme: vi.fn(),
};

describe("Header — backend badge", () => {
  it("renders the backend name and version when both are present", () => {
    render(<Header {...baseProps} />);
    expect(screen.getByText("lizyml v0.10.0")).toBeDefined();
  });

  it("hides the backend badge when name is missing", () => {
    const { container } = render(
      <Header {...baseProps} backendInfo={{ name: undefined, version: "0.10.0" }} />,
    );
    expect(container.textContent).not.toContain("v0.10.0");
  });
});

describe("Header — status indicator", () => {
  it.each([
    ["idle", "Idle"],
    ["data_loaded", "Data Loaded"],
    ["running", "Running"],
    ["completed", "Completed"],
    ["failed", "Failed"],
  ])("renders the %s status label", (status, expected) => {
    render(<Header {...baseProps} status={status} />);
    expect(screen.getByText(new RegExp(expected))).toBeDefined();
  });

  it("falls back to the idle badge for an unknown status", () => {
    render(<Header {...baseProps} status="surprise" />);
    expect(screen.getByText(/Idle/)).toBeDefined();
  });

  it("uses the success class for completed", () => {
    const { container } = render(<Header {...baseProps} status="completed" />);
    const badge = container.querySelector(".lzw-badge--success");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toContain("Completed");
  });
});

describe("Header — theme toggle", () => {
  it("displays the moon glyph when light is active", () => {
    render(<Header {...baseProps} theme="light" />);
    expect(screen.getByRole("button").textContent).toContain("🌙");
  });

  it("displays the sun glyph when dark is active", () => {
    render(<Header {...baseProps} theme="dark" />);
    expect(screen.getByRole("button").textContent).toContain("☀️");
  });

  it("aria-pressed reflects dark mode", () => {
    const { rerender } = render(<Header {...baseProps} theme="light" />);
    expect(screen.getByRole("button").getAttribute("aria-pressed")).toBe("false");
    rerender(<Header {...baseProps} theme="dark" />);
    expect(screen.getByRole("button").getAttribute("aria-pressed")).toBe("true");
  });

  it("invokes onToggleTheme on click", () => {
    const onToggleTheme = vi.fn();
    render(<Header {...baseProps} onToggleTheme={onToggleTheme} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onToggleTheme).toHaveBeenCalledTimes(1);
  });
});
