/**
 * Tests for useTheme hook — theme toggle and data attribute management.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/preact";
import { useTheme } from "../hooks/useTheme";

// jsdom doesn't provide matchMedia — provide a minimal mock
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false, // default: light mode
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

describe("useTheme", () => {
  let rootEl: HTMLDivElement;

  beforeEach(() => {
    rootEl = document.createElement("div");
    document.body.appendChild(rootEl);
  });

  it("defaults to auto mode resolving to light (jsdom default)", () => {
    const { result } = renderHook(() => useTheme(rootEl));
    expect(result.current.mode).toBe("auto");
    // jsdom's matchMedia defaults to no-match → light
    expect(result.current.resolved).toBe("light");
  });

  it("sets data-lzw-theme attribute on root element", () => {
    renderHook(() => useTheme(rootEl));
    expect(rootEl.getAttribute("data-lzw-theme")).toBe("light");
  });

  it("toggles from auto to opposite of system theme", () => {
    const { result } = renderHook(() => useTheme(rootEl));

    // jsdom system theme = light, so toggle should go to dark
    act(() => { result.current.toggle(); });
    expect(result.current.resolved).toBe("dark");
    expect(rootEl.getAttribute("data-lzw-theme")).toBe("dark");
  });

  it("toggles between light and dark", () => {
    const { result } = renderHook(() => useTheme(rootEl));

    // auto → dark (since system is light)
    act(() => { result.current.toggle(); });
    expect(result.current.resolved).toBe("dark");

    // dark → light
    act(() => { result.current.toggle(); });
    expect(result.current.resolved).toBe("light");
  });

  it("handles null rootEl without error", () => {
    const { result } = renderHook(() => useTheme(null));
    expect(result.current.resolved).toBe("light");

    // Toggle should not throw (mode changes, but resolved stays
    // because the effect that updates resolved skips when rootEl is null)
    act(() => { result.current.toggle(); });
    expect(result.current.mode).toBe("dark");
  });
});
