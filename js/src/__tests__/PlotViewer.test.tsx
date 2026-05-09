/**
 * PlotViewer tests — exercise the Plotly-loader paths under jsdom.
 *
 * Plotly is loaded dynamically from a CDN URL in production. JSDOM cannot
 * resolve a remote ESM import, so we install a mock on `window.Plotly`
 * BEFORE rendering. The component's `getPlotly()` short-circuits to that
 * cached value and never reaches the dynamic import.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/preact";

import { PlotViewer } from "../components/PlotViewer";

interface PlotlyStub {
  newPlot: ReturnType<typeof vi.fn>;
}

function installPlotly(): PlotlyStub {
  const stub: PlotlyStub = { newPlot: vi.fn() };
  (window as any).Plotly = stub;
  return stub;
}

afterEach(() => {
  delete (window as any).Plotly;
  cleanup();
});

describe("PlotViewer — empty / placeholder paths", () => {
  it('renders the "No plot selected" placeholder when plotType is empty', () => {
    render(
      <PlotViewer
        plotType=""
        plots={{}}
        loading={{}}
        onRequest={vi.fn()}
      />,
    );
    expect(screen.getByText(/No plot selected/i)).toBeDefined();
  });

  it("renders the loading text when loading[plotType] is true and no spec is ready", () => {
    render(
      <PlotViewer
        plotType="learning-curve"
        plots={{}}
        loading={{ "learning-curve": true }}
        onRequest={vi.fn()}
      />,
    );
    expect(screen.getByText(/Loading plot/i)).toBeDefined();
  });
});

describe("PlotViewer — onRequest behaviour", () => {
  it("invokes onRequest when plotType becomes non-empty", () => {
    const onRequest = vi.fn();
    render(
      <PlotViewer
        plotType="roc-curve"
        plots={{}}
        loading={{}}
        onRequest={onRequest}
      />,
    );
    expect(onRequest).toHaveBeenCalledWith("roc-curve");
  });

  it("does not invoke onRequest when plotType is empty", () => {
    const onRequest = vi.fn();
    render(
      <PlotViewer plotType="" plots={{}} loading={{}} onRequest={onRequest} />,
    );
    expect(onRequest).not.toHaveBeenCalled();
  });
});

describe("PlotViewer — Plotly render paths", () => {
  beforeEach(() => {
    installPlotly();
  });

  it("calls Plotly.newPlot with the spec layout under the light theme (no DARK_LAYOUT overrides)", async () => {
    const stub = (window as any).Plotly as PlotlyStub;
    const spec = {
      data: [{ x: [1, 2, 3], y: [1, 4, 9], type: "scatter" }],
      layout: { title: "OK" },
    };
    render(
      <PlotViewer
        plotType="learning-curve"
        plots={{ "learning-curve": spec }}
        loading={{}}
        onRequest={vi.fn()}
        theme="light"
      />,
    );

    await waitFor(() => expect(stub.newPlot).toHaveBeenCalled());
    const [, dataArg, layoutArg] = stub.newPlot.mock.calls[0];
    expect(dataArg).toEqual(spec.data);
    expect(layoutArg).toMatchObject({ title: "OK", paper_bgcolor: "transparent" });
    // Light theme must not inject the dark gridcolor (xaxis untouched).
    expect((layoutArg as any).xaxis).toBeUndefined();
  });

  it("merges DARK_LAYOUT and deep-merges xaxis/yaxis when the theme is dark", async () => {
    const stub = (window as any).Plotly as PlotlyStub;
    const spec = {
      data: [{ x: [1], y: [2], type: "scatter" }],
      layout: {
        title: "T",
        xaxis: { title: "X" },
        yaxis: { title: "Y", range: [0, 10] },
      },
    };
    render(
      <PlotViewer
        plotType="oof-distribution"
        plots={{ "oof-distribution": spec }}
        loading={{}}
        onRequest={vi.fn()}
        theme="dark"
      />,
    );

    await waitFor(() => expect(stub.newPlot).toHaveBeenCalled());
    const [, , layoutArg] = stub.newPlot.mock.calls[0];
    // Dark base merged.
    expect((layoutArg as any).plot_bgcolor).toBe("#2d2d2d");
    expect((layoutArg as any).font?.color).toBe("#e0e0e0");
    // Per-axis values from the spec must be preserved alongside the dark
    // gridcolor so backend-set labels/ranges survive the merge.
    expect((layoutArg as any).xaxis).toMatchObject({
      title: "X",
      gridcolor: "#555",
    });
    expect((layoutArg as any).yaxis).toMatchObject({
      title: "Y",
      range: [0, 10],
      gridcolor: "#555",
    });
  });

  it("re-renders when the plot spec for the same plotType changes", async () => {
    const stub = installPlotly();
    const specA = { data: [{ x: [1] }], layout: {} };
    const specB = { data: [{ x: [2, 3] }], layout: {} };

    const { rerender } = render(
      <PlotViewer
        plotType="roc-curve"
        plots={{ "roc-curve": specA }}
        loading={{}}
        onRequest={vi.fn()}
      />,
    );
    await waitFor(() => expect(stub.newPlot).toHaveBeenCalledTimes(1));

    rerender(
      <PlotViewer
        plotType="roc-curve"
        plots={{ "roc-curve": specB }}
        loading={{}}
        onRequest={vi.fn()}
      />,
    );
    await waitFor(() => expect(stub.newPlot).toHaveBeenCalledTimes(2));
    const [, secondData] = stub.newPlot.mock.calls[1];
    expect(secondData).toEqual(specB.data);
  });

  it("skips Plotly.newPlot when the spec for plotType is missing", () => {
    const stub = installPlotly();
    render(
      <PlotViewer
        plotType="roc-curve"
        plots={{ "learning-curve": { data: [], layout: {} } }}
        loading={{}}
        onRequest={vi.fn()}
      />,
    );
    expect(stub.newPlot).not.toHaveBeenCalled();
  });
});
