/**
 * Tests for PredTable — paginated prediction results table with CSV download.
 *
 * #114 Phase A: was at 60% — pull pagination, download, formatCell paths.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { PredTable } from "../components/PredTable";

describe("PredTable — empty state", () => {
  it("renders the placeholder when data is empty", () => {
    render(<PredTable data={[]} />);
    expect(screen.getByText(/No predictions available/)).toBeDefined();
  });
});

describe("PredTable — table rendering", () => {
  it("renders one row per data entry and a column header per key", () => {
    const data = [
      { pred: 0.7, label: "A" },
      { pred: 0.4, label: "B" },
      { pred: 0.9, label: "C" },
    ];
    const { container } = render(<PredTable data={data} />);
    const rows = container.querySelectorAll("tbody tr");
    expect(rows.length).toBe(3);
    const headers = container.querySelectorAll("thead th");
    // # + pred + label
    expect(headers.length).toBe(3);
  });

  it("displays the row count in the toolbar", () => {
    render(<PredTable data={[{ pred: 0.5 }]} />);
    expect(screen.getByText(/1 rows/)).toBeDefined();
  });

  it("formats integer numbers without decimals", () => {
    render(<PredTable data={[{ pred: 5, label: "A" }]} />);
    expect(screen.getByText("5")).toBeDefined();
  });

  it("formats float numbers with 4 decimals", () => {
    render(<PredTable data={[{ pred: 0.123456789, label: "A" }]} />);
    expect(screen.getByText("0.1235")).toBeDefined();
  });

  it("renders dash for null/undefined cells", () => {
    const { container } = render(
      <PredTable data={[{ pred: null, label: undefined }]} />,
    );
    expect(container.textContent).toContain("-");
  });
});

describe("PredTable — pagination", () => {
  const makeData = (n: number) =>
    Array.from({ length: n }, (_, i) => ({ pred: i / 10, idx: i }));

  it("hides pagination when data fits in a single page", () => {
    const { container } = render(<PredTable data={makeData(20)} pageSize={50} />);
    expect(container.querySelector(".lzw-pred-table__pagination")).toBeNull();
  });

  it("shows Prev/Next when data spans multiple pages", () => {
    render(<PredTable data={makeData(120)} pageSize={50} />);
    expect(screen.getByText("Prev")).toBeDefined();
    expect(screen.getByText("Next")).toBeDefined();
    expect(screen.getByText("1 / 3")).toBeDefined();
  });

  it("disables Prev on the first page", () => {
    render(<PredTable data={makeData(120)} pageSize={50} />);
    expect((screen.getByText("Prev") as HTMLButtonElement).disabled).toBe(true);
  });

  it("advances the page on Next click and re-enables Prev", () => {
    render(<PredTable data={makeData(120)} pageSize={50} />);
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText("2 / 3")).toBeDefined();
    expect((screen.getByText("Prev") as HTMLButtonElement).disabled).toBe(false);
  });
});

describe("PredTable — Download CSV", () => {
  beforeEach(() => {
    if (!URL.createObjectURL) {
      Object.defineProperty(URL, "createObjectURL", {
        value: vi.fn().mockReturnValue("blob:test"),
        writable: true,
      });
    }
    if (!URL.revokeObjectURL) {
      Object.defineProperty(URL, "revokeObjectURL", {
        value: vi.fn(),
        writable: true,
      });
    }
  });

  it("creates a blob URL and triggers a download on click", () => {
    const create = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
    const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    try {
      render(<PredTable data={[{ pred: 0.5, label: "A" }]} />);
      fireEvent.click(screen.getByText("Download CSV"));
      expect(create).toHaveBeenCalledTimes(1);
      expect(click).toHaveBeenCalledTimes(1);
      expect(revoke).toHaveBeenCalledWith("blob:mock");
    } finally {
      click.mockRestore();
      create.mockRestore();
      revoke.mockRestore();
    }
  });
});
