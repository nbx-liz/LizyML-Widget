/** Accordion — collapsible section with optional header-right slot. */
import { useState } from "preact/hooks";
import type { ComponentChildren } from "preact";

interface AccordionProps {
  title: string;
  defaultOpen?: boolean;
  headerRight?: ComponentChildren;
  children: ComponentChildren;
}

export function Accordion({
  title,
  defaultOpen = true,
  headerRight,
  children,
}: AccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div class="lzw-accordion">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <button
          class="lzw-accordion__trigger"
          onClick={() => setOpen((o) => !o)}
          type="button"
          style="flex:1"
        >
          <span class="lzw-accordion__arrow">{open ? "\u25BE" : "\u25B8"}</span>
          <span class="lzw-accordion__title">{title}</span>
        </button>
        {headerRight && (
          <div style="flex-shrink:0" onClick={(e) => e.stopPropagation()}>
            {headerRight}
          </div>
        )}
      </div>
      {open && <div class="lzw-accordion__body">{children}</div>}
    </div>
  );
}
