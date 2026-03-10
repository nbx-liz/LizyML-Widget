/** Accordion — collapsible section. */
import { useState } from "preact/hooks";
import type { ComponentChildren } from "preact";

interface AccordionProps {
  title: string;
  defaultOpen?: boolean;
  children: ComponentChildren;
}

export function Accordion({
  title,
  defaultOpen = true,
  children,
}: AccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div class="lzw-accordion">
      <button
        class="lzw-accordion__trigger"
        onClick={() => setOpen((o) => !o)}
        type="button"
      >
        <span class="lzw-accordion__arrow">{open ? "\u25BE" : "\u25B8"}</span>
        <span class="lzw-accordion__title">{title}</span>
      </button>
      {open && <div class="lzw-accordion__body">{children}</div>}
    </div>
  );
}
