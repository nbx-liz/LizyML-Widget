/** anywidget ESM entry — receives model and mounts Preact App. */
import { render } from "preact";
import { App } from "./App";

function render_widget({ model, el }: { model: any; el: HTMLElement }) {
  el.classList.add("lzw-root");
  render(<App model={model} rootEl={el} />, el);
  return () => render(null, el);
}

export default { render: render_widget };
