#!/usr/bin/env python
"""Inline data.json into the viz template to produce a self-contained page.

    python export_viz.py --run runs/<run> --out viz/data.json
    python build_viz.py                       # -> viz/index.html

The page must be one file with no fetches: an Artifact's CSP blocks requests to
any external host, and opening index.html straight off disk (file://) blocks
fetch() too. So the payload is embedded in a <script type="application/json">
rather than loaded at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=Path("viz/template.html"))
    ap.add_argument("--data", type=Path, default=Path("viz/data.json"))
    ap.add_argument("--out", type=Path, default=Path("viz/index.html"))
    args = ap.parse_args()

    tpl = args.template.read_text()
    raw = args.data.read_text()
    json.loads(raw)                      # fail loudly on malformed data

    if "__DATA__" not in tpl:
        raise SystemExit("template has no __DATA__ placeholder")
    # </script> inside the payload would close the host tag early.
    payload = raw.replace("</", "<\\/")
    args.out.write_text(tpl.replace("__DATA__", payload))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
