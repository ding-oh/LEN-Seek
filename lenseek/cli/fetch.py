"""lenseek-fetch: download and verify release assets (lenseek/configs/assets.yaml) into $LENSEEK_HOME/assets.

    lenseek-fetch --list
    lenseek-fetch checkpoint_main_s0 latentdb_main_s0 latentdb_main_s0_standard
"""
from __future__ import annotations

import argparse
import os
import sys

from ..assets import AssetError, asset_dir, fetch, local_path, registry


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lenseek-fetch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true", help="list assets, sizes and availability")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    reg = registry()
    if args.list or not args.names:
        mirror = os.environ.get("LENSEEK_ASSET_MIRROR")
        print(f"asset directory: {asset_dir()}")
        if mirror:
            print(f"asset mirror: {mirror}")
        print(f"{'name':34s} {'size':>10s}  {'status':13s} description")
        for name, e in reg.items():
            size = e.get("size")
            size_s = f"{size / 1e6:,.0f} MB" if size else "-"
            if local_path(name):
                status = "present"
            elif not e.get("file"):
                status = "not released"
            elif mirror and os.path.isdir(mirror):     # fetch() takes the file from a directory mirror only
                status = "mirror" if os.path.exists(os.path.join(mirror, e["file"])) else "not in mirror"
            elif mirror:
                status = "mirror"
            else:
                status = "download" if e.get("url") else "unpublished"
            print(f"{name:34s} {size_s:>10s}  {status:13s} {e.get('description', '').strip()}")
        return
    rc = 0
    for name in args.names:
        try:
            print(fetch(name, quiet=args.quiet))
        except AssetError as e:
            print(f"error: {e}", file=sys.stderr)
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
