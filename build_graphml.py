"""
build_graphml.py
================

One-time helper: download the San Francisco drivable road network from
OpenStreetMap and save it as `sf_road_network.graphml` for Stage 1+2+3.

Re-runs are no-ops if the file already exists; pass --force to rebuild.
"""

import argparse
import os
import sys

import osmnx as ox


OUT_PATH = "sf_road_network.graphml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the file already exists")
    parser.add_argument(
        "--place", default="San Francisco, California, USA",
        help="OSM place query (default: San Francisco, California, USA)",
    )
    args = parser.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        print(f"{OUT_PATH} already exists ({os.path.getsize(OUT_PATH)/1e6:.1f} MB). "
              "Use --force to rebuild.")
        return

    print(f"Downloading drivable road network for: {args.place}")
    G = ox.graph_from_place(args.place, network_type="drive")
    print(f"  nodes={G.number_of_nodes():,}  edges={G.number_of_edges():,}")

    print(f"Saving to {OUT_PATH}...")
    ox.save_graphml(G, OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Done. {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
