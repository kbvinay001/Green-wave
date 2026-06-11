#!/usr/bin/env python3
"""
Pick the green-wave corridor out of an OSM-converted SUMO net.

Give it two lat/lon points (where the ambulance enters and leaves the map)
and it walks the road between them, finds the junctions big enough to
deserve a signal, and prints everything the rest of the project needs:

  - the node ids for netconvert's --tls.set (to force signals there)
  - the corridor block to paste into common/config.yaml
  - the edge list for the ambulance route

Usage:
  python sim/build_corridor.py sim/nets/benz_circle/benz.net.xml \
      --from-latlon 16.5002,80.6474 --to-latlon 16.4965,80.6587 \
      --lane-id approach_west
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("SUMO_HOME", ""), "tools"))
import sumolib  # noqa: E402


def nearest_edge(net, lat, lon, radius=120):
    # convertLonLat2XY wants lon first -- easy to trip on
    x, y = net.convertLonLat2XY(lon, lat)
    hits = net.getNeighboringEdges(x, y, radius)
    # closest drivable edge wins
    hits = [(d, e) for e, d in hits if e.allows("passenger")]
    if not hits:
        raise SystemExit(f"no drivable edge within {radius}m of {lat},{lon}")
    return min(hits)[1]


def fastest_path(net, e_from, e_to):
    """Plain Dijkstra, cost = edge travel time at the speed limit."""
    import heapq
    dist = {e_from.getID(): e_from.getLength() / e_from.getSpeed()}
    prev = {}
    pq = [(dist[e_from.getID()], e_from.getID())]
    while pq:
        d, eid = heapq.heappop(pq)
        if eid == e_to.getID():
            break
        if d > dist.get(eid, float("inf")):
            continue
        edge = net.getEdge(eid)
        for nxt in edge.getOutgoing():
            if not nxt.allows("passenger"):
                continue
            nd = d + nxt.getLength() / nxt.getSpeed()
            if nd < dist.get(nxt.getID(), float("inf")):
                dist[nxt.getID()] = nd
                prev[nxt.getID()] = eid
                heapq.heappush(pq, (nd, nxt.getID()))
    if e_to.getID() not in dist:
        return None
    # walk the chain backwards
    out, cur = [], e_to.getID()
    while cur is not None:
        out.append(net.getEdge(cur))
        cur = prev.get(cur)
    return list(reversed(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--from-latlon", required=True)
    ap.add_argument("--to-latlon", required=True)
    ap.add_argument("--lane-id", default="approach_west")
    ap.add_argument("--min-incoming", type=int, default=3,
                    help="how many incoming edges make a junction 'major'")
    ap.add_argument("--min-gap-m", type=float, default=150,
                    help="skip junctions closer than this to the previous pick")
    ap.add_argument("--max-tls", type=int, default=6,
                    help="cap on corridor signals (keeps the green wave sane)")
    args = ap.parse_args()

    net = sumolib.net.readNet(args.net)
    lat1, lon1 = map(float, args.from_latlon.split(","))
    lat2, lon2 = map(float, args.to_latlon.split(","))

    e_from = nearest_edge(net, lat1, lon1)
    e_to = nearest_edge(net, lat2, lon2)
    print(f"route: {e_from.getID()} ({e_from.getName()})  ->  "
          f"{e_to.getID()} ({e_to.getName()})\n")

    # fastest, not shortest: keeps the route on the trunk road instead of
    # letting it shave metres through residential lanes. Our sumolib is too
    # old for fastest=True, so this is a tiny Dijkstra over travel time.
    path = fastest_path(net, e_from, e_to)
    if not path:
        raise SystemExit("no route between those points -- widen the bbox?")

    # A junction is signal-worthy when several streets actually meet there --
    # not just where the same road changes edge id. Count distinct incoming
    # street names (unnamed side lanes count as one bucket).
    def street_count(node):
        names = set()
        for e in node.getIncoming():
            names.add(e.getName() or "unnamed")
        return len(names)

    corridor = []          # (node, approach_edge, dist_from_entry)
    dist = 0.0
    last_pick = -1e9
    for edge in path:
        dist += edge.getLength()
        node = edge.getToNode()
        if len(node.getIncoming()) < args.min_incoming:
            continue
        if street_count(node) < 2:
            continue
        if dist - last_pick < args.min_gap_m:
            continue
        corridor.append((node, edge, dist))
        last_pick = dist

    corridor = corridor[:args.max_tls]

    print(f"path: {len(path)} edges, {dist:.0f} m total, "
          f"{len(corridor)} major junctions on the way:\n")

    for node, edge, d in corridor:
        x, y = node.getCoord()
        lon, lat = net.convertXY2LonLat(x, y)
        names = {e.getName() for e in node.getIncoming() if e.getName()}
        print(f"  {node.getID():30s} at {d:5.0f}m  ({lat:.5f},{lon:.5f})")
        for n in sorted(names)[:3]:
            print(f"      - {n}")

    print("\n--tls.set for netconvert:")
    print("  " + ",".join(n.getID() for n, _, _ in corridor))

    print("\ncorridor block for common/config.yaml:")
    print("intersection:")
    print("  corridors:")
    print(f"    - lane_id: {args.lane_id}")
    print("      intersections:")
    for node, edge, d in corridor:
        print(f"        - {{id: \"{node.getID()}\", distance_m: {d:.0f}, "
              f"approach_edge: \"{edge.getID()}\"}}")

    print("\nambulance route edges:")
    print("  " + " ".join(e.getID() for e in path))


if __name__ == "__main__":
    main()
