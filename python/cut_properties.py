#!/usr/bin/env python3
"""
cut_properties.py — Demonstrate the Blue Rule (cut property) and Red Rule (cycle property).

Usage:
    python3 cut_properties.py ../data/small_nodes.csv ../data/small_edges.csv --exercise blue
    python3 cut_properties.py ../data/small_nodes.csv ../data/small_edges.csv --exercise red
    python3 cut_properties.py ../data/small_nodes.csv ../data/small_edges.csv --exercise both
"""

import csv
import sys
import random
import argparse
from collections import defaultdict, deque

from mst_algorithms import load_graph, kruskal, UnionFind


# ============================================================================
# BLUE RULE (Cut Property) Verification
# ============================================================================

def verify_blue_rule(nodes, edges, adj, mst_edges, num_trials=10, seed=None):
    """
    Verify the Blue Rule: for a random cut (S, V\\S), the minimum-weight
    edge crossing the cut should be in the MST.

    Returns list of trial results.
    """
    if seed is not None:
        random.seed(seed)

    # Build MST edge set for fast lookup
    mst_set = set()
    for s, t, w, _ in mst_edges:
        mst_set.add((min(s, t), max(s, t)))

    node_ids = sorted(nodes.keys())
    n = len(node_ids)
    results = []

    for trial in range(num_trials):
        # Create a random cut: randomly assign each node to S or V\S
        # Ensure both sides are non-empty
        cut_size = random.randint(max(1, n // 10), min(n - 1, 9 * n // 10))
        S = set(random.sample(node_ids, cut_size))

        # Find all edges crossing the cut
        crossing_edges = []
        for s, t, w, rt in edges:
            if (s in S) != (t in S):  # One in S, one not
                crossing_edges.append((s, t, w, rt))

        if not crossing_edges:
            continue  # Skip if no crossing edges

        # Find minimum-weight crossing edge
        min_edge = min(crossing_edges, key=lambda e: e[2])
        min_key = (min(min_edge[0], min_edge[1]), max(min_edge[0], min_edge[1]))
        in_mst = min_key in mst_set

        # Check if there are ties
        min_weight = min_edge[2]
        ties = [e for e in crossing_edges if abs(e[2] - min_weight) < 1e-9]

        result = {
            "trial": trial + 1,
            "cut_size": len(S),
            "crossing_edges": len(crossing_edges),
            "min_weight": min_weight,
            "in_mst": in_mst,
            "num_ties": len(ties),
            "min_edge": (min_edge[0], min_edge[1]),
        }
        results.append(result)

        # Report
        status = "✓ IN MST" if in_mst else "✗ NOT IN MST"
        tie_note = f" ({len(ties)} ties)" if len(ties) > 1 else ""
        print(f"  Trial {trial+1:2d}: |S|={len(S):,}, "
              f"crossing={len(crossing_edges):,}, "
              f"min_weight={min_weight:.4f}{tie_note} → {status}")

    return results


def verify_blue_rule_jurisdictional(nodes, edges, mst_edges):
    """
    Special case: cut defined by jurisdiction (Panama vs others).
    Demonstrates the Blue Rule with a semantically meaningful cut.
    """
    mst_set = set()
    for s, t, w, _ in mst_edges:
        mst_set.add((min(s, t), max(s, t)))

    # Find jurisdictions
    jurisdictions = defaultdict(set)
    for nid, info in nodes.items():
        jur = info.get("jurisdiction", "Unknown") or "Unknown"
        jurisdictions[jur].add(nid)

    if len(jurisdictions) < 2:
        print("  Only one jurisdiction found. Skipping jurisdictional cut.")
        return None

    # Use the two largest jurisdictions
    sorted_jurs = sorted(jurisdictions.items(), key=lambda x: -len(x[1]))
    jur1_name, jur1_nodes = sorted_jurs[0]
    jur2_name, jur2_nodes = sorted_jurs[1]

    print(f"\n  Jurisdictional Cut: {jur1_name} ({len(jur1_nodes)} nodes) vs "
          f"{jur2_name} ({len(jur2_nodes)} nodes)")

    S = jur1_nodes
    crossing = []
    for s, t, w, rt in edges:
        if (s in S) != (t in S):
            crossing.append((s, t, w, rt))

    if not crossing:
        print("  No edges cross this jurisdictional boundary.")
        return None

    min_edge = min(crossing, key=lambda e: e[2])
    min_key = (min(min_edge[0], min_edge[1]), max(min_edge[0], min_edge[1]))
    in_mst = min_key in mst_set

    print(f"  Crossing edges: {len(crossing)}")
    print(f"  Min-weight crossing edge: ({min_edge[0]}, {min_edge[1]}) "
          f"weight={min_edge[2]:.4f} rel_type={min_edge[3]}")
    print(f"  Node {min_edge[0]}: {nodes.get(min_edge[0], {}).get('label', '?')}")
    print(f"  Node {min_edge[1]}: {nodes.get(min_edge[1], {}).get('label', '?')}")
    print(f"  In MST: {'✓ YES' if in_mst else '✗ NO'}")

    return {
        "jur1": jur1_name,
        "jur2": jur2_name,
        "crossing_edges": len(crossing),
        "min_weight": min_edge[2],
        "in_mst": in_mst,
    }


# ============================================================================
# RED RULE (Cycle Property) Verification
# ============================================================================

def find_cycles_via_non_tree_edges(nodes, edges, mst_edges, adj, num_cycles=10, seed=None):
    """
    Find cycles by adding non-tree edges to the MST.
    Each non-tree edge creates exactly one cycle with the MST.

    For each cycle, verify that the max-weight edge on the cycle
    is NOT in the MST (Red Rule).
    """
    if seed is not None:
        random.seed(seed)

    # Build MST adjacency list
    mst_adj = defaultdict(list)
    mst_set = set()
    for s, t, w, rt in mst_edges:
        mst_adj[s].append((t, w))
        mst_adj[t].append((s, w))
        mst_set.add((min(s, t), max(s, t)))

    # Find non-tree edges
    non_tree = []
    for s, t, w, rt in edges:
        key = (min(s, t), max(s, t))
        if key not in mst_set:
            non_tree.append((s, t, w, rt))

    if not non_tree:
        print("  No non-tree edges found (graph is already a tree).")
        return []

    random.shuffle(non_tree)
    results = []

    for i, (s, t, w_nt, rt_nt) in enumerate(non_tree[:num_cycles]):
        # Find path from s to t in MST using BFS
        path = bfs_path(mst_adj, s, t)
        if path is None:
            continue

        # Collect all edges on the cycle (MST path + non-tree edge)
        cycle_edges = []
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            # Find weight of this MST edge
            for neighbor, weight in mst_adj[u]:
                if neighbor == v:
                    key = (min(u, v), max(u, v))
                    cycle_edges.append((u, v, weight, key in mst_set))
                    break

        # Add the non-tree edge
        cycle_edges.append((s, t, w_nt, False))

        # Find max-weight edge on the cycle
        max_edge = max(cycle_edges, key=lambda e: e[2])
        max_in_mst = max_edge[3]

        result = {
            "cycle_num": i + 1,
            "cycle_length": len(cycle_edges),
            "non_tree_edge": (s, t, w_nt),
            "max_weight": max_edge[2],
            "max_edge": (max_edge[0], max_edge[1]),
            "max_in_mst": max_in_mst,
        }
        results.append(result)

        status = "✗ NOT in MST (Red Rule holds)" if not max_in_mst else "✓ IN MST (unexpected!)"
        print(f"  Cycle {i+1:2d}: length={len(cycle_edges)}, "
              f"max_weight={max_edge[2]:.4f} → {status}")

    return results


def bfs_path(adj, start, end):
    """Find shortest path from start to end using BFS."""
    if start == end:
        return [start]

    visited = {start}
    parent = {start: None}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor, _ in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                if neighbor == end:
                    # Reconstruct path
                    path = []
                    current = end
                    while current is not None:
                        path.append(current)
                        current = parent[current]
                    return list(reversed(path))
                queue.append(neighbor)

    return None  # No path found


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify Red Rule and Blue Rule on Panama Papers graph"
    )
    parser.add_argument("nodes_file", help="Path to nodes CSV")
    parser.add_argument("edges_file", help="Path to edges CSV")
    parser.add_argument("--exercise", choices=["blue", "red", "both"], default="both",
                        help="Which property to verify")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials for Blue Rule / cycles for Red Rule")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 60)
    print("Cut and Cycle Properties Verification")
    print("=" * 60)

    # Load graph
    print(f"\nLoading graph...")
    nodes, edges, adj = load_graph(args.nodes_file, args.edges_file)
    print(f"  Nodes: {len(nodes):,}, Edges: {len(edges):,}")

    # Compute MST first (needed for both exercises)
    print(f"\nComputing MST (Kruskal's)...")
    mst_edges, total_weight, _, elapsed = kruskal(nodes, edges)
    print(f"  MST: {len(mst_edges)} edges, weight={total_weight:.4f}, time={elapsed:.4f}s")

    if args.exercise in ("blue", "both"):
        print(f"\n{'=' * 40}")
        print(f"BLUE RULE (Cut Property) Verification")
        print(f"{'=' * 40}")
        print(f"\nThe Blue Rule states: for any cut (S, V\\S), the minimum-weight")
        print(f"edge crossing the cut must be in every MST.\n")

        results = verify_blue_rule(nodes, edges, adj, mst_edges,
                                    num_trials=args.trials, seed=args.seed)

        # Summary
        if results:
            all_in = sum(1 for r in results if r["in_mst"])
            print(f"\n  Summary: {all_in}/{len(results)} minimum crossing edges were in MST")
            if all_in < len(results):
                print(f"  (Exceptions likely due to duplicate edge weights)")

        # Jurisdictional cut
        print(f"\n--- Jurisdictional Cut (Semantic Blue Rule) ---")
        verify_blue_rule_jurisdictional(nodes, edges, mst_edges)

    if args.exercise in ("red", "both"):
        print(f"\n{'=' * 40}")
        print(f"RED RULE (Cycle Property) Verification")
        print(f"{'=' * 40}")
        print(f"\nThe Red Rule states: for any cycle, the maximum-weight edge")
        print(f"on the cycle is NOT in any MST.\n")

        results = find_cycles_via_non_tree_edges(
            nodes, edges, mst_edges, adj,
            num_cycles=args.trials, seed=args.seed
        )

        if results:
            none_in = sum(1 for r in results if not r["max_in_mst"])
            print(f"\n  Summary: {none_in}/{len(results)} max-weight cycle edges "
                  f"were correctly excluded from MST")


if __name__ == "__main__":
    main()
