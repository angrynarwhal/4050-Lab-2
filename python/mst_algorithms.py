#!/usr/bin/env python3
"""
mst_algorithms.py — Kruskal's and Prim's MST algorithms with timing and instrumentation.

Usage:
    python3 mst_algorithms.py ../data/tiny_nodes.csv ../data/tiny_edges.csv --algorithm kruskal
    python3 mst_algorithms.py ../data/tiny_nodes.csv ../data/tiny_edges.csv --algorithm prim
    python3 mst_algorithms.py ../data/tiny_nodes.csv ../data/tiny_edges.csv --algorithm both
"""

import csv
import sys
import time
import heapq
import argparse
from collections import defaultdict


# ============================================================================
# UNION-FIND (Disjoint Sets)
# ============================================================================

class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.

    Operations:
        find(x)    — returns the root representative of x's set
        union(x,y) — merges the sets containing x and y
        connected(x,y) — checks if x and y are in the same set

    With both optimizations, amortized cost per operation is O(α(n)),
    where α is the inverse Ackermann function (effectively constant).
    """

    def __init__(self, n, path_compression=True, union_by_rank=True):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n          # Size of each component
        self.num_components = n
        self.path_compression = path_compression
        self.union_by_rank = union_by_rank

        # Instrumentation
        self.find_calls = 0
        self.total_path_length = 0
        self.union_calls = 0

    def find(self, x):
        """Find the root representative of x's set (iterative)."""
        self.find_calls += 1
        path_len = 0

        # Phase 1: Walk to root
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
            path_len += 1

        # Phase 2: Path compression (if enabled) — flatten all nodes to root
        if self.path_compression:
            curr = x
            while self.parent[curr] != root:
                next_node = self.parent[curr]
                self.parent[curr] = root
                curr = next_node

        self.total_path_length += path_len
        return root

    def union(self, x, y):
        """
        Merge the sets containing x and y.
        Returns True if they were in different sets (merge happened).
        Returns False if they were already in the same set (would create cycle).
        """
        self.union_calls += 1
        rx, ry = self.find(x), self.find(y)

        if rx == ry:
            return False  # Already in same set

        if self.union_by_rank:
            # Attach smaller-rank tree under larger-rank tree
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            self.size[rx] += self.size[ry]
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
        else:
            # Naive union — always attach ry under rx
            self.parent[ry] = rx
            self.size[rx] += self.size[ry]

        self.num_components -= 1
        return True

    def connected(self, x, y):
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def component_size(self, x):
        """Return the size of the component containing x."""
        return self.size[self.find(x)]

    def get_max_height(self):
        """Compute the maximum tree height in the Union-Find forest (iterative)."""
        max_h = 0
        for i in range(len(self.parent)):
            h = 0
            x = i
            while self.parent[x] != x:
                x = self.parent[x]
                h += 1
            if h > max_h:
                max_h = h
        return max_h

    def get_stats(self):
        """Return instrumentation statistics."""
        avg_path = (self.total_path_length / self.find_calls) if self.find_calls > 0 else 0
        return {
            "find_calls": self.find_calls,
            "union_calls": self.union_calls,
            "avg_path_length": round(avg_path, 4),
            "max_height": self.get_max_height(),
            "num_components": self.num_components,
        }


# ============================================================================
# GRAPH LOADING
# ============================================================================

def load_graph(nodes_file, edges_file):
    """
    Load graph from CSV files.
    Returns: (nodes_dict, edges_list, adj_list)
        nodes_dict: {node_id: {label, node_type, jurisdiction, country_codes}}
        edges_list: [(source, target, weight, rel_type), ...]
        adj_list: {node_id: [(neighbor, weight), ...]}
    """
    nodes = {}
    with open(nodes_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["node_id"])
            nodes[nid] = {
                "label": row.get("label", ""),
                "node_type": row.get("node_type", ""),
                "jurisdiction": row.get("jurisdiction", ""),
                "country_codes": row.get("country_codes", ""),
            }

    edges = []
    adj = defaultdict(list)
    with open(edges_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row["source"])
            t = int(row["target"])
            w = float(row["weight"])
            rt = row.get("rel_type", "")
            edges.append((s, t, w, rt))
            adj[s].append((t, w))
            adj[t].append((s, w))

    return nodes, edges, adj


# ============================================================================
# KRUSKAL'S ALGORITHM
# ============================================================================

def kruskal(nodes, edges, path_compression=True, union_by_rank=True, verbose=False):
    """
    Kruskal's MST algorithm.

    1. Sort all edges by weight
    2. For each edge (lightest first):
       - If it connects two different components, add it to the MST
       - Use Union-Find to track components

    Returns: (mst_edges, total_weight, uf_stats, elapsed_time)
    """
    n = len(nodes)
    if n == 0:
        return [], 0.0, {}, 0.0

    start_time = time.perf_counter()

    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: e[2])

    # Initialize Union-Find
    uf = UnionFind(n, path_compression=path_compression, union_by_rank=union_by_rank)

    mst_edges = []
    total_weight = 0.0

    for s, t, w, rt in sorted_edges:
        if uf.union(s, t):
            mst_edges.append((s, t, w, rt))
            total_weight += w
            if verbose and len(mst_edges) % 500 == 0:
                print(f"    Kruskal: {len(mst_edges)} edges added, "
                      f"{uf.num_components} components remaining")
            if len(mst_edges) == n - 1:
                break  # MST complete

    elapsed = time.perf_counter() - start_time

    if len(mst_edges) < n - 1:
        print(f"  Warning: MST has {len(mst_edges)} edges (expected {n-1}). "
              f"Graph may not be connected.")

    return mst_edges, total_weight, uf.get_stats(), elapsed


# ============================================================================
# PRIM'S ALGORITHM
# ============================================================================

def prim(nodes, adj, start=0, verbose=False):
    """
    Prim's MST algorithm using a binary heap (min-priority queue).

    1. Start from a single vertex
    2. Maintain a priority queue of edges from the growing tree
    3. Always add the lightest edge to an unvisited vertex

    Returns: (mst_edges, total_weight, stats, elapsed_time)
    """
    n = len(nodes)
    if n == 0:
        return [], 0.0, {}, 0.0

    start_time = time.perf_counter()

    visited = set()
    mst_edges = []
    total_weight = 0.0

    # Priority queue: (weight, from_node, to_node, rel_type)
    # We don't have rel_type easily in adj list, store "" placeholder
    pq = []
    visited.add(start)

    # Add all edges from start node
    for neighbor, weight in adj[start]:
        heapq.heappush(pq, (weight, start, neighbor))

    pq_ops = 0
    while pq and len(mst_edges) < n - 1:
        w, u, v = heapq.heappop(pq)
        pq_ops += 1

        if v in visited:
            continue

        visited.add(v)
        mst_edges.append((u, v, w, ""))
        total_weight += w

        if verbose and len(mst_edges) % 500 == 0:
            print(f"    Prim: {len(mst_edges)} edges added, "
                  f"{n - len(visited)} vertices remaining")

        for neighbor, weight in adj[v]:
            if neighbor not in visited:
                heapq.heappush(pq, (weight, v, neighbor))
                pq_ops += 1

    elapsed = time.perf_counter() - start_time

    if len(mst_edges) < n - 1:
        print(f"  Warning: Prim's MST has {len(mst_edges)} edges (expected {n-1}). "
              f"Graph may not be connected from start node {start}.")

    stats = {
        "pq_operations": pq_ops,
        "vertices_visited": len(visited),
    }

    return mst_edges, total_weight, stats, elapsed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MST Algorithms on Panama Papers Graph")
    parser.add_argument("nodes_file", help="Path to nodes CSV")
    parser.add_argument("edges_file", help="Path to edges CSV")
    parser.add_argument("--algorithm", choices=["kruskal", "prim", "both"], default="both",
                        help="Which algorithm to run")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--output", default=None, help="Output MST edges to CSV file")
    args = parser.parse_args()

    print("=" * 60)
    print("MST Algorithm Comparison")
    print("=" * 60)

    # Load graph
    print(f"\nLoading graph...")
    nodes, edges, adj = load_graph(args.nodes_file, args.edges_file)
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Edges: {len(edges):,}")

    results = {}

    if args.algorithm in ("kruskal", "both"):
        print(f"\n--- Kruskal's Algorithm ---")
        mst_edges, total_weight, uf_stats, elapsed = kruskal(
            nodes, edges, verbose=args.verbose
        )
        print(f"  MST edges: {len(mst_edges)}")
        print(f"  Total MST weight: {total_weight:.4f}")
        print(f"  Time: {elapsed:.4f} seconds")
        print(f"  Union-Find stats:")
        for k, v in uf_stats.items():
            print(f"    {k}: {v}")
        results["kruskal"] = (mst_edges, total_weight, elapsed)

    if args.algorithm in ("prim", "both"):
        print(f"\n--- Prim's Algorithm ---")
        mst_edges, total_weight, stats, elapsed = prim(
            nodes, adj, verbose=args.verbose
        )
        print(f"  MST edges: {len(mst_edges)}")
        print(f"  Total MST weight: {total_weight:.4f}")
        print(f"  Time: {elapsed:.4f} seconds")
        print(f"  Stats:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        results["prim"] = (mst_edges, total_weight, elapsed)

    if args.algorithm == "both" and "kruskal" in results and "prim" in results:
        kw = results["kruskal"][1]
        pw = results["prim"][1]
        print(f"\n--- Comparison ---")
        print(f"  Kruskal total weight: {kw:.4f}")
        print(f"  Prim total weight:    {pw:.4f}")
        print(f"  Difference:           {abs(kw - pw):.6f}")
        print(f"  Kruskal time: {results['kruskal'][2]:.4f}s")
        print(f"  Prim time:    {results['prim'][2]:.4f}s")

    # Output MST
    if args.output and results:
        algo = list(results.keys())[0]
        mst_edges = results[algo][0]
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "weight", "rel_type"])
            for s, t, w, rt in mst_edges:
                writer.writerow([s, t, w, rt])
        print(f"\n  MST edges written to {args.output}")


if __name__ == "__main__":
    main()
