#!/usr/bin/env python3
"""
clustering.py — MST-based k-clustering on the Panama Papers graph.

Algorithm:
    1. Compute the MST
    2. Remove the k-1 heaviest MST edges
    3. The resulting forest has k connected components = k clusters

Usage:
    python3 clustering.py ../data/small_nodes.csv ../data/small_edges.csv --clusters 5
    python3 clustering.py ../data/small_nodes.csv ../data/small_edges.csv --clusters 5 --analyze
"""

import csv
import sys
import argparse
from collections import defaultdict, Counter

from mst_algorithms import load_graph, kruskal, UnionFind


def mst_clustering(mst_edges, num_nodes, k):
    """
    Perform k-clustering by removing the k-1 heaviest MST edges.

    Returns:
        clusters: dict mapping cluster_id -> set of node_ids
        removed_edges: list of removed edges (heaviest)
        remaining_edges: list of edges still in the forest
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > num_nodes:
        raise ValueError(f"k={k} exceeds number of nodes ({num_nodes})")

    # Sort MST edges by weight (descending) to find heaviest
    sorted_mst = sorted(mst_edges, key=lambda e: -e[2])

    # Remove k-1 heaviest edges
    removed = sorted_mst[:k - 1]
    remaining = sorted_mst[k - 1:]

    # Build forest from remaining edges using Union-Find
    uf = UnionFind(num_nodes)
    for s, t, w, rt in remaining:
        uf.union(s, t)

    # Extract clusters
    cluster_map = defaultdict(set)
    for nid in range(num_nodes):
        root = uf.find(nid)
        cluster_map[root].add(nid)

    # Relabel clusters 0, 1, 2, ...
    clusters = {}
    for i, (root, members) in enumerate(
        sorted(cluster_map.items(), key=lambda x: -len(x[1]))
    ):
        clusters[i] = members

    return clusters, removed, remaining


def analyze_clusters(clusters, nodes, removed_edges):
    """Analyze cluster composition (jurisdictions, node types, intermediaries)."""
    print(f"\n  {'=' * 50}")
    print(f"  Cluster Analysis")
    print(f"  {'=' * 50}")

    for cid, members in sorted(clusters.items()):
        print(f"\n  --- Cluster {cid} ({len(members)} nodes) ---")

        # Node type distribution
        types = Counter()
        jurisdictions = Counter()
        countries = Counter()
        names = []

        for nid in members:
            info = nodes.get(nid, {})
            types[info.get("node_type", "unknown")] += 1
            jur = info.get("jurisdiction", "")
            if jur:
                jurisdictions[jur] += 1
            cc = info.get("country_codes", "")
            if cc:
                for c in cc.split(";"):
                    c = c.strip()
                    if c:
                        countries[c] += 1
            label = info.get("label", "")
            if label and info.get("node_type") == "intermediary":
                names.append(label)

        print(f"    Node types: ", end="")
        print(", ".join(f"{t}={c}" for t, c in types.most_common()))

        if jurisdictions:
            print(f"    Top jurisdictions: ", end="")
            print(", ".join(f"{j}={c}" for j, c in jurisdictions.most_common(5)))

        if countries:
            print(f"    Top countries: ", end="")
            print(", ".join(f"{cc}={c}" for cc, c in countries.most_common(5)))

        if names:
            name_counts = Counter(names)
            print(f"    Key intermediaries: ", end="")
            print(", ".join(f"{n} ({c})" for n, c in name_counts.most_common(3)))

    # Removed edges analysis
    if removed_edges:
        print(f"\n  --- Removed Edges (Cluster Boundaries) ---")
        for i, (s, t, w, rt) in enumerate(removed_edges):
            s_info = nodes.get(s, {})
            t_info = nodes.get(t, {})
            print(f"    Edge {i+1}: weight={w:.4f}, rel_type={rt}")
            print(f"      {s_info.get('label', '?')[:50]} ({s_info.get('node_type', '?')})")
            print(f"      ↔ {t_info.get('label', '?')[:50]} ({t_info.get('node_type', '?')})")


def export_clusters(clusters, nodes, output_path):
    """Export cluster assignments to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "cluster_id", "label", "node_type", "jurisdiction"])
        for cid, members in sorted(clusters.items()):
            for nid in sorted(members):
                info = nodes.get(nid, {})
                writer.writerow([
                    nid, cid,
                    info.get("label", ""),
                    info.get("node_type", ""),
                    info.get("jurisdiction", ""),
                ])
    print(f"\n  Cluster assignments exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MST-based k-clustering on Panama Papers graph"
    )
    parser.add_argument("nodes_file", help="Path to nodes CSV")
    parser.add_argument("edges_file", help="Path to edges CSV")
    parser.add_argument("--clusters", "-k", type=int, required=True,
                        help="Number of clusters")
    parser.add_argument("--analyze", action="store_true",
                        help="Print detailed cluster analysis")
    parser.add_argument("--export", default=None,
                        help="Export cluster assignments to CSV")
    args = parser.parse_args()

    print("=" * 60)
    print(f"MST-Based Clustering (k={args.clusters})")
    print("=" * 60)

    # Load graph
    print(f"\nLoading graph...")
    nodes, edges, adj = load_graph(args.nodes_file, args.edges_file)
    print(f"  Nodes: {len(nodes):,}, Edges: {len(edges):,}")

    # Compute MST
    print(f"\nComputing MST...")
    mst_edges, total_weight, _, elapsed = kruskal(nodes, edges)
    print(f"  MST: {len(mst_edges)} edges, weight={total_weight:.4f}")

    # Perform clustering
    print(f"\nClustering into {args.clusters} groups...")
    clusters, removed, remaining = mst_clustering(
        mst_edges, len(nodes), args.clusters
    )

    # Report results
    sizes = sorted([len(m) for m in clusters.values()], reverse=True)
    print(f"\n  Results:")
    print(f"    Number of clusters: {len(clusters)}")
    print(f"    Cluster sizes: {sizes}")
    print(f"    Largest cluster: {sizes[0]} nodes")
    print(f"    Smallest cluster: {sizes[-1]} nodes")

    if removed:
        removed_weights = [w for _, _, w, _ in removed]
        print(f"\n  Removed edges (heaviest {len(removed)} MST edges):")
        for i, (s, t, w, rt) in enumerate(removed):
            print(f"    {i+1}. weight={w:.4f} ({s} ↔ {t})")
        print(f"    Weight range: {min(removed_weights):.4f} - {max(removed_weights):.4f}")

    # Inter-cluster spacing (min weight of removed edges = cluster spacing)
    if removed:
        spacing = min(w for _, _, w, _ in removed)
        print(f"\n  Cluster spacing (min removed weight): {spacing:.4f}")
        print(f"    This is the minimum distance between any two clusters.")
        print(f"    It represents the 'weakest link' that was severed.")

    # Detailed analysis
    if args.analyze:
        analyze_clusters(clusters, nodes, removed)

    # Export
    if args.export:
        export_clusters(clusters, nodes, args.export)


if __name__ == "__main__":
    main()
