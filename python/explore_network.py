#!/usr/bin/env python3
"""
explore_network.py — Exercise 6: Choose Your Own Network

This script is your template for exploring a custom subgraph from the
ICIJ Offshore Leaks Database. You will:

  1. Define your own network by choosing filter parameters
  2. Build the graph using build_graph.py
  3. Compute MST statistics, cut/cycle properties, and clustering
  4. Compare your network to the default Panama dataset
  5. Write up your findings

=== INSTRUCTIONS ===

Step 1: Choose a network to explore. Pick ONE of these approaches:

  A) A different COUNTRY (who uses offshore networks in your country?):
       python3 ../scripts/build_graph.py --country "China" --max-nodes 5000 --output my_network

  B) A different JURISDICTION (where are shells incorporated?):
       python3 ../scripts/build_graph.py --jurisdiction "British Virgin Islands" --max-nodes 5000 --output my_network

  C) A specific INTERMEDIARY (one law firm's client network):
       python3 ../scripts/build_graph.py --intermediary "Appleby" --max-nodes 5000 --output my_network

  D) A specific DATA SOURCE (Panama Papers vs Paradise Papers vs Pandora Papers):
       python3 ../scripts/build_graph.py --source "Paradise Papers" --max-nodes 5000 --output my_network

  E) COMBINE filters for a focused slice:
       python3 ../scripts/build_graph.py --country "Russia" --jurisdiction "Panama" --max-nodes 5000 --output my_network

Step 2: Edit the CONFIGURATION section below to point to your data files.

Step 3: Run this script:
       python3 explore_network.py

Step 4: Fill in the ANALYSIS section at the bottom with your observations.

Usage:
    python3 explore_network.py [--data-dir ../data]
"""

import os
import sys
import time
import argparse
from collections import Counter, defaultdict

from mst_algorithms import load_graph, kruskal, prim, UnionFind
from cut_properties import (verify_blue_rule, verify_blue_rule_jurisdictional,
                             find_cycles_via_non_tree_edges)
from clustering import mst_clustering, analyze_clusters


# ========================================================================
# CONFIGURATION — EDIT THIS SECTION
# ========================================================================

# Your custom network name (must match the --output you used in build_graph.py)
MY_NETWORK = "my_network"

# Brief description of what you chose and why
MY_DESCRIPTION = """
REPLACE THIS with a 2-3 sentence description of the network you chose.
Example: "I chose to explore entities connected to China because China is
one of the largest sources of offshore capital flows. I used --country China
with --max-nodes 5000 to get a manageable graph size."
"""

# The Panama 'small' dataset for comparison (don't change this)
COMPARISON_NETWORK = "small"

# Number of clusters to analyze
K_VALUES = [3, 5, 10, 20]

# ========================================================================
# END CONFIGURATION
# ========================================================================


def load_network(data_dir, name):
    """Load a network and return nodes, edges, adjacency list."""
    nodes_f = os.path.join(data_dir, f"{name}_nodes.csv")
    edges_f = os.path.join(data_dir, f"{name}_edges.csv")

    if not os.path.exists(nodes_f) or not os.path.exists(edges_f):
        print(f"Error: Data files not found for '{name}'.")
        print(f"  Expected: {nodes_f}")
        print(f"           {edges_f}")
        print(f"\nDid you run build_graph.py with --output {name}?")
        sys.exit(1)

    return load_graph(nodes_f, edges_f)


def network_profile(nodes, edges, adj, label="Network"):
    """Compute and print a comprehensive profile of the network."""
    n = len(nodes)
    m = len(edges)

    print(f"\n{'=' * 66}")
    print(f"  NETWORK PROFILE: {label}")
    print(f"{'=' * 66}")

    # --- Basic stats ---
    print(f"\n  Basic Statistics:")
    print(f"    Nodes:              {n:,}")
    print(f"    Edges:              {m:,}")
    print(f"    Edge/node ratio:    {m/n:.2f}")
    print(f"    Density:            {2*m / (n*(n-1)):.6f}" if n > 1 else "")

    # --- Degree distribution ---
    degrees = [len(adj.get(i, [])) for i in range(n)]
    avg_deg = sum(degrees) / n if n > 0 else 0
    max_deg = max(degrees) if degrees else 0
    max_deg_node = degrees.index(max_deg) if degrees else -1
    min_deg = min(degrees) if degrees else 0
    isolated = sum(1 for d in degrees if d == 0)

    print(f"\n  Degree Distribution:")
    print(f"    Average degree:     {avg_deg:.2f}")
    print(f"    Max degree:         {max_deg} (node {max_deg_node}: "
          f"{nodes.get(max_deg_node, {}).get('label', '?')[:40]})")
    print(f"    Min degree:         {min_deg}")
    print(f"    Isolated nodes:     {isolated}")

    # Degree histogram (buckets)
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-50": 0, "51-100": 0, "100+": 0}
    for d in degrees:
        if d <= 1:
            buckets["1"] += 1
        elif d <= 5:
            buckets["2-5"] += 1
        elif d <= 10:
            buckets["6-10"] += 1
        elif d <= 50:
            buckets["11-50"] += 1
        elif d <= 100:
            buckets["51-100"] += 1
        else:
            buckets["100+"] += 1
    print(f"    Degree histogram:   ", end="")
    print(", ".join(f"{k}:{v}" for k, v in buckets.items() if v > 0))

    # --- Node type distribution ---
    type_counts = Counter()
    jur_counts = Counter()
    country_counts = Counter()
    for nid, info in nodes.items():
        type_counts[info.get("node_type", "unknown")] += 1
        jur = info.get("jurisdiction", "")
        if jur:
            jur_counts[jur] += 1
        cc = info.get("country_codes", "")
        if cc:
            for c in cc.split(";"):
                c = c.strip()
                if c:
                    country_counts[c] += 1

    print(f"\n  Node Types:")
    for t, c in type_counts.most_common():
        print(f"    {t:<20s} {c:>6,}  ({100*c/n:.1f}%)")

    if jur_counts:
        print(f"\n  Top 10 Jurisdictions:")
        for j, c in jur_counts.most_common(10):
            print(f"    {j:<25s} {c:>6,}  ({100*c/n:.1f}%)")

    if country_counts:
        print(f"\n  Top 10 Countries:")
        for cc, c in country_counts.most_common(10):
            print(f"    {cc:<10s} {c:>6,}")

    # --- Edge type distribution ---
    rel_counts = Counter()
    weight_sum = 0
    weight_min = float('inf')
    weight_max = float('-inf')
    for s, t, w, rt in edges:
        rel_counts[rt] += 1
        weight_sum += w
        weight_min = min(weight_min, w)
        weight_max = max(weight_max, w)

    print(f"\n  Edge Types:")
    for rt, c in rel_counts.most_common():
        print(f"    {rt:<25s} {c:>6,}  ({100*c/m:.1f}%)")

    print(f"\n  Edge Weights:")
    print(f"    Min:                {weight_min:.4f}")
    print(f"    Max:                {weight_max:.4f}")
    print(f"    Mean:               {weight_sum/m:.4f}")

    return {
        "nodes": n, "edges": m, "avg_degree": avg_deg,
        "max_degree": max_deg, "density": 2*m/(n*(n-1)) if n > 1 else 0,
        "type_counts": type_counts, "jur_counts": jur_counts,
        "country_counts": country_counts, "rel_counts": rel_counts,
        "weight_min": weight_min, "weight_max": weight_max,
        "weight_mean": weight_sum / m,
    }


def mst_analysis(nodes, edges, adj, label="Network"):
    """Run MST construction and report statistics."""
    n = len(nodes)
    m = len(edges)

    print(f"\n{'=' * 66}")
    print(f"  MST ANALYSIS: {label}")
    print(f"{'=' * 66}")

    # Kruskal's
    k_edges, k_weight, k_stats, k_time = kruskal(nodes, edges)
    # Prim's
    p_edges, p_weight, p_stats, p_time = prim(nodes, adj)

    print(f"\n  MST Construction:")
    print(f"    Kruskal: weight={k_weight:.4f}, time={k_time:.4f}s, "
          f"MST edges={len(k_edges)}")
    print(f"    Prim:    weight={p_weight:.4f}, time={p_time:.4f}s, "
          f"MST edges={len(p_edges)}")
    print(f"    Weight match: {'✓ YES' if abs(k_weight - p_weight) < 0.001 else '✗ NO'}")

    # MST edge weight distribution
    mst_weights = sorted([w for _, _, w, _ in k_edges])
    if mst_weights:
        print(f"\n  MST Edge Weights:")
        print(f"    Min:    {mst_weights[0]:.4f}")
        print(f"    Max:    {mst_weights[-1]:.4f}")
        print(f"    Mean:   {sum(mst_weights)/len(mst_weights):.4f}")
        print(f"    Median: {mst_weights[len(mst_weights)//2]:.4f}")

    # MST relationship types
    mst_rel_counts = Counter(rt for _, _, _, rt in k_edges)
    print(f"\n  MST Edge Types (which relationships are critical?):")
    for rt, c in mst_rel_counts.most_common():
        pct = 100 * c / len(k_edges)
        print(f"    {rt:<25s} {c:>6,}  ({pct:.1f}%)")

    # Non-tree edges
    mst_set = set((min(s, t), max(s, t)) for s, t, _, _ in k_edges)
    non_tree = [(s, t, w, rt) for s, t, w, rt in edges
                if (min(s, t), max(s, t)) not in mst_set]
    print(f"\n  Non-tree edges:       {len(non_tree):,} "
          f"(these create cycles if added)")

    return k_edges, k_weight, k_time, p_time, k_stats


def cut_cycle_analysis(nodes, edges, adj, mst_edges, label="Network"):
    """Run Blue Rule and Red Rule verification."""
    print(f"\n{'=' * 66}")
    print(f"  CUT & CYCLE PROPERTIES: {label}")
    print(f"{'=' * 66}")

    # Blue Rule
    print(f"\n  Blue Rule (Cut Property) — 10 random cuts:")
    blue_results = verify_blue_rule(
        nodes, edges, adj, mst_edges, num_trials=10, seed=42)

    all_passed = all(r.get("in_mst", False) for r in blue_results)
    print(f"\n  Blue Rule holds for all trials: "
          f"{'✓ YES' if all_passed else '✗ NO'}")

    # Jurisdictional cut
    print(f"\n  Jurisdictional cut:")
    jur_result = verify_blue_rule_jurisdictional(nodes, edges, mst_edges)

    # Red Rule
    print(f"\n  Red Rule (Cycle Property) — 10 cycles:")
    red_results = find_cycles_via_non_tree_edges(
        nodes, edges, mst_edges, adj, num_cycles=10, seed=42)

    if red_results:
        all_hold = all(not r.get("max_in_mst", True) for r in red_results)
        print(f"\n  Red Rule holds for all cycles: "
              f"{'✓ YES' if all_hold else '✗ NO'}")
    else:
        print(f"\n  No cycles found (graph may be too sparse)")

    return blue_results, red_results


def clustering_analysis(nodes, mst_edges, k_values, label="Network"):
    """Run MST-based clustering for multiple k values."""
    n = len(nodes)

    print(f"\n{'=' * 66}")
    print(f"  CLUSTERING ANALYSIS: {label}")
    print(f"{'=' * 66}")

    results = []
    for k in k_values:
        if k >= n:
            continue

        clusters, removed, remaining = mst_clustering(mst_edges, n, k)
        sizes = sorted([len(m) for m in clusters.values()], reverse=True)
        removed_weights = sorted([w for _, _, w, _ in removed], reverse=True)

        result = {
            "k": k, "sizes": sizes,
            "largest": sizes[0], "smallest": sizes[-1],
            "max_removed": max(removed_weights) if removed_weights else 0,
            "min_removed": min(removed_weights) if removed_weights else 0,
        }
        results.append(result)

        print(f"\n  k={k}:")
        print(f"    Sizes:        {sizes[:8]}{'...' if len(sizes) > 8 else ''}")
        print(f"    Largest:      {sizes[0]:,}")
        print(f"    Smallest:     {sizes[-1]:,}")
        print(f"    Size ratio:   {sizes[0]/sizes[-1]:.1f}x")
        if removed_weights:
            print(f"    Max removed:  {removed_weights[0]:.4f}")
            print(f"    Min removed:  {removed_weights[-1]:.4f}")

    # Detailed analysis for k=5
    best_k = 5 if 5 in k_values else k_values[0]
    clusters, removed, remaining = mst_clustering(mst_edges, n, best_k)
    print(f"\n  Detailed cluster composition (k={best_k}):")
    analyze_clusters(clusters, nodes, removed)

    # Summary table
    print(f"\n  {'k':>3} {'Largest':>8} {'Smallest':>9} {'Ratio':>7} "
          f"{'Max Cut':>9} {'Min Cut':>9}")
    print(f"  {'-'*52}")
    for r in results:
        ratio = r['largest'] / r['smallest'] if r['smallest'] > 0 else float('inf')
        print(f"  {r['k']:>3} {r['largest']:>8,} {r['smallest']:>9,} "
              f"{ratio:>6.1f}x {r['max_removed']:>9.4f} {r['min_removed']:>9.4f}")

    return results


def union_find_analysis(nodes, edges, label="Network"):
    """Run Union-Find performance comparison."""
    print(f"\n{'=' * 66}")
    print(f"  UNION-FIND PERFORMANCE: {label}")
    print(f"{'=' * 66}")

    configs = [
        ("No optimizations", False, False),
        ("Path compression only", True, False),
        ("Union by rank only", False, True),
        ("Both optimizations", True, True),
    ]

    results = []
    for name, pc, ubr in configs:
        _, _, stats, elapsed = kruskal(
            nodes, edges, path_compression=pc, union_by_rank=ubr)
        results.append({"config": name, "time": elapsed, **stats})

    print(f"\n  {'Config':<25s} {'Time(s)':>8} {'Finds':>8} "
          f"{'Avg Path':>9} {'Max Ht':>7}")
    print(f"  {'-'*62}")
    for r in results:
        print(f"  {r['config']:<25s} {r['time']:>8.4f} {r['find_calls']:>8,} "
              f"{r['avg_path_length']:>9.4f} {r['max_height']:>7}")

    # Compute speedups
    base = results[0]['time']
    if base > 0:
        print(f"\n  Speedups vs naive:")
        for r in results[1:]:
            speedup = base / r['time'] if r['time'] > 0 else float('inf')
            print(f"    {r['config']:<25s} {speedup:.1f}x")

    return results


def comparison_summary(my_profile, cmp_profile, my_label, cmp_label):
    """Print side-by-side comparison of two networks."""
    print(f"\n{'=' * 66}")
    print(f"  COMPARISON: {my_label} vs {cmp_label}")
    print(f"{'=' * 66}")

    rows = [
        ("Nodes", my_profile['nodes'], cmp_profile['nodes']),
        ("Edges", my_profile['edges'], cmp_profile['edges']),
        ("Edge/node ratio",
         f"{my_profile['edges']/my_profile['nodes']:.2f}",
         f"{cmp_profile['edges']/cmp_profile['nodes']:.2f}"),
        ("Avg degree",
         f"{my_profile['avg_degree']:.2f}",
         f"{cmp_profile['avg_degree']:.2f}"),
        ("Max degree", my_profile['max_degree'], cmp_profile['max_degree']),
        ("Weight min",
         f"{my_profile['weight_min']:.4f}",
         f"{cmp_profile['weight_min']:.4f}"),
        ("Weight max",
         f"{my_profile['weight_max']:.4f}",
         f"{cmp_profile['weight_max']:.4f}"),
        ("Weight mean",
         f"{my_profile['weight_mean']:.4f}",
         f"{cmp_profile['weight_mean']:.4f}"),
    ]

    print(f"\n  {'Metric':<22s} {my_label:>18s} {cmp_label:>18s}")
    print(f"  {'-'*60}")
    for label, mine, theirs in rows:
        print(f"  {label:<22s} {str(mine):>18s} {str(theirs):>18s}")

    # Type breakdown comparison
    print(f"\n  Node type comparison:")
    all_types = set(my_profile['type_counts'].keys()) | set(cmp_profile['type_counts'].keys())
    for t in sorted(all_types):
        mc = my_profile['type_counts'].get(t, 0)
        cc = cmp_profile['type_counts'].get(t, 0)
        print(f"    {t:<20s} {mc:>8,} vs {cc:>8,}")


def main():
    parser = argparse.ArgumentParser(
        description="Exercise 6: Explore your own ICIJ network")
    parser.add_argument("--data-dir", default=os.path.join("..", "data"),
                        help="Directory containing graph data files")
    args = parser.parse_args()

    # ================================================================
    # HEADER
    # ================================================================
    print("=" * 66)
    print("  CS4050 Lab 2 — Exercise 6: Choose Your Own Network")
    print("=" * 66)
    print(f"\n  Your network:      {MY_NETWORK}")
    print(f"  Comparison:        {COMPARISON_NETWORK} (Panama)")
    print(f"  Description:")
    for line in MY_DESCRIPTION.strip().split("\n"):
        print(f"    {line.strip()}")

    # ================================================================
    # LOAD BOTH NETWORKS
    # ================================================================
    print(f"\n  Loading networks...")
    my_nodes, my_edges, my_adj = load_network(args.data_dir, MY_NETWORK)
    cmp_nodes, cmp_edges, cmp_adj = load_network(args.data_dir, COMPARISON_NETWORK)

    my_label = MY_NETWORK.replace("_", " ").title()
    cmp_label = f"Panama ({COMPARISON_NETWORK})"

    # ================================================================
    # SECTION 1: Network Profiles
    # ================================================================
    my_profile = network_profile(my_nodes, my_edges, my_adj, my_label)
    cmp_profile = network_profile(cmp_nodes, cmp_edges, cmp_adj, cmp_label)

    comparison_summary(my_profile, cmp_profile, my_label, cmp_label)

    # ================================================================
    # SECTION 2: MST Construction
    # ================================================================
    my_mst, my_mst_w, my_k_t, my_p_t, _ = mst_analysis(
        my_nodes, my_edges, my_adj, my_label)
    cmp_mst, cmp_mst_w, cmp_k_t, cmp_p_t, _ = mst_analysis(
        cmp_nodes, cmp_edges, cmp_adj, cmp_label)

    print(f"\n  MST Comparison:")
    print(f"    {'Metric':<25s} {my_label:>18s} {cmp_label:>18s}")
    print(f"    {'-'*63}")
    print(f"    {'Total MST weight':<25s} {my_mst_w:>18.4f} {cmp_mst_w:>18.4f}")
    print(f"    {'Avg MST edge weight':<25s} "
          f"{my_mst_w/len(my_mst):>18.4f} {cmp_mst_w/len(cmp_mst):>18.4f}")
    print(f"    {'Kruskal time (s)':<25s} {my_k_t:>18.4f} {cmp_k_t:>18.4f}")
    print(f"    {'Prim time (s)':<25s} {my_p_t:>18.4f} {cmp_p_t:>18.4f}")

    # ================================================================
    # SECTION 3: Cut & Cycle Properties
    # ================================================================
    cut_cycle_analysis(my_nodes, my_edges, my_adj, my_mst, my_label)

    # ================================================================
    # SECTION 4: Clustering
    # ================================================================
    my_clust = clustering_analysis(my_nodes, my_mst, K_VALUES, my_label)
    cmp_clust = clustering_analysis(cmp_nodes, cmp_mst, K_VALUES, cmp_label)

    # ================================================================
    # SECTION 5: Union-Find Performance
    # ================================================================
    my_uf = union_find_analysis(my_nodes, my_edges, my_label)
    cmp_uf = union_find_analysis(cmp_nodes, cmp_edges, cmp_label)

    # ================================================================
    # REPORT TEMPLATE
    # ================================================================
    print(f"\n\n{'#' * 66}")
    print(f"#  EXERCISE 6 — FILL IN YOUR ANALYSIS BELOW")
    print(f"#  Copy this section into your submission document.")
    print(f"{'#' * 66}")

    print(f"""
## Exercise 6: Choose Your Own Network

### Network Choice
- **Network**: {MY_NETWORK}
- **Filter used**: [FILL IN: e.g., --country "China" --max-nodes 5000]
- **Description**: {MY_DESCRIPTION.strip()}

### 1. Network Structure (2-3 sentences)
How does your network differ structurally from the Panama default?
Consider: node count, edge density, degree distribution, node types,
jurisdictions represented.

[YOUR ANSWER HERE]

### 2. MST Comparison (2-3 sentences)
Compare the MST of your network to Panama's. Which has higher average
MST edge weight? What does this tell you about connection strength?
Did Kruskal's or Prim's perform differently on your network vs Panama?

[YOUR ANSWER HERE]

### 3. Cut Properties (2-3 sentences)
Did the Blue Rule hold for all your random cuts? What was the most
interesting jurisdictional cut? What does the minimum crossing edge
represent in your network?

[YOUR ANSWER HERE]

### 4. Clustering (3-4 sentences)
How did MST-based clustering partition your network? Were clusters
balanced or skewed? What do the clusters correspond to in the real
world (e.g., different intermediaries, jurisdictions, time periods)?
How does the cluster structure compare to Panama's?

[YOUR ANSWER HERE]

### 5. Surprising Finding (2-3 sentences)
What was the most surprising or interesting thing you discovered
about your chosen network? Did the MST reveal structure that you
wouldn't have seen by just looking at raw node/edge counts?

[YOUR ANSWER HERE]
""")

    print(f"{'=' * 66}")
    print(f"  Exercise 6 complete!")
    print(f"  Copy the report template above into your submission.")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    main()
