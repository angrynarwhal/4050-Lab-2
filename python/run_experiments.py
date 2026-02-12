#!/usr/bin/env python3
"""
run_experiments.py — Guided experiment runner for Lab 2.

Runs all exercises in sequence, collecting data for the lab submission.

Usage:
    python3 run_experiments.py [--data-dir ../data] [--sizes tiny,small,medium]
"""

import os
import sys
import csv
import time
import argparse

from mst_algorithms import load_graph, kruskal, prim, UnionFind
from cut_properties import (verify_blue_rule, verify_blue_rule_jurisdictional,
                             find_cycles_via_non_tree_edges)
from clustering import mst_clustering, analyze_clusters


def check_data_files(data_dir, sizes):
    """Check which data files exist."""
    available = []
    missing = []
    for size in sizes:
        nodes_f = os.path.join(data_dir, f"{size}_nodes.csv")
        edges_f = os.path.join(data_dir, f"{size}_edges.csv")
        if os.path.exists(nodes_f) and os.path.exists(edges_f):
            available.append(size)
        else:
            missing.append(size)
    return available, missing


def run_exercise_1(data_dir, sizes):
    """Exercise 1: MST Construction and Comparison"""
    print("\n" + "=" * 60)
    print("EXERCISE 1: MST Construction and Comparison")
    print("=" * 60)

    results = []
    for size in sizes:
        nodes_f = os.path.join(data_dir, f"{size}_nodes.csv")
        edges_f = os.path.join(data_dir, f"{size}_edges.csv")

        print(f"\n--- {size} ---")
        nodes, edges, adj = load_graph(nodes_f, edges_f)
        n_nodes = len(nodes)
        n_edges = len(edges)

        # Kruskal's
        k_edges, k_weight, k_stats, k_time = kruskal(nodes, edges)

        # Prim's
        p_edges, p_weight, p_stats, p_time = prim(nodes, adj)

        result = {
            "size": size,
            "nodes": n_nodes,
            "edges": n_edges,
            "kruskal_time": k_time,
            "prim_time": p_time,
            "kruskal_weight": k_weight,
            "prim_weight": p_weight,
            "mst_edges": len(k_edges),
        }
        results.append(result)

        print(f"  Nodes: {n_nodes:,}, Edges: {n_edges:,}")
        print(f"  Kruskal: weight={k_weight:.4f}, time={k_time:.4f}s")
        print(f"  Prim:    weight={p_weight:.4f}, time={p_time:.4f}s")
        print(f"  Weight diff: {abs(k_weight - p_weight):.6f}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Network':<10} {'Nodes':>7} {'Edges':>7} {'Kruskal(s)':>11} "
          f"{'Prim(s)':>9} {'MST Weight':>12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['size']:<10} {r['nodes']:>7,} {r['edges']:>7,} "
              f"{r['kruskal_time']:>11.4f} {r['prim_time']:>9.4f} "
              f"{r['kruskal_weight']:>12.4f}")

    return results


def run_exercise_2(data_dir, size="small"):
    """Exercise 2: Blue Rule Verification"""
    print("\n" + "=" * 60)
    print("EXERCISE 2: Blue Rule (Cut Property) Verification")
    print("=" * 60)

    nodes_f = os.path.join(data_dir, f"{size}_nodes.csv")
    edges_f = os.path.join(data_dir, f"{size}_edges.csv")

    nodes, edges, adj = load_graph(nodes_f, edges_f)
    mst_edges, _, _, _ = kruskal(nodes, edges)

    print(f"\nUsing {size} dataset ({len(nodes):,} nodes, {len(edges):,} edges)")
    print(f"\nRandom cut trials:")
    results = verify_blue_rule(nodes, edges, adj, mst_edges, num_trials=10, seed=42)

    print(f"\nJurisdictional cut:")
    jur_result = verify_blue_rule_jurisdictional(nodes, edges, mst_edges)

    return results, jur_result


def run_exercise_3(data_dir, size="small"):
    """Exercise 3: Red Rule Verification"""
    print("\n" + "=" * 60)
    print("EXERCISE 3: Red Rule (Cycle Property) Verification")
    print("=" * 60)

    nodes_f = os.path.join(data_dir, f"{size}_nodes.csv")
    edges_f = os.path.join(data_dir, f"{size}_edges.csv")

    nodes, edges, adj = load_graph(nodes_f, edges_f)
    mst_edges, _, _, _ = kruskal(nodes, edges)

    print(f"\nUsing {size} dataset ({len(nodes):,} nodes, {len(edges):,} edges)")
    print(f"\nCycle analysis:")
    results = find_cycles_via_non_tree_edges(
        nodes, edges, mst_edges, adj, num_cycles=10, seed=42
    )

    return results


def run_exercise_4(data_dir, size="small"):
    """Exercise 4: MST-Based Clustering"""
    print("\n" + "=" * 60)
    print("EXERCISE 4: MST-Based Clustering")
    print("=" * 60)

    nodes_f = os.path.join(data_dir, f"{size}_nodes.csv")
    edges_f = os.path.join(data_dir, f"{size}_edges.csv")

    nodes, edges, adj = load_graph(nodes_f, edges_f)
    mst_edges, total_weight, _, _ = kruskal(nodes, edges)

    print(f"\nUsing {size} dataset ({len(nodes):,} nodes, {len(edges):,} edges)")

    results = []
    for k in [3, 5, 10, 20]:
        if k > len(nodes):
            print(f"\n  k={k} > num_nodes, skipping")
            continue

        print(f"\n--- k={k} clusters ---")
        clusters, removed, remaining = mst_clustering(mst_edges, len(nodes), k)
        sizes = sorted([len(m) for m in clusters.values()], reverse=True)

        removed_weights = [w for _, _, w, _ in removed]

        result = {
            "k": k,
            "sizes": sizes,
            "largest": sizes[0],
            "smallest": sizes[-1],
            "removed_weights": removed_weights,
        }
        results.append(result)

        print(f"  Cluster sizes: {sizes}")
        print(f"  Largest: {sizes[0]}, Smallest: {sizes[-1]}")
        if removed_weights:
            print(f"  Removed edge weights: "
                  f"{', '.join(f'{w:.4f}' for w in sorted(removed_weights, reverse=True)[:5])}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'k':>3} {'Largest':>8} {'Smallest':>9} {'Max Removed':>12} {'Min Removed':>12}")
    print(f"{'-'*70}")
    for r in results:
        rw = r["removed_weights"]
        print(f"{r['k']:>3} {r['largest']:>8,} {r['smallest']:>9,} "
              f"{max(rw):>12.4f} {min(rw):>12.4f}")

    return results


def run_exercise_5(data_dir, size="small"):
    """Exercise 5: Union-Find Performance"""
    print("\n" + "=" * 60)
    print("EXERCISE 5: Union-Find Performance Comparison")
    print("=" * 60)

    nodes_f = os.path.join(data_dir, f"{size}_nodes.csv")
    edges_f = os.path.join(data_dir, f"{size}_edges.csv")

    nodes, edges, adj = load_graph(nodes_f, edges_f)

    print(f"\nUsing {size} dataset ({len(nodes):,} nodes, {len(edges):,} edges)")

    configs = [
        ("No optimizations", False, False),
        ("Path compression only", True, False),
        ("Union by rank only", False, True),
        ("Both optimizations", True, True),
    ]

    results = []
    for name, pc, ubr in configs:
        print(f"\n--- {name} ---")
        _, _, stats, elapsed = kruskal(
            nodes, edges, path_compression=pc, union_by_rank=ubr
        )
        print(f"  Time: {elapsed:.4f}s")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        results.append({"config": name, "time": elapsed, **stats})

    # Summary
    print(f"\n{'=' * 80}")
    print(f"{'Configuration':<25} {'Time(s)':>8} {'Find Calls':>11} "
          f"{'Avg Path':>9} {'Max Height':>11}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['config']:<25} {r['time']:>8.4f} {r['find_calls']:>11,} "
              f"{r['avg_path_length']:>9.4f} {r['max_height']:>11}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all Lab 2 experiments")
    parser.add_argument("--data-dir", default=os.path.join("..", "data"),
                        help="Directory containing graph data files")
    parser.add_argument("--sizes", default="tiny,small",
                        help="Comma-separated list of dataset sizes to use")
    parser.add_argument("--exercises", default="1,2,3,4,5",
                        help="Comma-separated list of exercises to run (1-5)")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    exercises = [int(e.strip()) for e in args.exercises.split(",")]

    # Check data availability
    available, missing = check_data_files(args.data_dir, sizes)
    if missing:
        print(f"Warning: Data files missing for sizes: {missing}")
        print(f"Run build_graph.py first to generate them.")
        if not available:
            print("No data files found. Exiting.")
            sys.exit(1)
        print(f"Proceeding with available sizes: {available}")
        sizes = available

    print("=" * 60)
    print("CS4050 Lab 2: MST Experiments")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset sizes: {sizes}")
    print(f"Exercises: {exercises}")

    if 1 in exercises:
        run_exercise_1(args.data_dir, sizes)

    # Use first available size for single-dataset exercises
    default_size = sizes[0] if len(sizes) == 1 else (sizes[1] if len(sizes) > 1 else sizes[0])

    if 2 in exercises:
        run_exercise_2(args.data_dir, default_size)

    if 3 in exercises:
        run_exercise_3(args.data_dir, default_size)

    if 4 in exercises:
        run_exercise_4(args.data_dir, default_size)

    if 5 in exercises:
        run_exercise_5(args.data_dir, default_size)

    print(f"\n{'=' * 60}")
    print("All experiments complete!")
    print("Copy your results into lab2_submission/ for your submission.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
