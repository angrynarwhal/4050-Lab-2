#!/usr/bin/env python3
"""
build_graph.py — Convert ICIJ Offshore Leaks CSV data into weighted graph files
suitable for MST and clustering experiments.

Produces two CSV files:
  - {output}_nodes.csv: node_id, label, node_type, jurisdiction, country_codes
  - {output}_edges.csv: source, target, weight, rel_type

Edge weights are derived from:
  - Base weight by relationship type (officer_of=1.0, intermediary_of=2.0, etc.)
  - Penalty for cross-jurisdiction connections (+1.0)
  - Inverse frequency bonus for rare relationship types

Usage:
    python3 build_graph.py --jurisdiction Panama --max-nodes 1000 --output tiny
    python3 build_graph.py --country "United Kingdom" --max-nodes 5000 --output uk_small
    python3 build_graph.py --intermediary "Mossack Fonseca" --max-nodes 10000 --output mf_medium
"""

import os
import sys
import csv
import argparse
import random
from collections import defaultdict, deque

# Paths
RAW_DIR = os.path.join("..", "data", "raw")
OUT_DIR = os.path.join("..", "data")

# Relationship type base weights
# Lower weight = stronger/more direct connection
REL_WEIGHTS = {
    "officer_of": 1.0,
    "registered_address": 3.0,
    "intermediary_of": 2.0,
    "similar_name_and_address": 4.0,
    "same_name_as": 5.0,
    "same_address_as": 4.5,
    "probably_same_officer_as": 3.5,
    "connected_to": 2.5,
    "related_entity": 2.0,
    "underlying": 1.5,
    "beneficiary_of": 1.0,
    "shareholder_of": 1.0,
    "director_of": 1.0,
    "nominee_director_of": 1.5,
    "nominee_shareholder_of": 1.5,
    "secretary_of": 1.5,
    "protector_of": 2.0,
    "authorized_person_of": 2.0,
    "trust_settlor_of": 1.5,
    "trustee_of_trust_of": 1.5,
    "beneficiary_of_trust_of": 1.5,
    "power_of_attorney_of": 2.0,
    "alternate_director_of": 2.0,
}
DEFAULT_WEIGHT = 3.0


def load_nodes(raw_dir):
    """Load all node files and return a dict: node_id -> {name, type, jurisdiction, country_codes}"""
    nodes = {}

    files_and_types = [
        ("nodes-entities.csv", "entity"),
        ("nodes-officers.csv", "officer"),
        ("nodes-intermediaries.csv", "intermediary"),
    ]

    for fname, ntype in files_and_types:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Warning: {fname} not found, skipping.")
            continue

        print(f"  Loading {fname}...")
        count = 0
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                nid = row.get("node_id", "").strip()
                if not nid:
                    continue
                try:
                    nid = int(nid)
                except ValueError:
                    continue

                nodes[nid] = {
                    "name": row.get("name", "Unknown").strip()[:100],
                    "type": ntype,
                    "jurisdiction": row.get("jurisdiction", "").strip(),
                    "jurisdiction_description": row.get("jurisdiction_description", "").strip(),
                    "country_codes": row.get("country_codes", "").strip(),
                    "sourceID": row.get("sourceID", "").strip(),
                }
                count += 1
        print(f"    Loaded {count:,} {ntype} nodes.")

    # Also load addresses for later jurisdiction/country enrichment
    addr_path = os.path.join(raw_dir, "nodes-addresses.csv")
    addresses = {}
    if os.path.exists(addr_path):
        print(f"  Loading nodes-addresses.csv...")
        with open(addr_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                nid = row.get("node_id", "").strip()
                if nid:
                    try:
                        addresses[int(nid)] = row.get("country_codes", "").strip()
                    except ValueError:
                        pass
        print(f"    Loaded {len(addresses):,} addresses.")

    return nodes, addresses


def load_relationships(raw_dir):
    """Load relationships.csv and return list of (start, end, rel_type, link)"""
    fpath = os.path.join(raw_dir, "relationships.csv")
    if not os.path.exists(fpath):
        print(f"  Error: relationships.csv not found in {raw_dir}")
        sys.exit(1)

    print(f"  Loading relationships.csv...")
    edges = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            try:
                start = int(row.get("node_id_start", "").strip())
                end = int(row.get("node_id_end", "").strip())
            except (ValueError, AttributeError):
                continue

            rel_type = row.get("rel_type", "connected_to").strip().lower().replace(" ", "_")
            link = row.get("link", "").strip()
            edges.append((start, end, rel_type, link))

    print(f"    Loaded {len(edges):,} relationships.")
    return edges


def compute_weight(rel_type, node_a, node_b):
    """Compute edge weight based on relationship type and node attributes."""
    base = REL_WEIGHTS.get(rel_type, DEFAULT_WEIGHT)

    # Add penalty for cross-jurisdiction connections
    jur_a = node_a.get("jurisdiction", "")
    jur_b = node_b.get("jurisdiction", "")
    if jur_a and jur_b and jur_a != jur_b:
        base += 1.0

    # Add penalty for cross-country connections
    cc_a = set(node_a.get("country_codes", "").split(";")) - {""}
    cc_b = set(node_b.get("country_codes", "").split(";")) - {""}
    if cc_a and cc_b and not cc_a.intersection(cc_b):
        base += 0.5

    # Add small random noise to break ties (important for unique MST)
    base += random.random() * 0.01

    return round(base, 4)


def filter_by_jurisdiction(nodes, jurisdiction):
    """Return set of node_ids where jurisdiction matches."""
    jur_lower = jurisdiction.lower()
    return {nid for nid, info in nodes.items()
            if jur_lower in info.get("jurisdiction", "").lower()
            or jur_lower in info.get("jurisdiction_description", "").lower()}


def filter_by_country(nodes, country):
    """Return set of node_ids where country_codes contains the country."""
    country_lower = country.lower()
    return {nid for nid, info in nodes.items()
            if country_lower in info.get("country_codes", "").lower()}


def filter_by_intermediary(nodes, edges, intermediary_name):
    """Return set of node_ids connected to intermediaries matching the name."""
    int_lower = intermediary_name.lower()
    intermediary_ids = {nid for nid, info in nodes.items()
                        if info["type"] == "intermediary"
                        and int_lower in info.get("name", "").lower()}

    # Get all nodes connected to these intermediaries
    connected = set(intermediary_ids)
    for start, end, _, _ in edges:
        if start in intermediary_ids:
            connected.add(end)
        if end in intermediary_ids:
            connected.add(start)

    return connected


def extract_largest_connected_component(node_ids, edge_list):
    """Find the largest connected component in the subgraph."""
    # Build adjacency list for the subgraph
    adj = defaultdict(set)
    for u, v, _, _ in edge_list:
        if u in node_ids and v in node_ids:
            adj[u].add(v)
            adj[v].add(u)

    # BFS to find components
    visited = set()
    largest = set()

    for start in node_ids:
        if start in visited:
            continue
        # BFS
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(component) > len(largest):
            largest = component

    return largest


def build_subgraph(nodes, all_edges, target_nodes, max_nodes, seed=42):
    """Build a subgraph from target nodes, limited to max_nodes."""
    random.seed(seed)

    if len(target_nodes) > max_nodes:
        # Sample, but prioritize keeping diverse node types
        by_type = defaultdict(list)
        for nid in target_nodes:
            if nid in nodes:
                by_type[nodes[nid]["type"]].append(nid)

        sampled = set()
        # Ensure representation of each type
        for ntype, nids in by_type.items():
            n_take = max(10, int(max_nodes * len(nids) / len(target_nodes)))
            n_take = min(n_take, len(nids))
            sampled.update(random.sample(nids, n_take))

        # Fill remaining from full set
        remaining = list(target_nodes - sampled)
        random.shuffle(remaining)
        needed = max_nodes - len(sampled)
        if needed > 0:
            sampled.update(remaining[:needed])

        target_nodes = sampled

    # Filter edges to those within target nodes
    sub_edges = []
    for start, end, rel_type, link in all_edges:
        if start in target_nodes and end in target_nodes and start in nodes and end in nodes:
            weight = compute_weight(rel_type, nodes[start], nodes[end])
            sub_edges.append((start, end, weight, rel_type))

    # Extract largest connected component
    edge_tuples = [(s, e, rt, lk) for s, e, rt, lk in
                   [(s, e, rt, "") for s, e, _, rt in sub_edges]]
    lcc = extract_largest_connected_component(target_nodes, edge_tuples)

    if len(lcc) < len(target_nodes):
        print(f"    Note: Extracted largest connected component ({len(lcc):,} of {len(target_nodes):,} nodes)")

    # Filter to LCC
    sub_edges = [(s, e, w, rt) for s, e, w, rt in sub_edges if s in lcc and e in lcc]
    sub_nodes = {nid: nodes[nid] for nid in lcc if nid in nodes}

    # Remove self-loops and duplicate edges
    seen = set()
    clean_edges = []
    for s, e, w, rt in sub_edges:
        if s == e:
            continue
        edge_key = (min(s, e), max(s, e))
        if edge_key not in seen:
            seen.add(edge_key)
            clean_edges.append((s, e, w, rt))

    return sub_nodes, clean_edges


def remap_ids(sub_nodes, sub_edges):
    """Remap node IDs to consecutive integers starting from 0."""
    old_to_new = {}
    new_nodes = {}
    for i, (nid, info) in enumerate(sorted(sub_nodes.items())):
        old_to_new[nid] = i
        new_nodes[i] = {**info, "original_id": nid}

    new_edges = []
    for s, e, w, rt in sub_edges:
        if s in old_to_new and e in old_to_new:
            new_edges.append((old_to_new[s], old_to_new[e], w, rt))

    return new_nodes, new_edges


def write_output(out_nodes, out_edges, output_name, out_dir):
    """Write nodes and edges CSV files."""
    os.makedirs(out_dir, exist_ok=True)

    nodes_path = os.path.join(out_dir, f"{output_name}_nodes.csv")
    edges_path = os.path.join(out_dir, f"{output_name}_edges.csv")

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "label", "node_type", "jurisdiction", "country_codes"])
        for nid in sorted(out_nodes.keys()):
            info = out_nodes[nid]
            writer.writerow([
                nid,
                info.get("name", "Unknown"),
                info.get("type", "unknown"),
                info.get("jurisdiction", ""),
                info.get("country_codes", ""),
            ])

    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight", "rel_type"])
        for s, e, w, rt in sorted(out_edges, key=lambda x: x[2]):
            writer.writerow([s, e, w, rt])

    print(f"\n  Output files:")
    print(f"    Nodes: {nodes_path} ({len(out_nodes):,} nodes)")
    print(f"    Edges: {edges_path} ({len(out_edges):,} edges)")

    # Print summary statistics
    if out_edges:
        weights = [w for _, _, w, _ in out_edges]
        print(f"\n  Edge weight statistics:")
        print(f"    Min: {min(weights):.4f}")
        print(f"    Max: {max(weights):.4f}")
        print(f"    Mean: {sum(weights)/len(weights):.4f}")

    types = defaultdict(int)
    for info in out_nodes.values():
        types[info["type"]] += 1
    print(f"\n  Node type distribution:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c:,}")

    rel_types = defaultdict(int)
    for _, _, _, rt in out_edges:
        rel_types[rt] += 1
    print(f"\n  Relationship type distribution:")
    for rt, c in sorted(rel_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {rt}: {c:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Build weighted graph files from ICIJ Offshore Leaks data"
    )
    parser.add_argument("--raw-dir", default=RAW_DIR,
                        help="Directory containing raw ICIJ CSV files")
    parser.add_argument("--output-dir", default=OUT_DIR,
                        help="Directory for output graph files")
    parser.add_argument("--output", required=True,
                        help="Output name prefix (e.g., 'tiny', 'small', 'panama_medium')")
    parser.add_argument("--max-nodes", type=int, default=5000,
                        help="Maximum number of nodes in the subgraph")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Filtering options (at least one required)
    filter_group = parser.add_argument_group("Filters (use at least one)")
    filter_group.add_argument("--jurisdiction", type=str, default=None,
                              help="Filter by jurisdiction (e.g., 'Panama', 'British Virgin Islands')")
    filter_group.add_argument("--country", type=str, default=None,
                              help="Filter by country code or name (e.g., 'GBR', 'United Kingdom')")
    filter_group.add_argument("--intermediary", type=str, default=None,
                              help="Filter by intermediary name (e.g., 'Mossack Fonseca')")
    filter_group.add_argument("--source", type=str, default=None,
                              help="Filter by sourceID (e.g., 'Panama Papers')")

    args = parser.parse_args()

    # Verify raw data exists
    if not os.path.exists(args.raw_dir):
        print(f"Error: Raw data directory not found: {args.raw_dir}")
        print("Run download_data.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Building graph from ICIJ Offshore Leaks data")
    print("=" * 60)

    # Load data
    nodes, addresses = load_nodes(args.raw_dir)
    all_edges = load_relationships(args.raw_dir)

    print(f"\n  Total: {len(nodes):,} nodes, {len(all_edges):,} edges")

    # Apply filters
    target_nodes = set(nodes.keys())

    if args.jurisdiction:
        jur_nodes = filter_by_jurisdiction(nodes, args.jurisdiction)
        # Also include nodes connected to jurisdiction-filtered nodes
        connected = set()
        for s, e, _, _ in all_edges:
            if s in jur_nodes:
                connected.add(e)
            if e in jur_nodes:
                connected.add(s)
        target_nodes = jur_nodes | (connected & set(nodes.keys()))
        print(f"\n  Jurisdiction filter '{args.jurisdiction}': {len(target_nodes):,} nodes")

    if args.country:
        country_nodes = filter_by_country(nodes, args.country)
        connected = set()
        for s, e, _, _ in all_edges:
            if s in country_nodes:
                connected.add(e)
            if e in country_nodes:
                connected.add(s)
        country_expanded = country_nodes | (connected & set(nodes.keys()))
        target_nodes = target_nodes & country_expanded if args.jurisdiction else country_expanded
        print(f"  Country filter '{args.country}': {len(target_nodes):,} nodes")

    if args.intermediary:
        int_nodes = filter_by_intermediary(nodes, all_edges, args.intermediary)
        target_nodes = target_nodes & int_nodes if (args.jurisdiction or args.country) else int_nodes
        print(f"  Intermediary filter '{args.intermediary}': {len(target_nodes):,} nodes")

    if args.source:
        source_lower = args.source.lower()
        src_nodes = {nid for nid, info in nodes.items()
                     if source_lower in info.get("sourceID", "").lower()}
        target_nodes = target_nodes & src_nodes if (args.jurisdiction or args.country or args.intermediary) else src_nodes
        print(f"  Source filter '{args.source}': {len(target_nodes):,} nodes")

    if not any([args.jurisdiction, args.country, args.intermediary, args.source]):
        print("\n  No filter specified. Using Panama Papers source by default.")
        target_nodes = {nid for nid, info in nodes.items()
                        if "panama" in info.get("sourceID", "").lower()}
        # Include connected nodes
        connected = set()
        for s, e, _, _ in all_edges:
            if s in target_nodes:
                connected.add(e)
            if e in target_nodes:
                connected.add(s)
        target_nodes = target_nodes | (connected & set(nodes.keys()))
        print(f"  Panama Papers default: {len(target_nodes):,} nodes")

    if len(target_nodes) == 0:
        print("Error: No nodes match the filter criteria.")
        sys.exit(1)

    # Build subgraph
    print(f"\n  Building subgraph (max {args.max_nodes:,} nodes)...")
    sub_nodes, sub_edges = build_subgraph(nodes, all_edges, target_nodes,
                                           args.max_nodes, args.seed)

    if len(sub_nodes) == 0:
        print("Error: Resulting subgraph has no nodes. Try different filters.")
        sys.exit(1)

    if len(sub_edges) == 0:
        print("Error: Resulting subgraph has no edges. Try different filters.")
        sys.exit(1)

    # Remap to consecutive IDs
    remapped_nodes, remapped_edges = remap_ids(sub_nodes, sub_edges)

    # Write output
    write_output(remapped_nodes, remapped_edges, args.output, args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"Graph '{args.output}' ready for MST experiments!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
