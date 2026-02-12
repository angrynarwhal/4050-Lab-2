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
    """Return set of node_ids where country_codes contains the country.

    Accepts either:
      - ISO 3-letter codes: "GBR", "USA", "PAN"
      - Full country names: "United Kingdom", "United States", "Panama"
      - Partial matches: "United" will match "United Kingdom", "United States", etc.

    The ICIJ data stores country_codes as semicolon-separated 3-letter ISO codes
    (e.g., "GBR;USA").
    """
    # Common country name -> code mappings
    COUNTRY_NAME_TO_CODES = {
        "united kingdom": ["GBR"], "uk": ["GBR"], "great britain": ["GBR"],
        "england": ["GBR"],
        "united states": ["USA"], "us": ["USA"], "america": ["USA"],
        "panama": ["PAN"],
        "british virgin islands": ["VGB"], "bvi": ["VGB"],
        "cayman islands": ["CYM"],
        "bermuda": ["BMU"],
        "bahamas": ["BHS"],
        "hong kong": ["HKG"],
        "china": ["CHN", "HKG"],
        "switzerland": ["CHE"],
        "luxembourg": ["LUX"],
        "singapore": ["SGP"],
        "jersey": ["JEY"],
        "guernsey": ["GGY"],
        "isle of man": ["IMN"],
        "cyprus": ["CYP"],
        "malta": ["MLT"],
        "seychelles": ["SYC"],
        "samoa": ["WSM"],
        "cook islands": ["COK"],
        "nevis": ["KNA"],
        "barbados": ["BRB"],
        "aruba": ["ABW"],
        "canada": ["CAN"],
        "australia": ["AUS"],
        "germany": ["DEU"],
        "france": ["FRA"],
        "russia": ["RUS"],
        "brazil": ["BRA"],
        "india": ["IND"],
        "japan": ["JPN"],
        "south korea": ["KOR"],
        "taiwan": ["TWN"],
        "mexico": ["MEX"],
        "spain": ["ESP"],
        "italy": ["ITA"],
        "netherlands": ["NLD"],
        "sweden": ["SWE"],
        "norway": ["NOR"],
        "denmark": ["DNK"],
        "ireland": ["IRL"],
        "new zealand": ["NZL"],
        "south africa": ["ZAF"],
        "united arab emirates": ["ARE"], "uae": ["ARE"],
        "saudi arabia": ["SAU"],
        "israel": ["ISR"],
        "argentina": ["ARG"],
        "colombia": ["COL"],
        "venezuela": ["VEN"],
        "nigeria": ["NGA"],
        "kenya": ["KEN"],
        "egypt": ["EGY"],
        "ukraine": ["UKR"],
        "poland": ["POL"],
        "czech republic": ["CZE"],
        "austria": ["AUT"],
        "belgium": ["BEL"],
        "portugal": ["PRT"],
        "greece": ["GRC"],
        "turkey": ["TUR"],
        "thailand": ["THA"],
        "malaysia": ["MYS"],
        "indonesia": ["IDN"],
        "philippines": ["PHL"],
        "vietnam": ["VNM"],
        "chile": ["CHL"],
        "peru": ["PER"],
        "costa rica": ["CRI"],
        "uruguay": ["URY"],
        "ecuador": ["ECU"],
        "liechtenstein": ["LIE"],
        "monaco": ["MCO"],
        "andorra": ["AND"],
        "belize": ["BLZ"],
    }

    country_lower = country.lower().strip()

    # Check if it's already a 3-letter code
    if len(country_lower) == 3 and country_lower.isalpha():
        target_codes = [country.upper()]
    else:
        # Try exact match in mapping
        target_codes = COUNTRY_NAME_TO_CODES.get(country_lower)

        if not target_codes:
            # Try partial match in mapping keys
            for name, codes in COUNTRY_NAME_TO_CODES.items():
                if country_lower in name or name in country_lower:
                    target_codes = codes
                    break

        if not target_codes:
            # Fall back to substring search in the raw country_codes field
            print(f"    Note: '{country}' not in country mapping, "
                  f"trying substring match on country_codes field...")
            return {nid for nid, info in nodes.items()
                    if country_lower in info.get("country_codes", "").lower()}

    print(f"    Resolved '{country}' to code(s): {target_codes}")

    matched = set()
    for nid, info in nodes.items():
        cc = info.get("country_codes", "")
        if cc:
            node_codes = [c.strip().upper() for c in cc.split(";")]
            if any(tc in node_codes for tc in target_codes):
                matched.add(nid)

    return matched


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
    """
    Build a dense, connected subgraph using BFS expansion + bipartite projection.

    The ICIJ graph is inherently bipartite: officers connect to entities,
    intermediaries connect to entities, etc. Direct officer↔officer or
    entity↔entity edges are rare. This means BFS expansion produces tree-like
    subgraphs with few cycles — useless for MST exercises.

    Solution: after BFS expansion, add PROJECTION EDGES. Two visited nodes
    get a projection edge if they share a common neighbor in the full graph
    (even if that neighbor is outside the visited set). This is the standard
    bipartite graph projection and creates natural clusters:
      - Entities sharing an intermediary cluster together
      - Officers connected to the same entity cluster together

    Edge weights for projections: higher (weaker) than direct edges, scaled
    inversely by the number of shared neighbors (more shared = stronger link).
    """
    random.seed(seed)

    print(f"    Building full adjacency for {len(target_nodes):,} target nodes...")

    # ---- Step 1: Build adjacency within target set (for BFS) ----
    adj_target = defaultdict(set)
    edge_lookup = {}
    target_set = set(target_nodes)

    for start, end, rel_type, link in all_edges:
        if start in target_set and end in target_set and start != end:
            adj_target[start].add(end)
            adj_target[end].add(start)
            ekey = (min(start, end), max(start, end))
            if ekey not in edge_lookup:
                edge_lookup[ekey] = rel_type

    nodes_with_edges = {n for n in target_set if len(adj_target[n]) > 0}
    print(f"    Nodes with internal edges: {len(nodes_with_edges):,}")
    print(f"    Direct edges: {len(edge_lookup):,}")

    if len(nodes_with_edges) == 0:
        print("    Warning: No edges found among target nodes.")
        return {}, []

    # ---- Step 2: BFS expansion from highest-degree seed ----
    # Start from ONE seed to guarantee a connected subgraph.
    # Only add additional seeds if BFS exhausts its component before max_nodes.
    degree_list = sorted(
        [(nid, len(adj_target[nid])) for nid in nodes_with_edges],
        key=lambda x: -x[1]
    )

    primary_seed = degree_list[0][0]
    info = nodes.get(primary_seed, {})
    print(f"    Primary seed: {info.get('name', '?')[:60]} "
          f"(type={info.get('type','?')}, degree={len(adj_target[primary_seed])})")

    visited = set([primary_seed])
    queue = deque([primary_seed])

    while len(visited) < max_nodes:
        if not queue:
            # Current component exhausted — try next highest-degree unvisited node
            next_seed = None
            for nid, _ in degree_list:
                if nid not in visited:
                    next_seed = nid
                    break
            if next_seed is None:
                break
            info = nodes.get(next_seed, {})
            print(f"    Adding seed (new component): {info.get('name', '?')[:50]} "
                  f"(degree={len(adj_target[next_seed])})")
            visited.add(next_seed)
            queue.append(next_seed)

        node = queue.popleft()
        neighbors = list(adj_target[node])
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited and len(visited) < max_nodes:
                visited.add(neighbor)
                queue.append(neighbor)

    print(f"    BFS reached {len(visited):,} nodes")

    # ---- Step 3: Collect direct edges between visited nodes ----
    sub_edges = []
    seen_edges = set()

    for s in visited:
        for e in adj_target[s]:
            if e in visited:
                ekey = (min(s, e), max(s, e))
                if ekey not in seen_edges:
                    seen_edges.add(ekey)
                    rel_type = edge_lookup.get(ekey, "connected_to")
                    weight = compute_weight(rel_type, nodes.get(s, {}), nodes.get(e, {}))
                    sub_edges.append((s, e, weight, rel_type))

    direct_count = len(sub_edges)
    print(f"    Direct edges in subgraph: {direct_count:,}")

    # ---- Step 4: Add PROJECTION EDGES (bipartite projection) ----
    # The ICIJ graph is bipartite: officers↔entities↔intermediaries.
    # BFS expansion follows this tree structure, so direct edges alone
    # give a tree-like graph. We need projection edges to create cycles.
    #
    # For EVERY node (visited or not) that connects to 2+ visited nodes,
    # add edges between pairs of those visited neighbors. This creates
    # triangles and dense clusters:
    #   - Two officers sharing an entity get connected
    #   - Two entities sharing an intermediary get connected
    #   - Two entities sharing an officer get connected

    print(f"    Computing projection edges (shared-neighbor links)...")

    # Build mapping: hub_node → set of visited neighbors
    # Include both visited and non-visited hubs
    hub_to_visited_neighbors = defaultdict(set)

    for start, end, rel_type, link in all_edges:
        s_in = start in visited
        e_in = end in visited
        if s_in and e_in:
            # Both visited: each is a hub for the other's neighbors
            hub_to_visited_neighbors[start].add(end)
            hub_to_visited_neighbors[end].add(start)
        elif s_in:
            # start visited, end is external bridge
            hub_to_visited_neighbors[end].add(start)
        elif e_in:
            # end visited, start is external bridge
            hub_to_visited_neighbors[start].add(end)

    # Count hubs with 2+ visited neighbors
    projectable = {h: vn for h, vn in hub_to_visited_neighbors.items()
                   if len(vn) >= 2}
    print(f"    Hubs with 2+ visited neighbors: {len(projectable):,}")

    projection_count = 0
    MAX_PAIRS_PER_HUB = 15  # Cap pairs per hub to avoid O(n²) from mega-hubs

    for hub_node, visited_neighbors in projectable.items():
        vn_list = list(visited_neighbors)
        num_neighbors = len(vn_list)

        # For very large hubs, sample neighbors to keep edge count manageable
        if num_neighbors > 30:
            vn_list = random.sample(vn_list, 30)

        # Create pairs — cap total pairs per hub
        pairs_added = 0
        random.shuffle(vn_list)

        for i in range(len(vn_list)):
            if pairs_added >= MAX_PAIRS_PER_HUB:
                break
            for j in range(i + 1, len(vn_list)):
                if pairs_added >= MAX_PAIRS_PER_HUB:
                    break
                s, e = vn_list[i], vn_list[j]
                ekey = (min(s, e), max(s, e))
                if ekey not in seen_edges:
                    seen_edges.add(ekey)
                    # Weight based on shared-neighbor count:
                    #   Many shared neighbors → lower weight (stronger tie)
                    #   Few shared neighbors → higher weight (weaker tie)
                    # Range: 3.0 (very connected) to 6.0 (loosely connected)
                    strength = min(num_neighbors, 20) / 20.0  # 0.0 to 1.0
                    base = 6.0 - strength * 3.0  # 6.0 down to 3.0
                    weight = base + random.random() * 0.01
                    weight = round(weight, 4)
                    sub_edges.append((s, e, weight, "shared_connection"))
                    projection_count += 1
                    pairs_added += 1

    print(f"    Projection edges added: {projection_count:,}")

    # ---- Step 5: Summary ----
    sub_nodes = {nid: nodes[nid] for nid in visited if nid in nodes}

    total_edges = len(sub_edges)
    print(f"    Final subgraph: {len(sub_nodes):,} nodes, {total_edges:,} edges")
    if len(sub_nodes) > 0:
        ratio = total_edges / len(sub_nodes)
        print(f"    Edge/node ratio: {ratio:.2f} "
              f"(direct: {direct_count:,}, projected: {projection_count:,})")
        if ratio < 1.5:
            print(f"    Warning: ratio still low. Try a larger --max-nodes or "
                  f"different filter.")

    return sub_nodes, sub_edges


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
                              help="Filter by country name or ISO code "
                                   "(e.g., 'GBR', 'United Kingdom', 'USA', 'Panama')")
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
        # Expand to 2-hop neighborhood so the target pool has internal edges
        # (Officers connect to entities; entities connect to intermediaries;
        #  2 hops captures officer→entity→intermediary triangles)
        connected_1hop = set()
        for s, e, _, _ in all_edges:
            if s in jur_nodes:
                connected_1hop.add(e)
            if e in jur_nodes:
                connected_1hop.add(s)
        connected_1hop = connected_1hop & set(nodes.keys())

        connected_2hop = set()
        for s, e, _, _ in all_edges:
            if s in connected_1hop and e in nodes:
                connected_2hop.add(e)
            if e in connected_1hop and s in nodes:
                connected_2hop.add(s)

        target_nodes = jur_nodes | connected_1hop | connected_2hop
        print(f"\n  Jurisdiction filter '{args.jurisdiction}': "
              f"{len(jur_nodes):,} direct + {len(target_nodes)-len(jur_nodes):,} "
              f"neighbors = {len(target_nodes):,} nodes")

    if args.country:
        country_nodes = filter_by_country(nodes, args.country)
        # 2-hop expansion
        connected_1hop = set()
        for s, e, _, _ in all_edges:
            if s in country_nodes and e in nodes:
                connected_1hop.add(e)
            if e in country_nodes and s in nodes:
                connected_1hop.add(s)

        connected_2hop = set()
        for s, e, _, _ in all_edges:
            if s in connected_1hop and e in nodes:
                connected_2hop.add(e)
            if e in connected_1hop and s in nodes:
                connected_2hop.add(s)

        country_expanded = country_nodes | connected_1hop | connected_2hop
        target_nodes = target_nodes & country_expanded if args.jurisdiction else country_expanded
        print(f"  Country filter '{args.country}': "
              f"{len(country_nodes):,} direct, {len(target_nodes):,} with expansion")

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
