# CS4050 Lab 2: Minimum Spanning Trees, Cut Properties, and Clustering with the Panama Papers

Place all of your submission documents in the [./lab2_submission](lab2_submission) folder. You will submit a zip file of that directory in Canvas.

## Overview

This lab explores **Minimum Spanning Trees (MSTs)**, the **Red Rule**, the **Blue Rule**, and **graph-based clustering** using real-world data from the [ICIJ Offshore Leaks Database](https://offshoreleaks.icij.org/) — the dataset behind the Panama Papers, Paradise Papers, and Pandora Papers investigations.

The ICIJ data is a graph: **entities** (offshore companies), **officers** (people), **intermediaries** (law firms), and **addresses** are nodes. The relationships between them are edges. You will:

1. **Load** real offshore financial network data and construct weighted graphs
2. **Implement** Kruskal's and Prim's algorithms for MST construction
3. **Apply** the Red Rule and Blue Rule to reason about edge inclusion/exclusion
4. **Use** MST-based clustering to discover communities in offshore networks
5. **Analyze** how cuts, cycles, and spanning trees reveal hidden structure in financial networks

## Why the Panama Papers?

The Panama Papers represent one of the largest data leaks in history — 11.5 million documents exposing a global network of offshore shell companies. The underlying data is *inherently a graph problem*: people connect to companies through intermediaries across jurisdictions. Minimum spanning trees and clustering can reveal the most critical connections and community structure in these networks.

There are legitimate uses for offshore companies and trusts. **The inclusion of a person or entity in the ICIJ Offshore Leaks Database is not intended to suggest or imply that they have engaged in illegal or improper conduct.** We use this data purely as a real-world graph for algorithmic exploration.

## Learning Objectives

By completing this lab, you will:

* Implement **Kruskal's algorithm** using Union-Find (disjoint sets) and understand its greedy strategy
* Implement **Prim's algorithm** using a priority queue and understand its growth strategy
* Apply the **Blue Rule** (lightest edge crossing a cut must be in every MST) through direct experimentation
* Apply the **Red Rule** (heaviest edge on a cycle can be excluded from every MST) through direct experimentation
* Use MST edge deletion for **k-clustering** and understand the relationship between MSTs and optimal clustering
* Understand **cuts, cycles, and the cut property** as the theoretical foundation for MST correctness
* Work with real-world graph data at scale

## Data: ICIJ Offshore Leaks

### Downloading the Data

```bash
# Download the full ICIJ Offshore Leaks database (CSV format)
cd scripts
python3 download_data.py

# Or download manually from:
# https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip
# Unzip into the data/raw/ directory
```

### Data Structure

The ICIJ data consists of **node files** and a **relationship file**:

| File | Description | Key Columns |
|------|-------------|-------------|
| `nodes-entities.csv` | Offshore companies, trusts, foundations | `node_id`, `name`, `jurisdiction`, `jurisdiction_description`, `country_codes`, `incorporation_date`, `status` |
| `nodes-officers.csv` | Directors, shareholders, beneficiaries | `node_id`, `name`, `country_codes` |
| `nodes-intermediaries.csv` | Law firms, registered agents | `node_id`, `name`, `country_codes`, `status` |
| `nodes-addresses.csv` | Physical/mailing addresses | `node_id`, `address`, `country_codes` |
| `relationships.csv` | Edges connecting all node types | `node_id_start`, `node_id_end`, `rel_type`, `link` |

### Graph Construction

Our scripts convert this relational data into a weighted graph where:

* **Nodes** = entities, officers, and intermediaries (addresses become node attributes)
* **Edges** = relationships between nodes
* **Edge weights** = derived from relationship type, shared jurisdictions, and connection density (see `build_graph.py` for details)

Because the full dataset has 800,000+ nodes, we provide extraction scripts that create manageable subgraphs by jurisdiction, country, or intermediary.

## Quick Start

```bash
# 1. Download and prepare data
cd scripts
python3 download_data.py
python3 build_graph.py --jurisdiction "Panama" --max-nodes 1000 --output tiny
python3 build_graph.py --jurisdiction "Panama" --max-nodes 5000 --output small
python3 build_graph.py --jurisdiction "Panama" --max-nodes 20000 --output medium
python3 build_graph.py --country "GBR" --max-nodes 5000 --output uk_small
# Or equivalently: --country "United Kingdom"

# 2. Run MST experiments
cd ../python
python3 run_experiments.py

# 3. Or run individual components
python3 mst_algorithms.py ../data/tiny_nodes.csv ../data/tiny_edges.csv --algorithm kruskal
python3 mst_algorithms.py ../data/tiny_nodes.csv ../data/tiny_edges.csv --algorithm prim
python3 clustering.py ../data/small_nodes.csv ../data/small_edges.csv --clusters 5
```

## Network Sizes and Expected Behavior

| Size | Nodes | Edges | MST Behavior |
|------|-------|-------|--------------|
| tiny | ~1,000 | ~2,000 | Both algorithms fast, good for debugging and visualization |
| small | ~5,000 | ~12,000 | Noticeable difference between Prim's and Kruskal's |
| medium | ~20,000 | ~50,000 | Sort cost in Kruskal's becomes visible; Prim's with binary heap competitive |
| large | ~50,000 | ~120,000 | Performance differences clear; clustering reveals jurisdictional communities |
| uk_small | ~5,000 | ~10,000 | Country-filtered subgraph for comparative analysis |

## Repository Structure

```
4050-Lab-2/
├── data/                           # Generated graph files (after running scripts)
│   ├── tiny_nodes.csv
│   ├── tiny_edges.csv
│   ├── small_nodes.csv
│   ├── small_edges.csv
│   └── ...
├── lab2_submission/                 # YOUR submission goes here
│   ├── .keep
│   └── <<Your Submission in Markdown Format>>
├── scripts/
│   ├── download_data.py            # Downloads ICIJ data
│   └── build_graph.py              # Converts ICIJ CSVs to graph format
├── python/
│   ├── mst_algorithms.py           # Kruskal's, Prim's, Union-Find
│   ├── cut_properties.py           # Red Rule / Blue Rule demonstrations
│   ├── clustering.py               # MST-based k-clustering
│   ├── run_experiments.py          # Guided experiment runner (Exercises 1-5)
│   └── explore_network.py          # Exercise 6: Choose Your Own Network
├── c/
│   ├── mst.c                       # C implementations of MST algorithms
│   ├── union_find.c                # Union-Find with path compression + union by rank
│   ├── union_find.h
│   ├── Makefile
│   └── README.md
└── README.md                       # This file
```

## Background: Key Concepts

### Minimum Spanning Tree (MST)

Given a connected, weighted, undirected graph G = (V, E, w), a **minimum spanning tree** is a subset T ⊆ E such that:
1. T connects all vertices (spans)
2. T contains no cycles (is a tree)
3. The total weight Σw(e) for e ∈ T is minimized

For |V| = n vertices, the MST always has exactly **n - 1** edges.

### The Cut Property (Blue Rule)

A **cut** is a partition of vertices into two non-empty sets (S, V\S). An edge **crosses** the cut if one endpoint is in S and the other in V\S.

**Blue Rule:** For any cut of the graph, the minimum-weight edge crossing that cut **must** be in every MST (assuming unique edge weights). If weights are not unique, it is in *at least one* MST.

This is why both Prim's and Kruskal's work: they always select the lightest edge crossing some cut.

### The Cycle Property (Red Rule)

**Red Rule:** For any cycle in the graph, the maximum-weight edge on that cycle is **not** in any MST (assuming unique edge weights).

This provides a way to *exclude* edges: if you find a cycle, the heaviest edge on it is safe to remove.

### MST-Based Clustering

To find **k clusters** in a graph:
1. Compute the MST (n - 1 edges)
2. Remove the **k - 1 heaviest** edges from the MST
3. The resulting forest has exactly **k connected components** = k clusters

This works because MST edges represent the "cheapest" connections. Removing the most expensive MST edges severs the weakest links between natural communities.

## Understanding the Code

### Union-Find (Disjoint Sets)

Kruskal's algorithm needs to efficiently determine whether two nodes are in the same component. Union-Find supports this with near-constant time operations:

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Each node is its own parent
        self.rank = [0] * n            # Rank for union by rank

    def find(self, x):
        # Path compression: point directly to root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Union by rank: attach shorter tree under taller
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # Already in same set — adding edge would create cycle
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
```

**Complexity:** With path compression and union by rank, both `find` and `union` run in **O(α(n))** amortized time, where α is the inverse Ackermann function — effectively constant.

### Kruskal's Algorithm

```
KRUSKAL(G):
    Sort all edges by weight
    Initialize Union-Find with |V| components
    T = empty set
    For each edge (u, v, w) in sorted order:
        If find(u) ≠ find(v):        # u and v in different components
            Add (u, v, w) to T
            union(u, v)
    Return T
```

**Complexity:** O(E log E) dominated by the sort. The Union-Find operations are nearly O(1) each.

**Why it works:** Each edge added is the lightest edge crossing the cut between the two components being merged (Blue Rule).

### Prim's Algorithm

```
PRIM(G, start):
    T = empty set
    visited = {start}
    PQ = priority queue of edges from start
    While |T| < |V| - 1:
        (w, u, v) = extract-min from PQ
        If v not in visited:
            Add (u, v, w) to T
            visited.add(v)
            For each neighbor n of v:
                If n not in visited:
                    Insert (weight(v,n), v, n) into PQ
    Return T
```

**Complexity:** O(E log V) with a binary heap. O(E + V log V) with a Fibonacci heap.

**Why it works:** The tree grows one vertex at a time. Each added edge is the lightest crossing the cut (visited, unvisited) — a direct application of the Blue Rule.

### Key Operations Comparison

| Operation | Kruskal's | Prim's (Binary Heap) |
|-----------|-----------|---------------------|
| Overall | O(E log E) | O(E log V) |
| Best for | Sparse graphs | Dense graphs |
| Data structure | Union-Find | Priority Queue |
| Strategy | Global sort, greedy add | Grow from single vertex |
| Parallelizable? | Sort is; rest is sequential | Not easily |

## Lab Exercises

### Exercise 1: MST Construction and Comparison

```bash
cd c
make
./mst ../data/tiny_nodes.csv ../data/tiny_edges.csv
./mst ../data/small_nodes.csv ../data/small_edges.csv
./mst ../data/medium_nodes.csv ../data/medium_edges.csv
./mst ../data/uk_small_nodes.csv ../data/uk_small_edges.csv

```

Run both algorithms on progressively larger Panama Papers subgraphs:

| Network | Nodes | Edges | Kruskal Time | Prim Time | MST Weight |
|---------|-------|-------|-------------|-----------|------------|
| tiny    |       |       |             |           |            |
| small   |       |       |             |           |            |
| medium  |       |       |             |           |            |

**Questions:**

1. Do both algorithms produce the same total MST weight? Why or why not?
2. At what scale does the performance difference become visible? Which is faster and why?
3. For the Panama Papers graph structure (sparse, with high-degree intermediary nodes), which algorithm is more natural? Explain in terms of the graph's degree distribution.

### Exercise 2: The Blue Rule in Action

The script `cut_properties.py` lets you define cuts and verify the Blue Rule:

```bash
python3 cut_properties.py ../data/small_nodes.csv ../data/small_edges.csv --exercise blue
```

This will:
- Select a random partition of vertices into sets S and V\S
- Identify all edges crossing the cut
- Find the minimum-weight crossing edge
- Verify that this edge appears in the MST

**Run this 10 times and record your results:**

| Trial | Cut Size (|S|) | Crossing Edges | Min Crossing Edge Weight | In MST? |
|-------|---------------|----------------|-------------------------|---------|
| 1     |               |                |                         |         |
| 2     |               |                |                         |         |
| ...   |               |                |                         |         |
| 10    |               |                |                         |         |

**Questions:**

1. Was the minimum-weight crossing edge always in the MST? Explain any exceptions (hint: consider duplicate weights).
2. Consider a cut that separates Panama-based entities from British Virgin Islands entities. What does the minimum crossing edge represent in the context of the Panama Papers?
3. If you were an investigator, why would the *minimum-weight* edge crossing a jurisdictional cut be significant?

### Exercise 3: The Red Rule in Action

```bash
python3 cut_properties.py ../data/small_nodes.csv ../data/small_edges.csv --exercise red
```

This will:
- Find cycles in the graph
- Identify the maximum-weight edge on each cycle
- Verify that the heaviest edge on each cycle is NOT in the MST

**Run this and record results for 10 cycles:**

| Cycle | Cycle Length | Max Edge Weight | Max Edge in MST? |
|-------|-------------|-----------------|-------------------|
| 1     |             |                 |                   |
| 2     |             |                 |                   |
| ...   |             |                 |                   |

**Questions:**

1. Was the maximum-weight cycle edge ever in the MST? Under what conditions could it be?
2. In the Panama Papers network, a cycle might represent: Officer → Entity A → Intermediary → Entity B → Officer. What does the *heaviest* edge on such a cycle represent? Why is it "redundant" from the MST perspective?
3. How do the Red Rule and Blue Rule together guarantee MST correctness? Explain in your own words.

### Exercise 4: MST-Based Clustering — Discovering Communities

This is the core exercise. Use the MST to discover communities in the offshore network:

```bash
python3 clustering.py ../data/small_nodes.csv ../data/small_edges.csv --clusters 3
python3 clustering.py ../data/small_nodes.csv ../data/small_edges.csv --clusters 5
python3 clustering.py ../data/small_nodes.csv ../data/small_edges.csv --clusters 10
python3 clustering.py ../data/small_nodes.csv ../data/small_edges.csv --clusters 20
```

For each value of k, record:

| k | Removed Edge Weights | Largest Cluster | Smallest Cluster | Cluster Sizes (all) |
|---|---------------------|-----------------|------------------|---------------------|
| 3 |                     |                 |                  |                     |
| 5 |                     |                 |                  |                     |
| 10|                     |                 |                  |                     |
| 20|                     |                 |                  |                     |

**Questions:**

1. As k increases, what happens to the distribution of cluster sizes? Do you get many small clusters and a few large ones, or roughly equal sizes?
2. Examine the clusters for k=5. Do the clusters correspond to jurisdictions, intermediaries, or some other organizational principle? What pattern do you observe?
3. The edges removed to form clusters are the heaviest MST edges. What do these "expensive" connections represent in the offshore network? Why are they natural cluster boundaries?
4. This is the same algorithm used for single-linkage clustering in statistics. Why is it called "single-linkage"? How does that name relate to how the MST connects components?

### Exercise 5: Union-Find Performance

The C implementation includes instrumented Union-Find. Run Kruskal's and observe:

```bash
cd c
make
./mst ../data/small_nodes.csv ../data/small_edges.csv --instrument
```

Record:

| Metric | Without Path Compression | With Path Compression | With Both Optimizations |
|--------|-------------------------|----------------------|------------------------|
| Total find() calls |                   |                      |                        |
| Average path length |                  |                      |                        |
| Max tree height |                      |                      |                        |
| Total time |                           |                      |                        |

**Questions:**

1. What is the maximum possible height of a Union-Find tree without optimizations? What about with union by rank only? With both?
2. Path compression changes the tree structure during `find()`. Draw the before and after for a chain of 5 elements: 0→1→2→3→4 (where 4 is root). What does path compression do?
3. The inverse Ackermann function α(n) grows incredibly slowly (α(2^65536) = 4). Why does this matter for practical MST computation?

### Exercise 6: Choose Your Own Network

In this exercise you choose your own slice of the ICIJ database, run the full analysis pipeline, and compare your network to the Panama default.

**Step 1: Pick a network.** Choose ONE of these approaches:

```bash
cd scripts

# Option A: A different country
python3 build_graph.py --country "China" --max-nodes 5000 --output my_network

# Option B: A different jurisdiction
python3 build_graph.py --jurisdiction "British Virgin Islands" --max-nodes 5000 --output my_network

# Option C: A specific intermediary (law firm)
python3 build_graph.py --intermediary "Appleby" --max-nodes 5000 --output my_network

# Option D: A different data source
python3 build_graph.py --source "Paradise Papers" --max-nodes 5000 --output my_network

# Option E: Combine filters
python3 build_graph.py --country "Russia" --jurisdiction "Panama" --max-nodes 5000 --output my_network
```

**Step 2: Configure the template.** Open `python/explore_network.py` and edit the `CONFIGURATION` section at the top:
- Set `MY_NETWORK` to the `--output` name you used above
- Write a 2-3 sentence description of what you chose and why

**Step 3: Run the analysis.**

```bash
cd python
python3 explore_network.py
```

This will automatically:
- Profile your network (degree distribution, node types, jurisdictions)
- Build MSTs using both Kruskal's and Prim's
- Verify Blue Rule and Red Rule properties
- Run clustering for k = 3, 5, 10, 20 with detailed composition analysis
- Compare Union-Find performance across all 4 configurations
- Compare every metric side-by-side against the Panama default
- Print a report template for your submission

**Questions (fill in the template that prints at the end):**

1. **Network Structure**: How does your network differ structurally from Panama? Consider density, degree distribution, node types, and jurisdictions.
2. **MST Comparison**: Which network has higher average MST edge weight? What does this say about connection strength? Did algorithm performance differ?
3. **Cut Properties**: What was the most interesting jurisdictional cut in your network? What does the minimum crossing edge represent?
4. **Clustering**: Were clusters balanced or skewed? What real-world groupings do they correspond to? How does this compare to Panama's clustering?
5. **Surprising Finding**: What did the MST reveal that raw edge counts wouldn't show?

**Some interesting networks to try:**

| Filter | Why it's interesting |
|--------|---------------------|
| `--country "Russia"` | Large offshore presence, sanctions-era structures |
| `--country "China" --jurisdiction "BVI"` | Massive BVI usage for round-tripping capital |
| `--intermediary "Appleby"` | Paradise Papers firm, different client profile than Mossack Fonseca |
| `--source "Pandora Papers"` | Newest leak (2021), different jurisdictions than Panama Papers |
| `--country "United States"` | Domestic offshore usage patterns |
| `--jurisdiction "Seychelles"` | Popular alternative to Panama/BVI |

## Implementing Your Own Extensions

As an additional challenge, implement one or more of:

### Borůvka's Algorithm
The *oldest* MST algorithm (1926!). Each component simultaneously selects its lightest outgoing edge. All such edges are added, components merge, and the process repeats.

```python
def boruvka(nodes, edges):
    # Each node starts as its own component
    # In each round, every component picks its cheapest outgoing edge
    # All chosen edges are added simultaneously
    # Repeat until one component remains
    # O(E log V) — runs in O(log V) rounds
    pass
```

### Verify the MST with Cut/Cycle Optimality Conditions
For every non-tree edge e, verify that e is the heaviest edge on the cycle it would create if added to the MST. For every tree edge e, verify that e is the lightest edge crossing the cut defined by removing e from the MST.

### Second-Best MST
Find the MST whose total weight is second-smallest. This can be done in O(E log V) by trying each non-tree edge swap.

## Submission Requirements

Place all of your submission documents in the [./lab2_submission](lab2_submission) folder. You will submit a zip file of that directory in Canvas.

1. **Experimental data** from Exercises 1-5 (tables filled in, CSV exports, or screenshots)
2. **Written answers** to all questions (~1-2 paragraphs each)
3. **Analysis** of clustering results with at least one visualization
4. **Exercise 6 report** — completed template from `explore_network.py` output, including your network choice, comparison data, and answers to all 5 questions
5. **Code** for any modifications you made
6. **Reflection** (~2 paragraphs): What surprised you about the structure of the Panama Papers graph? How did the MST reveal structure that raw edge lists don't?

## Additional Resources

* [Kruskal's Algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
* [Prim's Algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
* [Cut Property Proof](https://en.wikipedia.org/wiki/Minimum_spanning_tree#Cut_property)
* [Cycle Property Proof](https://en.wikipedia.org/wiki/Minimum_spanning_tree#Cycle_property)
* [ICIJ Offshore Leaks Database](https://offshoreleaks.icij.org/)
* [Union-Find / Disjoint Set Data Structure](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
* [Single-Linkage Clustering](https://en.wikipedia.org/wiki/Single-linkage_clustering)

## Troubleshooting

**"Download failed"**: The ICIJ data URL may change. Visit https://offshoreleaks.icij.org/pages/database and download manually, then unzip into `data/raw/`.

**"Not enough memory for large graph"**: Use a smaller `--max-nodes` value or filter by a single jurisdiction.

**"MST has fewer than n-1 edges"**: Your subgraph may not be connected. The scripts automatically extract the largest connected component, but check the output messages.

**"Different MST weights from Kruskal's and Prim's"**: If edge weights are not unique, there can be multiple valid MSTs with the same total weight. Both algorithms should produce the same *total weight* even if the specific edges differ.
