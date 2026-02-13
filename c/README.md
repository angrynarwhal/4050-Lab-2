# C Implementation: Kruskal's and Prim's MST Algorithms

## Building

```bash
make
```

## Running

### Exercise 1: Kruskal's vs Prim's comparison

C is fast — use large datasets and `--iterations` for stable timing:

```bash
# Generate larger datasets first (from the scripts/ directory):
cd ../scripts
python3 build_graph.py --jurisdiction "Panama" --max-nodes 50000 --output large
python3 build_graph.py --max-nodes 200000 --output xlarge
python3 build_graph.py --max-nodes 500000 --output huge

# Then run comparisons (--iterations averages over N runs):
cd ../c
./mst ../data/large_nodes.csv ../data/large_edges.csv --compare --iterations 10
./mst ../data/xlarge_nodes.csv ../data/xlarge_edges.csv --compare --iterations 5
./mst ../data/huge_nodes.csv ../data/huge_edges.csv --compare --iterations 3
```

This runs both algorithms on the same graph and prints:
- MST weight and edge count (should match)
- Total wall-clock time for each
- Kruskal's sort vs Union-Find phase breakdown
- Speed ratio and analysis notes

### Exercise 5: Union-Find mode comparison

```bash
./mst ../data/large_nodes.csv ../data/large_edges.csv --instrument
```

Runs Kruskal's four times with different Union-Find configurations:
1. **No optimizations**: Naive find + naive union
2. **Path compression only**: Flatten trees during find
3. **Union by rank only**: Attach shorter tree under taller
4. **Both optimizations**: Path compression + union by rank (amortized O(α(n)))

### Single algorithm runs

```bash
./mst ../data/small_nodes.csv ../data/small_edges.csv                # Kruskal's (default)
./mst ../data/small_nodes.csv ../data/small_edges.csv --kruskal      # Kruskal's (explicit)
./mst ../data/small_nodes.csv ../data/small_edges.csv --prim         # Prim's only
```

## Algorithms

### Kruskal's — O(E log E)

1. **Sort** all edges by weight — O(E log E)
2. **Iterate** through sorted edges; for each edge (u, v):
   - If u and v are in different components (Union-Find `find`), add edge to MST and `union` them
   - Otherwise skip (would create a cycle)
3. Stop when MST has V−1 edges

The sort dominates. Union-Find operations with both optimizations run in amortized O(α(n)) ≈ O(1).

### Prim's — O(E log V)

1. **Initialize** a binary min-heap with all vertices; start vertex gets key 0, all others get ∞
2. **Extract-min** from the heap — this is the next vertex to join the MST
3. **Update** all neighbors of the extracted vertex: if the edge weight is less than the neighbor's current key, decrease its key in the heap
4. Repeat until heap is empty

Each vertex is extracted once — O(V log V). Each edge triggers at most one decrease-key — O(E log V). Total: O(E log V).

### When is each faster?

| Scenario | Kruskal's | Prim's |
|----------|-----------|--------|
| Sparse graph (E ≈ V) | Similar | Similar |
| Dense graph (E ≈ V²) | Sort is O(V² log V) | Heap is O(V² log V) |
| Very dense (E >> V) | Sort cost grows | Heap cost grows slower |
| Already-sorted edges | Very fast | No benefit |

For the Panama Papers graphs (sparse, E ≈ 1–2V), both perform similarly at small sizes. Use `large`+ datasets with `--iterations` for stable, meaningful timing:

| Dataset | Nodes | Expected Kruskal | Expected Prim | Notes |
|---------|-------|-------------------|---------------|-------|
| small | 5,000 | <1 ms | <2 ms | Too fast for C |
| large | 50,000 | ~2 ms | ~17 ms | Timing visible |
| xlarge | 200,000 | ~9 ms | ~70 ms | Clear differences |
| huge | 500,000 | ~30 ms | ~250 ms | Dramatic scaling |

Use `--iterations 10` with smaller datasets for stable averages, `--iterations 3` for huge.

When no filter is specified and `--max-nodes` >= 100,000, `build_graph.py` uses the entire ICIJ database (~800K nodes across all leaks) instead of just the Panama Papers subset.

## Data structures in this implementation

- **Union-Find** (`union_find.c`): Array-based disjoint sets with configurable path compression and union by rank. Instrumented with find_calls, path_length, and max_height counters.

- **Adjacency List** (`mst.c`): Linked-list representation for Prim's. Each undirected edge stored twice (one per endpoint). Built from the edge array in O(E).

- **Binary Min-Heap** (`mst.c`): Indexed priority queue supporting O(log V) insert, extract-min, and decrease-key. The `pos[]` array enables O(1) lookup of a node's position for decrease-key. This is the textbook data structure for Prim's.
