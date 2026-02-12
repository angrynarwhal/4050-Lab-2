# C Implementation: Kruskal's Algorithm with Instrumented Union-Find

## Building

```bash
make
```

## Running

### Standard mode (both optimizations):
```bash
./mst ../data/small_nodes.csv ../data/small_edges.csv
```

### Instrumented mode (Exercise 5 — compares all four Union-Find configurations):
```bash
./mst ../data/small_nodes.csv ../data/small_edges.csv --instrument
```

This runs Kruskal's algorithm four times with different Union-Find configurations:
1. **No optimizations**: Naive find (follow chain) + naive union (arbitrary attachment)
2. **Path compression only**: Flatten trees during find, but no rank-based union
3. **Union by rank only**: Attach shorter tree under taller, but no path compression
4. **Both optimizations**: Path compression + union by rank (amortized O(α(n)))

## What to observe

For each configuration, the program reports:
- **Find calls**: Total number of `find()` invocations
- **Union calls**: Total number of `union()` invocations
- **Average path length**: Mean number of parent pointers traversed per find
- **Max tree height**: Deepest tree in the Union-Find forest after all operations
- **Time**: Wall-clock time for the Kruskal's portion (excluding sort)

## Expected behavior

On a graph with ~5,000 nodes:
- Without optimizations: max height can reach O(n), average path ~O(n)
- With union by rank: max height O(log n), average path ~O(log n)
- With path compression: trees flatten over time, average path approaches O(1)
- With both: near-constant time per operation (O(α(n)) amortized)
