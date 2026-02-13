#if defined(__linux__)
#define _POSIX_C_SOURCE 199309L  /* clock_gettime on older Linux */
#endif

/*
 * mst.c — Kruskal's and Prim's MST algorithms in C.
 *
 * Usage:
 *   ./mst nodes.csv edges.csv                  # Kruskal's (default)
 *   ./mst nodes.csv edges.csv --compare        # Exercise 1: Kruskal vs Prim
 *   ./mst nodes.csv edges.csv --instrument     # Exercise 5: Union-Find modes
 *
 * Kruskal's: Sort edges by weight, greedily add via Union-Find.
 *   Time: O(E log E) for sort + O(E alpha(n)) for Union-Find = O(E log E)
 *
 * Prim's: Grow MST from a start vertex using a binary min-heap.
 *   Time: O(E log V) with a binary heap priority queue.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <float.h>

#include "union_find.h"

#define MAX_LABEL_LEN 128
#define MAX_LINE_LEN 4096
#define INITIAL_CAPACITY 1024

/* ========================================================================== */
/* Data structures                                                             */
/* ========================================================================== */

typedef struct {
    int node_id;
    char label[MAX_LABEL_LEN];
    char node_type[32];
    char jurisdiction[64];
} Node;

typedef struct {
    int source;
    int target;
    double weight;
    char rel_type[64];
} Edge;

/* Comparison function for sorting edges by weight (Kruskal's) */
int compare_edges(const void *a, const void *b) {
    double wa = ((Edge *)a)->weight;
    double wb = ((Edge *)b)->weight;
    if (wa < wb) return -1;
    if (wa > wb) return 1;
    return 0;
}

/* ========================================================================== */
/* Adjacency List (for Prim's)                                                 */
/* ========================================================================== */

/*
 * Each node has a linked list of adjacent edges.
 * We store both directions: if (u, v, w) is an edge,
 * u's list gets (v, w) and v's list gets (u, w).
 */

typedef struct AdjEntry {
    int neighbor;
    double weight;
    char rel_type[64];
    struct AdjEntry *next;
} AdjEntry;

typedef struct {
    AdjEntry **heads;   /* heads[i] = linked list for node i */
    int num_nodes;
} AdjList;

AdjList *adj_create(int num_nodes) {
    AdjList *adj = (AdjList *)malloc(sizeof(AdjList));
    adj->num_nodes = num_nodes;
    adj->heads = (AdjEntry **)calloc(num_nodes, sizeof(AdjEntry *));
    return adj;
}

void adj_add_edge(AdjList *adj, int u, int v, double weight, const char *rel_type) {
    /* Add v to u's list */
    AdjEntry *e1 = (AdjEntry *)malloc(sizeof(AdjEntry));
    e1->neighbor = v;
    e1->weight = weight;
    strncpy(e1->rel_type, rel_type, 63);
    e1->rel_type[63] = '\0';
    e1->next = adj->heads[u];
    adj->heads[u] = e1;

    /* Add u to v's list */
    AdjEntry *e2 = (AdjEntry *)malloc(sizeof(AdjEntry));
    e2->neighbor = u;
    e2->weight = weight;
    strncpy(e2->rel_type, rel_type, 63);
    e2->rel_type[63] = '\0';
    e2->next = adj->heads[v];
    adj->heads[v] = e2;
}

void adj_destroy(AdjList *adj) {
    for (int i = 0; i < adj->num_nodes; i++) {
        AdjEntry *cur = adj->heads[i];
        while (cur) {
            AdjEntry *next = cur->next;
            free(cur);
            cur = next;
        }
    }
    free(adj->heads);
    free(adj);
}

/* Build adjacency list from edge array */
AdjList *build_adj_list(int num_nodes, Edge *edges, int num_edges) {
    AdjList *adj = adj_create(num_nodes);
    for (int i = 0; i < num_edges; i++) {
        adj_add_edge(adj, edges[i].source, edges[i].target,
                     edges[i].weight, edges[i].rel_type);
    }
    return adj;
}

/* ========================================================================== */
/* Binary Min-Heap (Indexed Priority Queue for Prim's)                         */
/* ========================================================================== */

/*
 * An indexed min-heap that supports decrease-key in O(log n).
 *
 * heap[i]     = node_id of the i-th element in heap order
 * pos[v]      = position of node v in the heap array (-1 if not in heap)
 * key[v]      = priority (edge weight) for node v
 *
 * This is the standard data structure for efficient Prim's:
 *   - insert:       O(log V)
 *   - extract_min:  O(log V)
 *   - decrease_key: O(log V)
 */

typedef struct {
    int *heap;      /* heap[i] = node_id at position i */
    int *pos;       /* pos[v]  = position of node v in heap (-1 = not present) */
    double *key;    /* key[v]  = current priority of node v */
    int size;       /* current number of elements */
    int capacity;   /* max elements */
} MinHeap;

MinHeap *heap_create(int capacity) {
    MinHeap *h = (MinHeap *)malloc(sizeof(MinHeap));
    h->heap = (int *)malloc(capacity * sizeof(int));
    h->pos = (int *)malloc(capacity * sizeof(int));
    h->key = (double *)malloc(capacity * sizeof(double));
    h->size = 0;
    h->capacity = capacity;
    for (int i = 0; i < capacity; i++) {
        h->pos[i] = -1;
        h->key[i] = DBL_MAX;
    }
    return h;
}

void heap_destroy(MinHeap *h) {
    free(h->heap);
    free(h->pos);
    free(h->key);
    free(h);
}

static void heap_swap(MinHeap *h, int i, int j) {
    int vi = h->heap[i], vj = h->heap[j];
    h->heap[i] = vj;
    h->heap[j] = vi;
    h->pos[vi] = j;
    h->pos[vj] = i;
}

static void heap_bubble_up(MinHeap *h, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->key[h->heap[i]] < h->key[h->heap[parent]]) {
            heap_swap(h, i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

static void heap_bubble_down(MinHeap *h, int i) {
    while (1) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < h->size && h->key[h->heap[left]] < h->key[h->heap[smallest]])
            smallest = left;
        if (right < h->size && h->key[h->heap[right]] < h->key[h->heap[smallest]])
            smallest = right;

        if (smallest != i) {
            heap_swap(h, i, smallest);
            i = smallest;
        } else {
            break;
        }
    }
}

void heap_insert(MinHeap *h, int node, double key) {
    h->key[node] = key;
    h->heap[h->size] = node;
    h->pos[node] = h->size;
    h->size++;
    heap_bubble_up(h, h->size - 1);
}

/* Extract the node with minimum key */
int heap_extract_min(MinHeap *h) {
    if (h->size == 0) return -1;

    int min_node = h->heap[0];
    h->pos[min_node] = -1;

    h->size--;
    if (h->size > 0) {
        h->heap[0] = h->heap[h->size];
        h->pos[h->heap[0]] = 0;
        heap_bubble_down(h, 0);
    }

    return min_node;
}

/* Decrease the key of a node already in the heap */
void heap_decrease_key(MinHeap *h, int node, double new_key) {
    if (h->pos[node] == -1) {
        /* Not in heap yet — insert it */
        heap_insert(h, node, new_key);
        return;
    }
    if (new_key < h->key[node]) {
        h->key[node] = new_key;
        heap_bubble_up(h, h->pos[node]);
    }
}

bool heap_empty(MinHeap *h) {
    return h->size == 0;
}

/* ========================================================================== */
/* CSV Loading                                                                 */
/* ========================================================================== */

int load_nodes(const char *filename, Node **out_nodes) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return -1;
    }

    int capacity = INITIAL_CAPACITY;
    int count = 0;
    Node *nodes = (Node *)malloc(capacity * sizeof(Node));
    char line[MAX_LINE_LEN];

    /* Skip header */
    if (!fgets(line, MAX_LINE_LEN, f)) { fclose(f); *out_nodes = NULL; return 0; }

    while (fgets(line, MAX_LINE_LEN, f)) {
        if (count >= capacity) {
            capacity *= 2;
            nodes = (Node *)realloc(nodes, capacity * sizeof(Node));
        }

        Node *n = &nodes[count];
        /* Parse: node_id,label,node_type,jurisdiction,country_codes */
        char *tok = strtok(line, ",");
        if (!tok) continue;
        n->node_id = atoi(tok);

        tok = strtok(NULL, ",");
        if (tok) strncpy(n->label, tok, MAX_LABEL_LEN - 1);
        else n->label[0] = '\0';

        tok = strtok(NULL, ",");
        if (tok) strncpy(n->node_type, tok, 31);
        else n->node_type[0] = '\0';

        tok = strtok(NULL, ",");
        if (tok) strncpy(n->jurisdiction, tok, 63);
        else n->jurisdiction[0] = '\0';

        count++;
    }

    fclose(f);
    *out_nodes = nodes;
    return count;
}

int load_edges(const char *filename, Edge **out_edges) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return -1;
    }

    int capacity = INITIAL_CAPACITY;
    int count = 0;
    Edge *edges = (Edge *)malloc(capacity * sizeof(Edge));
    char line[MAX_LINE_LEN];

    /* Skip header */
    if (!fgets(line, MAX_LINE_LEN, f)) { fclose(f); *out_edges = NULL; return 0; }

    while (fgets(line, MAX_LINE_LEN, f)) {
        if (count >= capacity) {
            capacity *= 2;
            edges = (Edge *)realloc(edges, capacity * sizeof(Edge));
        }

        Edge *e = &edges[count];
        /* Parse: source,target,weight,rel_type */
        char *tok = strtok(line, ",");
        if (!tok) continue;
        e->source = atoi(tok);

        tok = strtok(NULL, ",");
        if (!tok) continue;
        e->target = atoi(tok);

        tok = strtok(NULL, ",");
        if (!tok) continue;
        e->weight = atof(tok);

        tok = strtok(NULL, ",\n\r");
        if (tok) strncpy(e->rel_type, tok, 63);
        else e->rel_type[0] = '\0';

        count++;
    }

    fclose(f);
    *out_edges = edges;
    return count;
}

/* ========================================================================== */
/* Kruskal's Algorithm                                                         */
/* ========================================================================== */

/*
 * Strategy: Sort all edges by weight, then greedily add edges that don't
 * create a cycle (checked via Union-Find).
 *
 * Time complexity:
 *   O(E log E) for sorting  +  O(E * alpha(n)) for Union-Find operations
 *   = O(E log E) overall    (since alpha(n) is effectively constant)
 *
 * The sort dominates. This is efficient when E is not much larger than V.
 */

typedef struct {
    Edge *mst_edges;
    int mst_count;
    double total_weight;
    double sort_seconds;    /* Time spent sorting edges */
    double uf_seconds;      /* Time spent on Union-Find (Kruskal's core) */
    double total_seconds;   /* Wall-clock total */
} MSTResult;

MSTResult kruskal(int num_nodes, Edge *edges, int num_edges, uf_mode_t mode, bool quiet) {
    MSTResult result;
    result.mst_edges = (Edge *)malloc((num_nodes - 1) * sizeof(Edge));
    result.mst_count = 0;
    result.total_weight = 0.0;

    struct timespec t0, t1, t2;

    /* Phase 1: Sort edges by weight */
    Edge *sorted = (Edge *)malloc(num_edges * sizeof(Edge));
    memcpy(sorted, edges, num_edges * sizeof(Edge));

    clock_gettime(CLOCK_MONOTONIC, &t0);
    qsort(sorted, num_edges, sizeof(Edge), compare_edges);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    result.sort_seconds = (t1.tv_sec - t0.tv_sec) +
                           (t1.tv_nsec - t0.tv_nsec) / 1e9;

    /* Phase 2: Greedy edge selection with Union-Find */
    UnionFind *uf = uf_create(num_nodes, mode);

    for (int i = 0; i < num_edges && result.mst_count < num_nodes - 1; i++) {
        Edge *e = &sorted[i];
        if (uf_union(uf, e->source, e->target)) {
            result.mst_edges[result.mst_count++] = *e;
            result.total_weight += e->weight;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    result.uf_seconds = (t2.tv_sec - t1.tv_sec) +
                         (t2.tv_nsec - t1.tv_nsec) / 1e9;
    result.total_seconds = (t2.tv_sec - t0.tv_sec) +
                            (t2.tv_nsec - t0.tv_nsec) / 1e9;

    if (!quiet) uf_print_stats(uf);
    uf_destroy(uf);
    free(sorted);

    return result;
}

/* ========================================================================== */
/* Prim's Algorithm                                                            */
/* ========================================================================== */

/*
 * Strategy: Grow the MST from a start vertex. Maintain a priority queue of
 * frontier edges. At each step, extract the minimum-weight edge connecting
 * a non-tree vertex to the tree, add it, and update neighbors.
 *
 * Time complexity:
 *   Each vertex is extracted once:        O(V log V)
 *   Each edge triggers at most one
 *   decrease-key:                         O(E log V)
 *   Total:                                O(E log V)
 *
 * With a Fibonacci heap this would be O(E + V log V), but binary heaps
 * are simpler and have better constants for moderate graph sizes.
 */

MSTResult prim(int num_nodes, AdjList *adj, int start) {
    MSTResult result;
    result.mst_edges = (Edge *)malloc((num_nodes - 1) * sizeof(Edge));
    result.mst_count = 0;
    result.total_weight = 0.0;
    result.sort_seconds = 0.0;  /* N/A for Prim's */

    /* Track which node connected each vertex to the tree */
    int *parent = (int *)malloc(num_nodes * sizeof(int));
    double *best_weight = (double *)malloc(num_nodes * sizeof(double));
    char (*best_rel)[64] = (char (*)[64])malloc(num_nodes * 64);
    bool *in_tree = (bool *)calloc(num_nodes, sizeof(bool));

    for (int i = 0; i < num_nodes; i++) {
        parent[i] = -1;
        best_weight[i] = DBL_MAX;
        best_rel[i][0] = '\0';
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Initialize: start vertex has key 0 */
    MinHeap *pq = heap_create(num_nodes);
    best_weight[start] = 0.0;
    heap_insert(pq, start, 0.0);

    /* Insert all other vertices with key = infinity */
    for (int v = 0; v < num_nodes; v++) {
        if (v != start) {
            heap_insert(pq, v, DBL_MAX);
        }
    }

    while (!heap_empty(pq)) {
        int u = heap_extract_min(pq);
        in_tree[u] = true;

        /* Add the edge that connected u to the tree (skip start vertex) */
        if (parent[u] != -1) {
            Edge e;
            e.source = parent[u];
            e.target = u;
            e.weight = best_weight[u];
            strncpy(e.rel_type, best_rel[u], 63);
            e.rel_type[63] = '\0';
            result.mst_edges[result.mst_count++] = e;
            result.total_weight += best_weight[u];
        }

        /* Update neighbors */
        AdjEntry *cur = adj->heads[u];
        while (cur) {
            int v = cur->neighbor;
            if (!in_tree[v] && cur->weight < best_weight[v]) {
                best_weight[v] = cur->weight;
                parent[v] = u;
                snprintf(best_rel[v], 64, "%s", cur->rel_type);
                heap_decrease_key(pq, v, cur->weight);
            }
            cur = cur->next;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    result.total_seconds = (t1.tv_sec - t0.tv_sec) +
                            (t1.tv_nsec - t0.tv_nsec) / 1e9;
    result.uf_seconds = result.total_seconds;  /* All time is "core" for Prim's */

    heap_destroy(pq);
    free(parent);
    free(best_weight);
    free(best_rel);
    free(in_tree);

    return result;
}

/* ========================================================================== */
/* Result Printing                                                             */
/* ========================================================================== */

void print_mst_result(MSTResult *r, int num_nodes) {
    printf("    MST edges:    %d", r->mst_count);
    if (r->mst_count < num_nodes - 1) {
        printf("  (expected %d -- graph may not be connected)", num_nodes - 1);
    }
    printf("\n");
    printf("    Total weight: %.4f\n", r->total_weight);
    printf("    Total time:   %.6f s\n", r->total_seconds);

    if (r->sort_seconds > 0) {
        /* Kruskal's: show sort vs UF breakdown */
        double sort_pct = (r->total_seconds > 0)
            ? 100.0 * r->sort_seconds / r->total_seconds : 0;
        double uf_pct = (r->total_seconds > 0)
            ? 100.0 * r->uf_seconds / r->total_seconds : 0;
        printf("      Sort:       %.6f s (%.1f%%)\n", r->sort_seconds, sort_pct);
        printf("      Union-Find: %.6f s (%.1f%%)\n", r->uf_seconds, uf_pct);
    }
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

void print_usage(const char *prog) {
    printf("Usage: %s <nodes.csv> <edges.csv> [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --compare          Exercise 1: Compare Kruskal's vs Prim's\n");
    printf("  --instrument       Exercise 5: Compare Union-Find configurations\n");
    printf("  --prim             Run only Prim's algorithm\n");
    printf("  --kruskal          Run only Kruskal's algorithm (default)\n");
    printf("  --iterations N     Run each algorithm N times, report average (default: 1)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s ../data/large_nodes.csv ../data/large_edges.csv --compare --iterations 10\n", prog);
    printf("  %s ../data/xlarge_nodes.csv ../data/xlarge_edges.csv --compare --iterations 5\n", prog);
    printf("  %s ../data/large_nodes.csv ../data/large_edges.csv --instrument --iterations 5\n", prog);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *nodes_file = argv[1];
    const char *edges_file = argv[2];
    bool do_compare = false;
    bool do_instrument = false;
    bool do_prim_only = false;
    int iterations = 1;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--compare") == 0) do_compare = true;
        else if (strcmp(argv[i], "--instrument") == 0) do_instrument = true;
        else if (strcmp(argv[i], "--prim") == 0) do_prim_only = true;
        else if (strcmp(argv[i], "--kruskal") == 0) { /* default */ }
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
            if (iterations < 1) iterations = 1;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Load data */
    printf("Loading graph...\n");
    Node *nodes;
    int num_nodes = load_nodes(nodes_file, &nodes);
    if (num_nodes < 0) return 1;

    Edge *edges;
    int num_edges = load_edges(edges_file, &edges);
    if (num_edges < 0) { free(nodes); return 1; }

    printf("  Nodes: %d, Edges: %d\n", num_nodes, num_edges);
    if (iterations > 1)
        printf("  Iterations: %d (reporting averages)\n", iterations);
    printf("\n");

    /* ---------------------------------------------------------------------- */
    /* Exercise 1: Compare Kruskal's vs Prim's                                */
    /* ---------------------------------------------------------------------- */
    if (do_compare) {
        printf("============================================================\n");
        printf("Exercise 1: Kruskal's vs Prim's Comparison\n");
        printf("============================================================\n");

        /* Build adjacency list once (shared across Prim's iterations) */
        AdjList *adj = build_adj_list(num_nodes, edges, num_edges);

        /* Run Kruskal's */
        printf("\n--- Kruskal's Algorithm ---\n");
        printf("    Strategy: Sort all edges, greedily add via Union-Find\n");
        printf("    Complexity: O(E log E)\n\n");

        double kr_total_time = 0, kr_sort_time = 0, kr_uf_time = 0;
        double kr_weight = 0;
        int kr_mst_count = 0;

        for (int it = 0; it < iterations; it++) {
            MSTResult kr = kruskal(num_nodes, edges, num_edges, UF_MODE_BOTH, it > 0);
            kr_total_time += kr.total_seconds;
            kr_sort_time += kr.sort_seconds;
            kr_uf_time += kr.uf_seconds;
            if (it == 0) {
                kr_weight = kr.total_weight;
                kr_mst_count = kr.mst_count;
            }
            free(kr.mst_edges);
        }
        kr_total_time /= iterations;
        kr_sort_time /= iterations;
        kr_uf_time /= iterations;

        printf("    MST edges:    %d\n", kr_mst_count);
        printf("    Total weight: %.4f\n", kr_weight);
        if (iterations > 1) printf("    Avg time:     %.6f s (%d iterations)\n", kr_total_time, iterations);
        else                printf("    Total time:   %.6f s\n", kr_total_time);
        if (kr_total_time > 0) {
            printf("      Sort:       %.6f s (%.1f%%)\n",
                   kr_sort_time, 100.0 * kr_sort_time / kr_total_time);
            printf("      Union-Find: %.6f s (%.1f%%)\n",
                   kr_uf_time, 100.0 * kr_uf_time / kr_total_time);
        }

        /* Run Prim's */
        printf("\n--- Prim's Algorithm ---\n");
        printf("    Strategy: Grow tree from vertex 0, binary min-heap PQ\n");
        printf("    Complexity: O(E log V)\n\n");

        double pr_total_time = 0;
        double pr_weight = 0;
        int pr_mst_count = 0;

        for (int it = 0; it < iterations; it++) {
            MSTResult pr = prim(num_nodes, adj, 0);
            pr_total_time += pr.total_seconds;
            if (it == 0) {
                pr_weight = pr.total_weight;
                pr_mst_count = pr.mst_count;
            }
            free(pr.mst_edges);
        }
        pr_total_time /= iterations;

        printf("    MST edges:    %d\n", pr_mst_count);
        printf("    Total weight: %.4f\n", pr_weight);
        if (iterations > 1) printf("    Avg time:     %.6f s (%d iterations)\n", pr_total_time, iterations);
        else                printf("    Total time:   %.6f s\n", pr_total_time);

        adj_destroy(adj);

        /* Comparison summary */
        printf("\n============================================================\n");
        printf("Comparison Summary%s\n",
               iterations > 1 ? " (averaged)" : "");
        printf("============================================================\n");
        printf("  %-22s %14s %14s\n", "", "Kruskal", "Prim");
        printf("  %-22s %14s %14s\n", "----------------------",
               "--------------", "--------------");
        printf("  %-22s %14.4f %14.4f\n", "MST weight",
               kr_weight, pr_weight);
        printf("  %-22s %14d %14d\n", "MST edges",
               kr_mst_count, pr_mst_count);
        printf("  %-22s %12.6f s %12.6f s\n",
               iterations > 1 ? "Avg time" : "Total time",
               kr_total_time, pr_total_time);
        printf("  %-22s %12.6f s %14s\n",
               "  Sort phase", kr_sort_time, "N/A");
        printf("  %-22s %12.6f s %14s\n",
               "  Union-Find phase", kr_uf_time, "N/A");

        double weight_diff = kr_weight - pr_weight;
        if (weight_diff < 0) weight_diff = -weight_diff;
        printf("\n  Weight difference: %.6f %s\n",
               weight_diff,
               (weight_diff < 0.001) ? "(match)" : "(MISMATCH -- check for ties)");

        if (kr_total_time > 0 && pr_total_time > 0) {
            double ratio = kr_total_time / pr_total_time;
            printf("  Speed ratio: Kruskal/Prim = %.2fx %s\n",
                   ratio,
                   ratio > 1.0 ? "(Prim faster)" : "(Kruskal faster)");
        }

        printf("\n  Kruskal's is dominated by its O(E log E) sort.\n");
        printf("  Prim's heap operations cost O(E log V) per iteration.\n");
        printf("  On sparse graphs Kruskal's sort is very cache-friendly,\n");
        printf("  while Prim's pointer-chasing through adjacency lists is not.\n");
    }

    /* ---------------------------------------------------------------------- */
    /* Exercise 5: Union-Find Instrumentation                                  */
    /* ---------------------------------------------------------------------- */
    else if (do_instrument) {
        printf("============================================================\n");
        printf("Exercise 5: Union-Find Performance Comparison\n");
        printf("============================================================\n");

        uf_mode_t modes[] = {UF_MODE_NAIVE, UF_MODE_COMPRESS,
                              UF_MODE_RANK, UF_MODE_BOTH};
        const char *names[] = {
            "No optimizations",
            "Path compression only",
            "Union by rank only",
            "Both optimizations"
        };

        double avg_times[4] = {0};
        double weights[4] = {0};

        for (int m = 0; m < 4; m++) {
            printf("\n--- %s ---\n", names[m]);

            for (int it = 0; it < iterations; it++) {
                MSTResult r = kruskal(num_nodes, edges, num_edges, modes[m], it > 0);
                avg_times[m] += r.total_seconds;
                if (it == 0) {
                    weights[m] = r.total_weight;
                    print_mst_result(&r, num_nodes);
                }
                free(r.mst_edges);
            }
            avg_times[m] /= iterations;

            if (iterations > 1) {
                printf("    Avg time:     %.6f s (%d iterations)\n",
                       avg_times[m], iterations);
            }
        }

        /* Summary table */
        printf("\n============================================================\n");
        printf("Summary%s\n", iterations > 1 ? " (averaged)" : "");
        printf("============================================================\n");
        printf("  %-25s %10s %10s\n",
               "Configuration",
               iterations > 1 ? "Avg (s)" : "Time (s)",
               "MST Weight");
        printf("  %-25s %10s %10s\n",
               "-------------------------", "----------", "----------");
        for (int m = 0; m < 4; m++) {
            printf("  %-25s %10.6f %10.4f\n",
                   names[m], avg_times[m], weights[m]);
        }

        if (avg_times[0] > 0) {
            printf("\n  Speedups vs naive:\n");
            for (int m = 1; m < 4; m++) {
                double speedup = avg_times[0] / avg_times[m];
                printf("    %-25s %.1fx\n", names[m], speedup);
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* Single algorithm run                                                    */
    /* ---------------------------------------------------------------------- */
    else if (do_prim_only) {
        printf("--- Prim's Algorithm ---\n\n");
        AdjList *adj = build_adj_list(num_nodes, edges, num_edges);
        double total = 0;
        for (int it = 0; it < iterations; it++) {
            MSTResult pr = prim(num_nodes, adj, 0);
            total += pr.total_seconds;
            if (it == 0) print_mst_result(&pr, num_nodes);
            free(pr.mst_edges);
        }
        if (iterations > 1)
            printf("    Avg time: %.6f s (%d iterations)\n", total / iterations, iterations);
        adj_destroy(adj);
    }
    else {
        /* Default: Kruskal's with both optimizations */
        printf("--- Kruskal's Algorithm ---\n\n");
        double total = 0;
        for (int it = 0; it < iterations; it++) {
            MSTResult kr = kruskal(num_nodes, edges, num_edges, UF_MODE_BOTH, it > 0);
            total += kr.total_seconds;
            if (it == 0) print_mst_result(&kr, num_nodes);
            free(kr.mst_edges);
        }
        if (iterations > 1)
            printf("    Avg time: %.6f s (%d iterations)\n", total / iterations, iterations);
    }

    free(nodes);
    free(edges);
    return 0;
}
