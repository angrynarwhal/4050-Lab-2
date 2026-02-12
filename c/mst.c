#define _POSIX_C_SOURCE 199309L

/*
 * mst.c — Kruskal's MST algorithm in C with Union-Find instrumentation.
 *
 * Usage:
 *   ./mst nodes.csv edges.csv [--instrument]
 *
 * With --instrument, runs Kruskal's with all four Union-Find configurations
 * and reports comparative performance data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

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

/* Comparison function for sorting edges by weight */
int compare_edges(const void *a, const void *b) {
    double wa = ((Edge *)a)->weight;
    double wb = ((Edge *)b)->weight;
    if (wa < wb) return -1;
    if (wa > wb) return 1;
    return 0;
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

typedef struct {
    Edge *mst_edges;
    int mst_count;
    double total_weight;
    double elapsed_seconds;
} KruskalResult;

KruskalResult kruskal(int num_nodes, Edge *edges, int num_edges, uf_mode_t mode) {
    KruskalResult result;
    result.mst_edges = (Edge *)malloc((num_nodes - 1) * sizeof(Edge));
    result.mst_count = 0;
    result.total_weight = 0.0;

    /* Sort edges by weight */
    Edge *sorted = (Edge *)malloc(num_edges * sizeof(Edge));
    memcpy(sorted, edges, num_edges * sizeof(Edge));
    qsort(sorted, num_edges, sizeof(Edge), compare_edges);

    /* Initialize Union-Find */
    UnionFind *uf = uf_create(num_nodes, mode);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_edges && result.mst_count < num_nodes - 1; i++) {
        Edge *e = &sorted[i];
        if (uf_union(uf, e->source, e->target)) {
            result.mst_edges[result.mst_count++] = *e;
            result.total_weight += e->weight;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    result.elapsed_seconds = (end.tv_sec - start.tv_sec) +
                              (end.tv_nsec - start.tv_nsec) / 1e9;

    uf_print_stats(uf);
    printf("    MST edges: %d\n", result.mst_count);
    printf("    Total weight: %.4f\n", result.total_weight);
    printf("    Time: %.6f seconds\n", result.elapsed_seconds);

    uf_destroy(uf);
    free(sorted);

    return result;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

void print_usage(const char *prog) {
    printf("Usage: %s <nodes.csv> <edges.csv> [--instrument]\n\n", prog);
    printf("  --instrument  Run all four Union-Find configurations and compare\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *nodes_file = argv[1];
    const char *edges_file = argv[2];
    bool instrument = false;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--instrument") == 0) {
            instrument = true;
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

    printf("  Nodes: %d, Edges: %d\n\n", num_nodes, num_edges);

    if (instrument) {
        /* Run all four Union-Find modes */
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

        for (int m = 0; m < 4; m++) {
            printf("\n--- %s ---\n", names[m]);
            KruskalResult r = kruskal(num_nodes, edges, num_edges, modes[m]);
            free(r.mst_edges);
        }
    } else {
        /* Default: run with both optimizations */
        printf("--- Kruskal's Algorithm (both optimizations) ---\n");
        KruskalResult r = kruskal(num_nodes, edges, num_edges, UF_MODE_BOTH);
        free(r.mst_edges);
    }

    free(nodes);
    free(edges);
    return 0;
}
