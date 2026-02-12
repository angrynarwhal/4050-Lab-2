#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <stdbool.h>

/*
 * Union-Find (Disjoint Sets) with optional path compression and union by rank.
 *
 * Supports three modes for Exercise 5:
 *   UF_MODE_NAIVE:      No optimizations (worst case O(n) per find)
 *   UF_MODE_RANK:       Union by rank only (O(log n) per find)
 *   UF_MODE_COMPRESS:   Path compression only (amortized O(log n))
 *   UF_MODE_BOTH:       Both optimizations (amortized O(α(n)))
 */

typedef enum {
    UF_MODE_NAIVE = 0,
    UF_MODE_RANK = 1,
    UF_MODE_COMPRESS = 2,
    UF_MODE_BOTH = 3,
} uf_mode_t;

typedef struct {
    int *parent;
    int *rank;
    int *size;
    int num_elements;
    int num_components;
    uf_mode_t mode;

    /* Instrumentation counters */
    long long find_calls;
    long long total_path_length;
    long long union_calls;
} UnionFind;

/* Create a new Union-Find with n elements (0 to n-1) */
UnionFind *uf_create(int n, uf_mode_t mode);

/* Free a Union-Find */
void uf_destroy(UnionFind *uf);

/* Find the root representative of x */
int uf_find(UnionFind *uf, int x);

/* Merge the sets containing x and y. Returns true if merge happened. */
bool uf_union(UnionFind *uf, int x, int y);

/* Check if x and y are in the same set */
bool uf_connected(UnionFind *uf, int x, int y);

/* Get maximum tree height in the forest */
int uf_max_height(UnionFind *uf);

/* Print instrumentation stats */
void uf_print_stats(UnionFind *uf);

#endif /* UNION_FIND_H */
