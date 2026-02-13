#include "union_find.h"
#include <stdlib.h>
#include <stdio.h>

UnionFind *uf_create(int n, uf_mode_t mode) {
    UnionFind *uf = (UnionFind *)malloc(sizeof(UnionFind));
    if (!uf) return NULL;

    uf->parent = (int *)malloc(n * sizeof(int));
    uf->rank = (int *)calloc(n, sizeof(int));
    uf->size = (int *)malloc(n * sizeof(int));

    if (!uf->parent || !uf->rank || !uf->size) {
        free(uf->parent);
        free(uf->rank);
        free(uf->size);
        free(uf);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        uf->parent[i] = i;
        uf->size[i] = 1;
    }

    uf->num_elements = n;
    uf->num_components = n;
    uf->mode = mode;

    uf->find_calls = 0;
    uf->total_path_length = 0;
    uf->union_calls = 0;

    return uf;
}

void uf_destroy(UnionFind *uf) {
    if (uf) {
        free(uf->parent);
        free(uf->rank);
        free(uf->size);
        free(uf);
    }
}

int uf_find(UnionFind *uf, int x) {
    uf->find_calls++;
    int path_len = 0;

    if (uf->mode == UF_MODE_COMPRESS || uf->mode == UF_MODE_BOTH) {
        /* Path compression: make every node on the path point to root */
        int root = x;
        while (uf->parent[root] != root) {
            root = uf->parent[root];
            path_len++;
        }
        /* Second pass: compress — point every node on path directly to root */
        while (uf->parent[x] != root) {
            int next = uf->parent[x];
            uf->parent[x] = root;
            x = next;
        }
        x = root;  /* Return the actual root, not the last compressed node */
    } else {
        /* No compression: just follow pointers */
        while (uf->parent[x] != x) {
            x = uf->parent[x];
            path_len++;
        }
    }

    uf->total_path_length += path_len;
    return x;
}

bool uf_union(UnionFind *uf, int x, int y) {
    uf->union_calls++;

    int rx = uf_find(uf, x);
    int ry = uf_find(uf, y);

    if (rx == ry) return false;

    if (uf->mode == UF_MODE_RANK || uf->mode == UF_MODE_BOTH) {
        /* Union by rank */
        if (uf->rank[rx] < uf->rank[ry]) {
            int tmp = rx; rx = ry; ry = tmp;
        }
        uf->parent[ry] = rx;
        uf->size[rx] += uf->size[ry];
        if (uf->rank[rx] == uf->rank[ry]) {
            uf->rank[rx]++;
        }
    } else {
        /* Naive: always attach ry under rx */
        uf->parent[ry] = rx;
        uf->size[rx] += uf->size[ry];
    }

    uf->num_components--;
    return true;
}

bool uf_connected(UnionFind *uf, int x, int y) {
    return uf_find(uf, x) == uf_find(uf, y);
}

int uf_max_height(UnionFind *uf) {
    /* Compute height of each node (distance to root without compression) */
    int max_h = 0;
    for (int i = 0; i < uf->num_elements; i++) {
        int h = 0;
        int x = i;
        while (uf->parent[x] != x) {
            x = uf->parent[x];
            h++;
        }
        if (h > max_h) max_h = h;
    }
    return max_h;
}

void uf_print_stats(UnionFind *uf) {
    double avg_path = (uf->find_calls > 0) ?
        (double)uf->total_path_length / uf->find_calls : 0.0;

    const char *mode_str;
    switch (uf->mode) {
        case UF_MODE_NAIVE:    mode_str = "No optimizations"; break;
        case UF_MODE_RANK:     mode_str = "Union by rank only"; break;
        case UF_MODE_COMPRESS: mode_str = "Path compression only"; break;
        case UF_MODE_BOTH:     mode_str = "Both optimizations"; break;
        default:               mode_str = "Unknown"; break;
    }

    printf("  Union-Find Stats (%s):\n", mode_str);
    printf("    Find calls:       %lld\n", uf->find_calls);
    printf("    Union calls:      %lld\n", uf->union_calls);
    printf("    Avg path length:  %.4f\n", avg_path);
    printf("    Max tree height:  %d\n", uf_max_height(uf));
    printf("    Components:       %d\n", uf->num_components);
}
