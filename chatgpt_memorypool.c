/* ============================
 * memory_pool.h
 * ============================ */
#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

#define MAX_POOL_BUFFERS 64

typedef struct {
    bool in_use[MAX_POOL_BUFFERS];
    size_t capacity;
    pthread_mutex_t lock;
} buffer_tracker_t;

void init_buffer_tracker(buffer_tracker_t* tracker, size_t capacity);
int get_free_index(buffer_tracker_t* tracker);
void release_index(buffer_tracker_t* tracker, int index);

#endif


/* ============================
 * memory_pool.c
 * ============================ */
#include "memory_pool.h"
#include <string.h>

void init_buffer_tracker(buffer_tracker_t* tracker, size_t capacity) {
    tracker->capacity = capacity;
    memset(tracker->in_use, 0, sizeof(tracker->in_use));
    pthread_mutex_init(&tracker->lock, NULL);
}

int get_free_index(buffer_tracker_t* tracker) {
    pthread_mutex_lock(&tracker->lock);
    for (size_t i = 0; i < tracker->capacity; ++i) {
        if (!tracker->in_use[i]) {
            tracker->in_use[i] = true;
            pthread_mutex_unlock(&tracker->lock);
            return (int)i;
        }
    }
    pthread_mutex_unlock(&tracker->lock);
    return -1; // No free buffer
}

void release_index(buffer_tracker_t* tracker, int index) {
    if (index >= 0 && (size_t)index < tracker->capacity) {
        pthread_mutex_lock(&tracker->lock);
        tracker->in_use[index] = false;
        pthread_mutex_unlock(&tracker->lock);
    }
}


/* ============================
 * Update to MLModel in mldev.h
 * ============================ */
#include "memory_pool.h"
...
typedef struct {
    ...
    buffer_tracker_t input_tracker;
    buffer_tracker_t output_tracker;
} MLModel;


/* ============================
 * Update to ml_dev_load()
 * ============================ */
#include "memory_pool.h"
...
    init_buffer_tracker(&model->input_tracker, model->input_pool_size);
    init_buffer_tracker(&model->output_tracker, model->output_pool_size);


/* ============================
 * Replace hardcoded index in multimodal.c
 * ============================ */
#include "memory_pool.h"
...
    int in_idx = get_free_index(&models[dev_idx].input_tracker);
    int out_idx = get_free_index(&models[dev_idx].output_tracker);
    if (in_idx < 0 || out_idx < 0) {
        fprintf(stderr, "No free buffer index\n");
        continue; // or wait
    }
    temp_args.input_buffer_index = in_idx;
    temp_args.output_buffer_index = out_idx;

...
// After inference completes (at end of thread_infer or inference_thread_func)
    release_index(&args->model->input_tracker, args->input_buffer_index);
    release_index(&args->model->output_tracker, args->output_buffer_index);


/* ============================
 * Optional: Use per-thread affinity or static assignment if needed
 * ============================ */
// Could be implemented via mapping thread ID to fixed buffer index
// if number of threads is fixed and matches pool size
