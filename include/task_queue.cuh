
#pragma once

#include <cuda_runtime.h>

#define MAX_TASKS 64

struct TaskQueue {
    int tasks[MAX_TASKS];
    int head;
    int tail;
    int lock;

    // Watchdog information
    unsigned long long start_time;
    int watchdog_triggered;
};

__device__ void queue_lock(TaskQueue* q) {
    while (atomicCAS(&q->lock, 0, 1) != 0) {
        // Wait for queue lock
    }
}

__device__ void queue_unlock(TaskQueue* q) {
    __threadfence();
    atomicExch(&q->lock, 0);
}

__device__ bool enqueue(TaskQueue* q, int task_id) {
    queue_lock(q);

    bool success = false;

    if (q->tail < MAX_TASKS) {
        q->tasks[q->tail++] = task_id;
        success = true;
    }

    queue_unlock(q);
    return success;
}

__device__ int dequeue(TaskQueue* q) {
    queue_lock(q);

    int task_id = -1;

    if (q->head < q->tail) {
        task_id = q->tasks[q->head++];
    }

    queue_unlock(q);
    return task_id;
}
