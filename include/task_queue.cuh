
#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>

#define MAX_TASKS 64

struct TaskQueue {
    int tasks[MAX_TASKS];

    // Monotonically increasing counters.
    // Physical array positions use modulo MAX_TASKS.
    unsigned long long head;
    unsigned long long tail;

    // Spinlock.
    int lock;

    // Queue error reporting.
    int overflow;

    // Watchdog information.
    unsigned long long last_progress_time;
    int watchdog_triggered;
};

// =============================================================
// QUEUE LOCK: ACQUIRE
// =============================================================

__device__ inline void queue_lock(TaskQueue* q) {

    cuda::atomic_ref<int, cuda::thread_scope_device>
        lock_ref(q->lock);

    int expected = 0;

    while (!lock_ref.compare_exchange_weak(
        expected,
        1,
        cuda::memory_order_acquire,
        cuda::memory_order_relaxed
    )) {
        expected = 0;
    }
}

// =============================================================
// QUEUE LOCK: RELEASE
// =============================================================

__device__ inline void queue_unlock(TaskQueue* q) {

    cuda::atomic_ref<int, cuda::thread_scope_device>
        lock_ref(q->lock);

    lock_ref.store(0, cuda::memory_order_release);
}

// =============================================================
// ENQUEUE: RING BUFFER
// =============================================================

__device__ inline bool enqueue(TaskQueue* q, int task_id) {

    queue_lock(q);

    bool success = false;

    // Number of currently occupied queue entries.
    unsigned long long count = q->tail - q->head;

    if (count < MAX_TASKS) {

        // Reuse consumed slots through circular indexing.
        unsigned long long index = q->tail % MAX_TASKS;

        q->tasks[index] = task_id;

        // Publish the new entry.
        q->tail++;

        success = true;

    } else {

        // Never silently discard a task.
        q->overflow = 1;
    }

    queue_unlock(q);

    return success;
}

// =============================================================
// DEQUEUE: RING BUFFER
// =============================================================

__device__ inline int dequeue(TaskQueue* q) {

    queue_lock(q);

    int task_id = -1;

    if (q->head < q->tail) {

        // Circular indexing.
        unsigned long long index = q->head % MAX_TASKS;

        task_id = q->tasks[index];

        q->head++;
    }

    queue_unlock(q);

    return task_id;
}
