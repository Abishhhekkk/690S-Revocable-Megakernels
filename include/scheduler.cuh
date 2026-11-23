
#pragma once

#include "task.cuh"
#include "task_queue.cuh"
#include "task_record.cuh"

#define WATCHDOG_TIMEOUT_NS 5000000000ULL

// Read GPU global timer (nanoseconds)
__device__ unsigned long long get_global_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

// Initialize scheduler
__global__ void initialize_scheduler(
    Task* tasks,
    int num_tasks,
    TaskQueue* queue,
    int* completed_count
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        queue->head = 0;
        queue->tail = 0;
        queue->lock = 0;

        queue->start_time = get_global_time();
        queue->watchdog_triggered = 0;

        *completed_count = 0;

        for (int i = 0; i < num_tasks; i++) {
            tasks[i].state = TASK_WAITING;
            tasks[i].start_time = 0;
            tasks[i].end_time = 0;
            tasks[i].execution_count = 0;

            if (tasks[i].remaining_dependencies == 0) {
                tasks[i].state = TASK_READY;
                enqueue(queue, i);
            }
        }
    }
}

// Persistent GPU worker kernel
__global__ void persistent_scheduler(
    Task* tasks,
    int num_tasks,
    TaskQueue* queue,
    int* completed_count
) {
    if (threadIdx.x != 0) {
        return;
    }

    while (atomicAdd(completed_count, 0) < num_tasks) {

        // Stop all workers if watchdog detects a hang
        if (atomicAdd(&queue->watchdog_triggered, 0)) {
            return;
        }

        unsigned long long elapsed =
            get_global_time() - queue->start_time;

        if (elapsed > WATCHDOG_TIMEOUT_NS) {
            atomicExch(&queue->watchdog_triggered, 1);
            return;
        }

        int task_id = dequeue(queue);

        if (task_id < 0) {
            continue;
        }

        if (task_id >= num_tasks) {
            continue;
        }

        Task* task = &tasks[task_id];

        if (atomicCAS(&task->state, TASK_READY, TASK_RUNNING)
            != TASK_READY) {
            continue;
        }

        atomicAdd(&task->execution_count, 1);

        unsigned long long t_start = get_global_time();
        task->start_time = t_start;
        uint32_t log_slot = log_task_begin((uint32_t)task_id, t_start);  // no-op unless ENABLE_TASK_LOG

        // Simulate task execution
        unsigned long long start = clock64();

        while (clock64() - start < task->duration) {
            // Simulated work
        }

        unsigned long long t_end = get_global_time();
        task->end_time = t_end;
        log_task_end(log_slot, t_end);  // before dependents are released

        __threadfence();

        atomicExch(&task->state, TASK_DONE);

        // Release dependent tasks
        for (int i = 0; i < task->dependent_count; i++) {
            int dependent_id = task->dependents[i];

            int old_value = atomicSub(
                &tasks[dependent_id].remaining_dependencies, 1
            );

            if (old_value == 1) {
                atomicExch(
                    &tasks[dependent_id].state,
                    TASK_READY
                );

                enqueue(queue, dependent_id);
            }
        }

        atomicAdd(completed_count, 1);
    }
}
