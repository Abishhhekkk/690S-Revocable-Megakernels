
#pragma once

#include "task.cuh"
#include "task_queue.cuh"

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
	queue->overflow = 0;

        queue->last_progress_time = get_global_time();
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
    // Reserve dynamic shared memory to control block residency.
    extern __shared__ unsigned char residency_smem[];

    (void)residency_smem;

    if (threadIdx.x != 0) {
        return;
    }

    while (atomicAdd(completed_count, 0) < num_tasks) {

        // Stop all workers if watchdog detects a hang.
        if (atomicAdd(&queue->watchdog_triggered, 0)) {
            return;
        }

        unsigned long long now = get_global_time();

	unsigned long long last_progress =
    		atomicAdd(&queue->last_progress_time, 0ULL);

	if (now - last_progress > WATCHDOG_TIMEOUT_NS) {
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

        task->start_time = get_global_time();

        // Simulate task execution.
        unsigned long long start = clock64();

        while (clock64() - start < task->duration) {
            // Simulated work.
        }

        task->end_time = get_global_time();

	// Record successful scheduler progress.
	atomicExch(
    		&queue->last_progress_time,
    		get_global_time()
	);
        __threadfence();

        atomicExch(&task->state, TASK_DONE);

        // Release dependent tasks.
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
