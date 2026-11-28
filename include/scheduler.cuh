
#pragma once

#include "task.cuh"
#include "task_queue.cuh"
#include "scheduler_control.cuh"
#include "task_record.cuh"

#define WATCHDOG_TIMEOUT_NS 5000000000ULL

// =============================================================
// GPU GLOBAL TIMER
// =============================================================

__device__ inline unsigned long long get_global_time() {
    unsigned long long time;

    asm volatile(
        "mov.u64 %0, %%globaltimer;"
        : "=l"(time)
    );

    return time;
}

// =============================================================
// INITIALIZE SCHEDULER
// Called only for a fresh execution, never for resume.
// =============================================================

__global__ void initialize_scheduler(
    Task* tasks,
    int num_tasks,
    TaskQueue* queue,
    int* completed_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

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

            if (!enqueue(queue, i)) {
                queue->overflow = 1;
            }
        }
    }
}

// =============================================================
// REGISTER ONE WORKER AS YIELDED
// =============================================================

__device__ inline void register_worker_yield(
    SchedulerControl* control
) {
    cuda::atomic_ref<int, cuda::thread_scope_system>
        yielded(control->yielded_workers);

    cuda::atomic_ref<int, cuda::thread_scope_system>
        status(control->status);

    int previous = yielded.fetch_add(
        1,
        cuda::memory_order_acq_rel
    );

    // The final worker reports YIELDED only if the
    // scheduler has not already completed or failed.
    if (previous + 1 == control->total_workers) {

        int expected = SCHEDULER_RUNNING;

        status.compare_exchange_strong(
            expected,
            SCHEDULER_YIELDED,
            cuda::memory_order_acq_rel,
            cuda::memory_order_acquire
        );
    }
}

// =============================================================
// PERSISTENT GPU SCHEDULER WITH YIELD SUPPORT
// =============================================================

__global__ void persistent_scheduler(
    Task* tasks,
    int num_tasks,
    TaskQueue* queue,
    int* completed_count,
    SchedulerControl* control
) {
    // Preserve the existing shared-memory residency strategy.
    extern __shared__ unsigned char residency_smem[];

    (void)residency_smem;

    // One active worker thread per block.
    if (threadIdx.x != 0) {
        return;
    }

    cuda::atomic_ref<int, cuda::thread_scope_system>
        yield_flag(control->yield_requested);

    cuda::atomic_ref<int, cuda::thread_scope_system>
        scheduler_status(control->status);

    // =========================================================
    // WORKER LOOP
    // =========================================================

    while (atomicAdd(completed_count, 0) < num_tasks) {

        // -----------------------------------------------------
        // 1. CHECK WATCHDOG
        // -----------------------------------------------------

        if (atomicAdd(&queue->watchdog_triggered, 0)) {
            return;
        }

        unsigned long long now = get_global_time();

        unsigned long long last_progress =
            atomicAdd(&queue->last_progress_time, 0ULL);

        if (now - last_progress > WATCHDOG_TIMEOUT_NS) {

            atomicExch(&queue->watchdog_triggered, 1);

            scheduler_status.store(
                SCHEDULER_ERROR,
                cuda::memory_order_release
            );

            return;
        }

        // -----------------------------------------------------
        // 2. CHECK FOR YIELD BEFORE DEQUEUE
        // -----------------------------------------------------

        if (yield_flag.load(cuda::memory_order_acquire)) {

            register_worker_yield(control);
            return;
        }

        // -----------------------------------------------------
        // 3. DEQUEUE A READY TASK
        // -----------------------------------------------------

        int task_id = dequeue(queue);

        if (task_id < 0) {
            continue;
        }

        if (task_id >= num_tasks) {

            scheduler_status.store(
                SCHEDULER_ERROR,
                cuda::memory_order_release
            );

            return;
        }

        Task* task = &tasks[task_id];

        // -----------------------------------------------------
        // 4. CLAIM TASK
        // -----------------------------------------------------

        if (atomicCAS(
                &task->state,
                TASK_READY,
                TASK_RUNNING
            ) != TASK_READY) {

            continue;
        }

        // A host yield may have arrived after the first check
        // or while this worker was claiming the task.
        //
        // Return the unstarted task to the queue.

        if (yield_flag.load(cuda::memory_order_acquire)) {

            atomicExch(&task->state, TASK_READY);

            if (!enqueue(queue, task_id)) {

                scheduler_status.store(
                    SCHEDULER_ERROR,
                    cuda::memory_order_release
                );

                return;
            }

            register_worker_yield(control);
            return;
        }

        // -----------------------------------------------------
        // 5. EXECUTE CURRENT TASK TO COMPLETION
        // -----------------------------------------------------

        atomicAdd(&task->execution_count, 1);

        unsigned long long t_start = get_global_time();
        task->start_time = t_start;
        uint32_t log_slot = log_task_begin((uint32_t)task_id, t_start);  // no-op unless ENABLE_TASK_LOG

        unsigned long long start = clock64();

        while (clock64() - start < task->duration) {
            // Simulated task execution.
        }

        unsigned long long t_end = get_global_time();
        task->end_time = t_end;
        log_task_end(log_slot, t_end);  // before dependents are released

        // -----------------------------------------------------
        // 6. MARK TASK COMPLETE
        // -----------------------------------------------------

        __threadfence();

        atomicExch(&task->state, TASK_DONE);

        atomicExch(
            &queue->last_progress_time,
            get_global_time()
        );

        // -----------------------------------------------------
        // 7. UPDATE DEPENDENCY COUNTERS
        // -----------------------------------------------------

        for (int i = 0; i < task->dependent_count; i++) {

            int dependent_id = task->dependents[i];

            int old_value = atomicSub(
                &tasks[dependent_id].remaining_dependencies,
                1
            );

            if (old_value == 1) {

                atomicExch(
                    &tasks[dependent_id].state,
                    TASK_READY
                );

                if (!enqueue(queue, dependent_id)) {

                    scheduler_status.store(
                        SCHEDULER_ERROR,
                        cuda::memory_order_release
                    );

                    return;
                }
            }
        }

        // -----------------------------------------------------
        // 8. UPDATE GLOBAL COMPLETION COUNT
        // -----------------------------------------------------

        int previous_completed =
            atomicAdd(completed_count, 1);

        if (previous_completed + 1 == num_tasks) {

            scheduler_status.store(
                SCHEDULER_COMPLETED,
                cuda::memory_order_release
            );

            return;
        }

        // -----------------------------------------------------
        // 9. CHECK YIELD ON NEXT ITERATION
        // -----------------------------------------------------

        // The worker will check the yield flag before
        // dequeuing another task.
    }
}

// =============================================================
// PREPARE SCHEDULER FOR RESUME
// Does NOT reset tasks, queue, or completed_count.
// =============================================================

__global__ void prepare_scheduler_resume(
    SchedulerControl* control,
    int num_workers
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {

        control->yielded_workers = 0;
        control->yield_requested = 0;
        control->total_workers = num_workers;

        control->resume_generation++;
        control->error_code = 0;

        cuda::atomic_ref<int, cuda::thread_scope_system>
            status(control->status);

        status.store(
            SCHEDULER_RUNNING,
            cuda::memory_order_release
        );
    }
}
