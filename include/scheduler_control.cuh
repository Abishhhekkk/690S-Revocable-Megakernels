
#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdint.h>

// =============================================================
// SCHEDULER CONTROL STATES
// =============================================================

enum SchedulerStatus : int {
    SCHEDULER_IDLE = 0,
    SCHEDULER_RUNNING = 1,
    SCHEDULER_YIELD_REQUESTED = 2,
    SCHEDULER_YIELDED = 3,
    SCHEDULER_COMPLETED = 4,
    SCHEDULER_ERROR = 5
};

// =============================================================
// SCHEDULER CONTROL STRUCTURE
// =============================================================

struct alignas(8) SchedulerControl {

    int yield_requested;
    int status;

    int total_workers;
    int yielded_workers;

    int resume_generation;
    int error_code;

    unsigned long long yield_timestamp;
};

// =============================================================
// SYSTEM-SCOPE ATOMIC HELPERS
// =============================================================

// Host and GPU can use these helpers to access the same
// control fields with system-scope atomic operations.

__host__ __device__ inline void request_scheduler_yield(
    SchedulerControl* control
) {
    cuda::atomic_ref<int, cuda::thread_scope_system> flag(
        control->yield_requested
    );

    flag.store(1, cuda::memory_order_release);
}

__host__ __device__ inline int scheduler_yield_requested(
    SchedulerControl* control
) {
    cuda::atomic_ref<int, cuda::thread_scope_system> flag(
        control->yield_requested
    );

    return flag.load(cuda::memory_order_acquire);
}

__host__ __device__ inline void set_scheduler_status(
    SchedulerControl* control,
    int status
) {
    cuda::atomic_ref<int, cuda::thread_scope_system> state(
        control->status
    );

    state.store(status, cuda::memory_order_release);
}

__host__ __device__ inline int get_scheduler_status(
    SchedulerControl* control
) {
    cuda::atomic_ref<int, cuda::thread_scope_system> state(
        control->status
    );

    return state.load(cuda::memory_order_acquire);
}
