#pragma once

// Per-task execution records for the independent correctness checker (Part B1).
//
// Everything here is compiled in only when ENABLE_TASK_LOG is defined, so timing
// runs (A5, B6) pay nothing for it. The record layout is shared with
// tools/checker.py (numpy dtype TASK_RECORD_DTYPE) -- change both together.

#include <cuda_runtime.h>
#include <stdint.h>

// 40 bytes, 8-byte aligned. Keep field order in sync with checker.py.
struct TaskRecord {
    uint64_t start_ns;   // %globaltimer when the task body began
    uint64_t end_ns;     // %globaltimer when the task body finished (0 = never finished)
    uint32_t task_id;    // node id inside one step's task graph
    uint32_t step;       // decode step / iteration index
    uint32_t launch_id;  // 0 = main kernel, then relaunches and helper launches
    uint32_t sm_id;      // %smid of the executing block
    uint32_t worker_id;  // logical worker (blockIdx.x in the toy kernel)
    uint32_t reserved;   // padding / future flags
};
static_assert(sizeof(TaskRecord) == 40, "TaskRecord layout changed; update checker.py");

// Log header + fixed-capacity record array living in device global memory.
// Records are APPENDED with atomicAdd. Do not index slots by (step, task_id):
// that would overwrite a duplicate execution and hide exactly the bug we want.
struct TaskLog {
    uint32_t count;      // next free slot (may exceed capacity if overflowed)
    uint32_t capacity;   // number of records in `records`
    uint32_t launch_id;  // host sets this before every kernel launch
    uint32_t step;       // current decode step (toy kernel: always 0)
    TaskRecord* records; // device pointer, `capacity` entries
};

#ifdef ENABLE_TASK_LOG

// Set once from the host with cudaMemcpyToSymbol. A global symbol (instead of a
// kernel argument) keeps the persistent_scheduler signature unchanged.
__device__ TaskLog* g_task_log = nullptr;

__device__ __forceinline__ uint32_t read_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// Claim a slot and write the start half of the record. Returns the slot index,
// or 0xFFFFFFFF if logging is off or the buffer is full.
// Called once per task execution, right after the task was claimed (state
// READY -> RUNNING) and immediately before the body runs.
__device__ __forceinline__ uint32_t log_task_begin(uint32_t task_id, uint64_t start_ns) {
    TaskLog* log = g_task_log;
    if (log == nullptr) return 0xFFFFFFFFu;

    uint32_t slot = atomicAdd(&log->count, 1u);
    if (slot >= log->capacity) return 0xFFFFFFFFu;  // host detects count > capacity

    TaskRecord* r = &log->records[slot];
    r->start_ns  = start_ns;
    r->end_ns    = 0;                 // stays 0 if the task never finishes (hang)
    r->task_id   = task_id;
    r->step      = log->step;
    r->launch_id = log->launch_id;
    r->sm_id     = read_smid();
    r->worker_id = blockIdx.x;
    r->reserved  = 0;
    return slot;
}

// Write the end timestamp. Must run BEFORE dependents are released so that a
// correct execution always satisfies end(dep) <= start(dependent).
__device__ __forceinline__ void log_task_end(uint32_t slot, uint64_t end_ns) {
    TaskLog* log = g_task_log;
    if (log == nullptr || slot == 0xFFFFFFFFu) return;
    log->records[slot].end_ns = end_ns;
}

#else  // ENABLE_TASK_LOG not defined: compile to nothing

__device__ __forceinline__ uint32_t log_task_begin(uint32_t, uint64_t) { return 0xFFFFFFFFu; }
__device__ __forceinline__ void     log_task_end(uint32_t, uint64_t) {}

#endif  // ENABLE_TASK_LOG
