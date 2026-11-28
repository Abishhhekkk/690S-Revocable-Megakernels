
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "task.cuh"
#include "task_queue.cuh"
#include "scheduler.cuh"
#include "scheduler_control.cuh"

#define NUM_TASKS 6

// =============================================================
// CUDA ERROR CHECKING
// =============================================================

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr,                                   \
                "\nCUDA ERROR at %s:%d\n"                     \
                "Call: %s\n"                                  \
                "Message: %s\n",                              \
                __FILE__, __LINE__, #call,                    \
                cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// =============================================================
// HOST-SIDE YIELD REQUEST
// =============================================================

// The control block lives in device memory: the A100 has no host-native
// atomics, so system-scope atomics on mapped host memory are not safe. The
// host sets the flag with an async copy on a non-blocking stream, which runs
// while the persistent kernel occupies the legacy default stream.
void host_request_yield(SchedulerControl* device_control,
                        cudaStream_t control_stream) {

    // Give GPU workers time to begin execution.
    std::this_thread::sleep_for(
        std::chrono::milliseconds(100)
    );

    static const int one = 1;
    CUDA_CHECK(cudaMemcpyAsync(&device_control->yield_requested, &one,
                               sizeof(int), cudaMemcpyHostToDevice,
                               control_stream));
    CUDA_CHECK(cudaStreamSynchronize(control_stream));

    printf("\nHOST: Yield requested!\n");
}

// =============================================================
// MAIN
// =============================================================

int main() {

    // =========================================================
    // 1. DEVICE CONFIGURATION
    // =========================================================

    int device = 0;

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int num_sms = prop.multiProcessorCount;

    printf("\n========== GPU CONFIGURATION ==========\n");

    printf("GPU: %s\n", prop.name);

    printf("Compute capability: %d.%d\n",
           prop.major, prop.minor);

    printf("SM count: %d\n", num_sms);

    // =========================================================
    // 2. CHECK REQUIRED HOST-GPU CAPABILITIES
    // =========================================================

    // (The control block is in device memory, so host-native atomics are
    // not required; see host_request_yield.)

    // =========================================================
    // 3. CHECK COOPERATIVE LAUNCH SUPPORT
    // =========================================================

    int cooperative_supported = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &cooperative_supported,
        cudaDevAttrCooperativeLaunch,
        device
    ));

    if (!cooperative_supported) {

        fprintf(stderr,
            "ERROR: Cooperative launch is not supported.\n");

        return EXIT_FAILURE;
    }

    printf("Cooperative launch: Supported\n");

    // =========================================================
    // 4. DEFINE TASK DEPENDENCY GRAPH
    // =========================================================

    Task host_tasks[NUM_TASKS] = {};

    // Task 0 -> Task 2
    host_tasks[0].id = 0;
    host_tasks[0].remaining_dependencies = 0;
    host_tasks[0].duration = 100000000;
    host_tasks[0].dependent_count = 1;
    host_tasks[0].dependents[0] = 2;

    // Task 1 -> Task 2
    host_tasks[1].id = 1;
    host_tasks[1].remaining_dependencies = 0;
    host_tasks[1].duration = 100000000;
    host_tasks[1].dependent_count = 1;
    host_tasks[1].dependents[0] = 2;

    // Task 2 -> Tasks 3 and 4
    host_tasks[2].id = 2;
    host_tasks[2].remaining_dependencies = 2;
    host_tasks[2].duration = 100000000;
    host_tasks[2].dependent_count = 2;
    host_tasks[2].dependents[0] = 3;
    host_tasks[2].dependents[1] = 4;

    // Task 3 -> Task 5
    host_tasks[3].id = 3;
    host_tasks[3].remaining_dependencies = 1;
    host_tasks[3].duration = 100000000;
    host_tasks[3].dependent_count = 1;
    host_tasks[3].dependents[0] = 5;

    // Task 4 -> Task 5
    host_tasks[4].id = 4;
    host_tasks[4].remaining_dependencies = 1;
    host_tasks[4].duration = 100000000;
    host_tasks[4].dependent_count = 1;
    host_tasks[4].dependents[0] = 5;

    // Task 5: Final task
    host_tasks[5].id = 5;
    host_tasks[5].remaining_dependencies = 2;
    host_tasks[5].duration = 100000000;
    host_tasks[5].dependent_count = 0;

    // =========================================================
    // 5. ALLOCATE DEVICE CONTROL MEMORY (+ host mirror)
    // =========================================================

    SchedulerControl host_mirror{};
    SchedulerControl* host_control = &host_mirror;   // read back after syncs
    SchedulerControl* device_control = nullptr;
    cudaStream_t control_stream = nullptr;

    CUDA_CHECK(cudaMalloc(&device_control, sizeof(SchedulerControl)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&control_stream,
                                         cudaStreamNonBlocking));

    host_control->yield_requested = 0;
    host_control->status = SCHEDULER_IDLE;
    host_control->total_workers = num_sms;
    host_control->yielded_workers = 0;
    host_control->resume_generation = 0;
    host_control->error_code = 0;
    host_control->yield_timestamp = 0;
    CUDA_CHECK(cudaMemcpy(device_control, host_control,
                          sizeof(SchedulerControl), cudaMemcpyHostToDevice));

    printf("\nA2: Device scheduler control memory allocated.\n");

    // =========================================================
    // 6. GPU MEMORY ALLOCATION
    // =========================================================

    Task* device_tasks = nullptr;
    TaskQueue* device_queue = nullptr;
    int* device_completed = nullptr;

    CUDA_CHECK(cudaMalloc(
        &device_tasks,
        sizeof(host_tasks)
    ));

    CUDA_CHECK(cudaMalloc(
        &device_queue,
        sizeof(TaskQueue)
    ));

    CUDA_CHECK(cudaMalloc(
        &device_completed,
        sizeof(int)
    ));

    CUDA_CHECK(cudaMemcpy(
        device_tasks,
        host_tasks,
        sizeof(host_tasks),
        cudaMemcpyHostToDevice
    ));

#ifdef ENABLE_TASK_LOG
    // ---- Part B1: per-task record log (see include/task_record.cuh) ----
    const uint32_t LOG_CAPACITY = 4096;
    TaskRecord* device_records = nullptr;
    TaskLog* device_log = nullptr;

    CUDA_CHECK(cudaMalloc(&device_records, LOG_CAPACITY * sizeof(TaskRecord)));
    CUDA_CHECK(cudaMemset(device_records, 0, LOG_CAPACITY * sizeof(TaskRecord)));
    CUDA_CHECK(cudaMalloc(&device_log, sizeof(TaskLog)));

    TaskLog init_log = {};
    init_log.capacity  = LOG_CAPACITY;
    init_log.launch_id = 0;   // main kernel; bumped before the resume launch
    init_log.step      = 0;   // toy kernel has a single step
    init_log.records   = device_records;
    CUDA_CHECK(cudaMemcpy(device_log, &init_log, sizeof(TaskLog),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(g_task_log, &device_log, sizeof(TaskLog*)));
#endif

    // =========================================================
    // 7. INITIALIZE SCHEDULER
    // =========================================================

    printf("\nLaunching scheduler initialization...\n");

    initialize_scheduler<<<1, 1>>>(
        device_tasks,
        NUM_TASKS,
        device_queue,
        device_completed
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Scheduler initialization successful.\n");

    // =========================================================
    // 8. CONFIGURE SHARED MEMORY
    // =========================================================

    size_t dynamic_smem_bytes =
        prop.sharedMemPerMultiprocessor / 2 + 1;

    size_t max_shared_memory =
        prop.sharedMemPerBlockOptin;

    if (max_shared_memory == 0) {
        max_shared_memory = prop.sharedMemPerBlock;
    }

    printf("\n========== SHARED MEMORY ==========\n");

    printf("Shared memory per SM: %zu bytes\n",
           prop.sharedMemPerMultiprocessor);

    printf("Maximum shared memory per block: %zu bytes\n",
           max_shared_memory);

    printf("Requested dynamic shared memory: %zu bytes\n",
           dynamic_smem_bytes);

    if (dynamic_smem_bytes > max_shared_memory) {

        fprintf(stderr,
            "ERROR: Requested shared memory exceeds block limit.\n");

        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaFuncSetAttribute(
        persistent_scheduler,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem_bytes)
    ));

    // =========================================================
    // 9. VERIFY OCCUPANCY
    // =========================================================

    int active_blocks_per_sm = 0;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        persistent_scheduler,
        1,
        dynamic_smem_bytes
    ));

    printf("Active blocks per SM: %d\n",
           active_blocks_per_sm);

    if (active_blocks_per_sm != 1) {

        fprintf(stderr,
            "ERROR: Expected exactly one active block per SM.\n");

        return EXIT_FAILURE;
    }

    int max_cooperative_blocks =
        active_blocks_per_sm * num_sms;

    if (num_sms > max_cooperative_blocks) {

        fprintf(stderr,
            "ERROR: Cooperative grid exceeds residency limit.\n");

        return EXIT_FAILURE;
    }

    // =========================================================
    // 10. INITIALIZE A2 WORKER COORDINATION
    // =========================================================

    host_control->total_workers = num_sms;
    host_control->yielded_workers = 0;
    host_control->yield_requested = 0;
    host_control->status = SCHEDULER_RUNNING;
    host_control->resume_generation = 0;
    CUDA_CHECK(cudaMemcpy(device_control, host_control,
                          sizeof(SchedulerControl), cudaMemcpyHostToDevice));

    printf("\nA2: Worker coordination initialized.\n");

    // =========================================================
    // 11. PREPARE COOPERATIVE KERNEL ARGUMENTS
    // =========================================================

    int num_tasks = NUM_TASKS;

    void* kernel_args[] = {
        &device_tasks,
        &num_tasks,
        &device_queue,
        &device_completed,
        &device_control
    };

    printf("\nLaunching %d persistent GPU workers...\n",
           num_sms);

    // =========================================================
    // 12. FIRST SCHEDULER LAUNCH
    // =========================================================

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)persistent_scheduler,
        dim3(num_sms),
        dim3(1),
        kernel_args,
        dynamic_smem_bytes,
        0
    ));

    CUDA_CHECK(cudaGetLastError());

    printf("GPU scheduler launched.\n");

    // =========================================================
    // 13. HOST REQUESTS YIELD
    // =========================================================

    std::thread yield_thread(
        host_request_yield,
        device_control,
        control_stream
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    yield_thread.join();

    printf("\nHOST: First GPU kernel terminated.\n");

    CUDA_CHECK(cudaMemcpy(host_control, device_control,
                          sizeof(SchedulerControl), cudaMemcpyDeviceToHost));

    // =========================================================
    // 14. CHECK YIELD RESULT
    // =========================================================

    int final_status =
        get_scheduler_status(host_control);

    int final_yielded_workers =
        cuda::atomic_ref<int, cuda::thread_scope_system>(
            host_control->yielded_workers
        ).load(cuda::memory_order_acquire);

    printf("\n========== A2 YIELD RESULTS ==========\n");

    printf("Scheduler status: %d\n", final_status);

    printf("Workers yielded: %d/%d\n",
           final_yielded_workers,
           num_sms);

    printf("Yield requested: %d\n",
           scheduler_yield_requested(host_control));

    if (final_status != SCHEDULER_YIELDED ||
        final_yielded_workers != num_sms) {

        fprintf(stderr,
            "\nERROR: Scheduler did not yield successfully.\n");

        CUDA_CHECK(cudaFree(device_tasks));
        CUDA_CHECK(cudaFree(device_queue));
        CUDA_CHECK(cudaFree(device_completed));
        CUDA_CHECK(cudaFree(device_control));
        CUDA_CHECK(cudaStreamDestroy(control_stream));

        return EXIT_FAILURE;
    }

    printf("\nSUCCESS: All workers yielded.\n");

    // =========================================================
    // 15. PREPARE SCHEDULER FOR RESUME
    // =========================================================

    printf("\n========== PREPARING RESUME ==========\n");

    prepare_scheduler_resume<<<1, 1>>>(
        device_control,
        num_sms
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Scheduler resume preparation complete.\n");

    // =========================================================
    // 16. SECOND SCHEDULER LAUNCH
    // =========================================================

    printf("\n========== RESUMING SCHEDULER ==========\n");

#ifdef ENABLE_TASK_LOG
    {
        uint32_t resume_launch_id = 1;   // records from the relaunch
        CUDA_CHECK(cudaMemcpy(&device_log->launch_id, &resume_launch_id,
                              sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
#endif

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)persistent_scheduler,
        dim3(num_sms),
        dim3(1),
        kernel_args,
        dynamic_smem_bytes,
        0
    ));

    CUDA_CHECK(cudaGetLastError());

    printf("Scheduler relaunched.\n");

    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Resumed scheduler terminated.\n");

    // =========================================================
    // 17. RETRIEVE FINAL SCHEDULER STATUS
    // =========================================================

    CUDA_CHECK(cudaMemcpy(host_control, device_control,
                          sizeof(SchedulerControl), cudaMemcpyDeviceToHost));

    final_status = get_scheduler_status(host_control);

    printf("\n========== FINAL SCHEDULER STATUS ==========\n");

    printf("Scheduler status: %d\n", final_status);

    printf("Resume generation: %d\n",
           host_control->resume_generation);

    // =========================================================
    // 18. RETRIEVE TASK RESULTS
    // =========================================================

    int completed = 0;

    CUDA_CHECK(cudaMemcpy(
        &completed,
        device_completed,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    TaskQueue host_queue{};

    CUDA_CHECK(cudaMemcpy(
        &host_queue,
        device_queue,
        sizeof(TaskQueue),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        host_tasks,
        device_tasks,
        sizeof(host_tasks),
        cudaMemcpyDeviceToHost
    ));

    printf("\n========== FINAL EXECUTION RESULTS ==========\n");

    printf("Watchdog: %s\n",
           host_queue.watchdog_triggered ? "TRIGGERED" : "OK");

    printf("Queue overflow: %s\n",
           host_queue.overflow ? "YES" : "NO");

    bool exactly_once = true;

    for (int i = 0; i < NUM_TASKS; i++) {

        printf(
            "Task %d | State: %d | Executions: %d | "
            "Start: %llu | End: %llu\n",
            i,
            host_tasks[i].state,
            host_tasks[i].execution_count,
            host_tasks[i].start_time,
            host_tasks[i].end_time
        );

        if (host_tasks[i].state != TASK_DONE ||
            host_tasks[i].execution_count != 1) {

            exactly_once = false;
        }
    }

    printf("\nCompleted tasks: %d/%d\n",
           completed, NUM_TASKS);

    // =========================================================
    // 19. FINAL VALIDATION
    // =========================================================

    bool success =
        final_status == SCHEDULER_COMPLETED &&
        completed == NUM_TASKS &&
        exactly_once &&
        !host_queue.overflow &&
        !host_queue.watchdog_triggered;

    if (success) {

        printf(
            "\nSUCCESS: All tasks completed exactly once "
            "across both scheduler launches.\n"
        );

    } else {

        fprintf(stderr,
            "\nFAILURE: Scheduler validation failed.\n");
    }

#ifdef ENABLE_TASK_LOG
    {
        // Dump records + task graph for tools/checker.py.
        TaskLog final_log;
        CUDA_CHECK(cudaMemcpy(&final_log, device_log, sizeof(TaskLog),
                              cudaMemcpyDeviceToHost));

        uint32_t n = final_log.count < final_log.capacity ? final_log.count
                                                          : final_log.capacity;
        if (final_log.count > final_log.capacity) {
            fprintf(stderr, "WARNING: task log overflowed (%u > %u records)\n",
                    final_log.count, final_log.capacity);
        }

        std::vector<TaskRecord> recs(n);
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(recs.data(), device_records,
                                  n * sizeof(TaskRecord),
                                  cudaMemcpyDeviceToHost));
        }

        FILE* fb = fopen("task_log.bin", "wb");
        if (fb) {
            fwrite(recs.data(), sizeof(TaskRecord), n, fb);
            fclose(fb);
        }

        // dependencies are the inverse of the `dependents` lists
        std::vector<std::vector<int>> deps(NUM_TASKS);
        for (int i = 0; i < NUM_TASKS; i++) {
            for (int j = 0; j < host_tasks[i].dependent_count; j++) {
                deps[host_tasks[i].dependents[j]].push_back(i);
            }
        }

        FILE* fg = fopen("task_graph.json", "w");
        if (fg) {
            fprintf(fg, "{\n  \"num_steps\": 1,\n  \"cross_step\": \"none\",\n");
            fprintf(fg, "  \"log_count\": %u,\n  \"log_capacity\": %u,\n",
                    final_log.count, final_log.capacity);
            fprintf(fg, "  \"tasks\": [\n");
            for (int i = 0; i < NUM_TASKS; i++) {
                fprintf(fg, "    {\"id\": %d, \"deps\": [", i);
                for (size_t k = 0; k < deps[i].size(); k++) {
                    fprintf(fg, "%s%d", k ? ", " : "", deps[i][k]);
                }
                fprintf(fg, "]}%s\n", i + 1 < NUM_TASKS ? "," : "");
            }
            fprintf(fg, "  ]\n}\n");
            fclose(fg);
        }
        printf("Wrote task_log.bin (%u records) and task_graph.json\n", n);
    }
#endif

    // =========================================================
    // 20. CLEANUP
    // =========================================================

    CUDA_CHECK(cudaFree(device_tasks));
    CUDA_CHECK(cudaFree(device_queue));
    CUDA_CHECK(cudaFree(device_completed));

    CUDA_CHECK(cudaFree(device_control));
    CUDA_CHECK(cudaStreamDestroy(control_stream));
#ifdef ENABLE_TASK_LOG
    CUDA_CHECK(cudaFree(device_records));
    CUDA_CHECK(cudaFree(device_log));
#endif

    printf("\nA2 execution finished.\n");

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
