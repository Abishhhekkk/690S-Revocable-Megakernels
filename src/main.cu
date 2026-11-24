
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "task.cuh"
#include "task_queue.cuh"
#include "scheduler.cuh"

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
// MAIN
// =============================================================

int main() {

    // =========================================================
    // 1. DEFINE TASK DEPENDENCY GRAPH
    // =========================================================

    Task host_tasks[NUM_TASKS] = {};

    // Task 0 -> Task 2
    host_tasks[0].id = 0;
    host_tasks[0].remaining_dependencies = 0;
    host_tasks[0].duration = 5000000;
    host_tasks[0].dependent_count = 1;
    host_tasks[0].dependents[0] = 2;

    // Task 1 -> Task 2
    host_tasks[1].id = 1;
    host_tasks[1].remaining_dependencies = 0;
    host_tasks[1].duration = 5000000;
    host_tasks[1].dependent_count = 1;
    host_tasks[1].dependents[0] = 2;

    // Task 2 -> Tasks 3 and 4
    host_tasks[2].id = 2;
    host_tasks[2].remaining_dependencies = 2;
    host_tasks[2].duration = 5000000;
    host_tasks[2].dependent_count = 2;
    host_tasks[2].dependents[0] = 3;
    host_tasks[2].dependents[1] = 4;

    // Task 3 -> Task 5
    host_tasks[3].id = 3;
    host_tasks[3].remaining_dependencies = 1;
    host_tasks[3].duration = 5000000;
    host_tasks[3].dependent_count = 1;
    host_tasks[3].dependents[0] = 5;

    // Task 4 -> Task 5
    host_tasks[4].id = 4;
    host_tasks[4].remaining_dependencies = 1;
    host_tasks[4].duration = 5000000;
    host_tasks[4].dependent_count = 1;
    host_tasks[4].dependents[0] = 5;

    // Task 5: Final task
    host_tasks[5].id = 5;
    host_tasks[5].remaining_dependencies = 2;
    host_tasks[5].duration = 5000000;
    host_tasks[5].dependent_count = 0;

    // =========================================================
    // 2. GPU MEMORY ALLOCATION
    // =========================================================

    Task* device_tasks = nullptr;
    TaskQueue* device_queue = nullptr;
    int* device_completed = nullptr;

    CUDA_CHECK(cudaMalloc(&device_tasks, sizeof(host_tasks)));
    CUDA_CHECK(cudaMalloc(&device_queue, sizeof(TaskQueue)));
    CUDA_CHECK(cudaMalloc(&device_completed, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        device_tasks,
        host_tasks,
        sizeof(host_tasks),
        cudaMemcpyHostToDevice
    ));

    // =========================================================
    // 3. INITIALIZE SCHEDULER
    // =========================================================

    printf("Launching scheduler initialization...\n");

    initialize_scheduler<<<1, 1>>>(
        device_tasks,
        NUM_TASKS,
        device_queue,
        device_completed
    );

    // Check initialization kernel launch.
    CUDA_CHECK(cudaGetLastError());

    // Check initialization kernel execution.
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Scheduler initialization successful.\n");

    // =========================================================
    // 4. QUERY GPU PROPERTIES
    // =========================================================

    int device = 0;
    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int num_sms = prop.multiProcessorCount;

    printf("\n========== GPU CONFIGURATION ==========\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n",
           prop.major, prop.minor);
    printf("SM count: %d\n", num_sms);

    // =========================================================
    // 5. CHECK COOPERATIVE LAUNCH SUPPORT
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
    // 6. CONFIGURE SHARED MEMORY
    // =========================================================

    size_t dynamic_smem_bytes =
        prop.sharedMemPerMultiprocessor / 2 + 1;

    size_t max_shared_memory = prop.sharedMemPerBlockOptin;

    if (max_shared_memory == 0) {
        max_shared_memory = prop.sharedMemPerBlock;
    }

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
    // 7. VERIFY OCCUPANCY
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
    // 8. COOPERATIVE KERNEL LAUNCH
    // =========================================================

    printf("\nLaunching %d persistent GPU workers...\n", num_sms);

    int num_tasks = NUM_TASKS;

    void* kernel_args[] = {
        &device_tasks,
        &num_tasks,
        &device_queue,
        &device_completed
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)persistent_scheduler,
        dim3(num_sms),
        dim3(1),
        kernel_args,
        dynamic_smem_bytes,
        0
    ));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Persistent scheduler finished.\n");

    // =========================================================
    // 9. RETRIEVE RESULTS
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

    printf("\nWatchdog: %s\n",
           host_queue.watchdog_triggered ? "TRIGGERED" : "OK");

    printf("Queue overflow: %s\n",
       	   host_queue.overflow ? "YES" : "NO");
    
    CUDA_CHECK(cudaMemcpy(
        host_tasks,
        device_tasks,
        sizeof(host_tasks),
        cudaMemcpyDeviceToHost
    ));

    // =========================================================
    // 10. VALIDATE EXECUTION
    // =========================================================

    printf("\n========== EXECUTION RESULTS ==========\n");

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

    if (completed == NUM_TASKS && exactly_once) {
        printf("SUCCESS: All tasks completed exactly once!\n");
    } else {
        printf("FAILURE: Correctness validation failed.\n");
    }

    // =========================================================
    // 11. CLEANUP
    // =========================================================

    CUDA_CHECK(cudaFree(device_tasks));
    CUDA_CHECK(cudaFree(device_queue));
    CUDA_CHECK(cudaFree(device_completed));

    return (completed == NUM_TASKS && exactly_once)
        ? EXIT_SUCCESS : EXIT_FAILURE;
}
