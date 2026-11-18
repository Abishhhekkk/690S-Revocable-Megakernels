
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "task.cuh"
#include "task_queue.cuh"
#include "scheduler.cuh"

#define NUM_TASKS 6
#define NUM_WORKERS 4

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error: %s\n",               \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                              \
        }                                                     \
    } while (0)

int main() {
    Task host_tasks[NUM_TASKS] = {};

    // Task 0: No dependencies -> Task 2
    host_tasks[0].id = 0;
    host_tasks[0].remaining_dependencies = 0;
    host_tasks[0].duration = 5000000;
    host_tasks[0].dependent_count = 1;
    host_tasks[0].dependents[0] = 2;

    // Task 1: No dependencies -> Task 2
    host_tasks[1].id = 1;
    host_tasks[1].remaining_dependencies = 0;
    host_tasks[1].duration = 5000000;
    host_tasks[1].dependent_count = 1;
    host_tasks[1].dependents[0] = 2;

    // Task 2: Depends on Tasks 0 and 1 -> Tasks 3 and 4
    host_tasks[2].id = 2;
    host_tasks[2].remaining_dependencies = 2;
    host_tasks[2].duration = 5000000;
    host_tasks[2].dependent_count = 2;
    host_tasks[2].dependents[0] = 3;
    host_tasks[2].dependents[1] = 4;

    // Task 3: Depends on Task 2 -> Task 5
    host_tasks[3].id = 3;
    host_tasks[3].remaining_dependencies = 1;
    host_tasks[3].duration = 5000000;
    host_tasks[3].dependent_count = 1;
    host_tasks[3].dependents[0] = 5;

    // Task 4: Depends on Task 2 -> Task 5
    host_tasks[4].id = 4;
    host_tasks[4].remaining_dependencies = 1;
    host_tasks[4].duration = 5000000;
    host_tasks[4].dependent_count = 1;
    host_tasks[4].dependents[0] = 5;

    // Task 5: Depends on Tasks 3 and 4
    host_tasks[5].id = 5;
    host_tasks[5].remaining_dependencies = 2;
    host_tasks[5].duration = 5000000;
    host_tasks[5].dependent_count = 0;

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

    printf("Launching scheduler initialization...\n");

    initialize_scheduler<<<1, 1>>>(
        device_tasks,
        NUM_TASKS,
        device_queue,
        device_completed
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Launching %d persistent GPU workers...\n", NUM_WORKERS);

    persistent_scheduler<<<NUM_WORKERS, 1>>>(
        device_tasks,
        NUM_TASKS,
        device_queue,
        device_completed
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int completed = 0;

    CUDA_CHECK(cudaMemcpy(
        &completed,
        device_completed,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

TaskQueue host_queue;

CUDA_CHECK(cudaMemcpy(
    &host_queue,
    device_queue,
    sizeof(TaskQueue),
    cudaMemcpyDeviceToHost
));

printf("Watchdog: %s\n",
       host_queue.watchdog_triggered ? "TRIGGERED" : "OK");



    CUDA_CHECK(cudaMemcpy(
        host_tasks,
        device_tasks,
        sizeof(host_tasks),
        cudaMemcpyDeviceToHost
    ));

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

    printf("\nCompleted tasks: %d/%d\n", completed, NUM_TASKS);

    if (completed == NUM_TASKS && exactly_once) {
        printf("SUCCESS: All tasks completed exactly once!\n");
    } else {
        printf("FAILURE: Correctness validation failed.\n");
    }

    CUDA_CHECK(cudaFree(device_tasks));
    CUDA_CHECK(cudaFree(device_queue));
    CUDA_CHECK(cudaFree(device_completed));

    return (completed == NUM_TASKS && exactly_once)
        ? EXIT_SUCCESS : EXIT_FAILURE;
}
