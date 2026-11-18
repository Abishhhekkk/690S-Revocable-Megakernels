
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Possible execution states of a task
enum TaskState : int {
    TASK_WAITING = 0,
    TASK_READY   = 1,
    TASK_RUNNING = 2,
    TASK_DONE    = 3
};

// Represents one node in our task graph
struct Task {
    int id;

    // Number of unfinished dependencies
    int remaining_dependencies;

    // Execution state
    int state;

    // Simulated execution duration in clock cycles
    unsigned long long duration;

    // IDs of tasks that depend on this task
    int dependent_count;
    int dependents[4];

    // Execution tracking
    unsigned long long start_time;
    unsigned long long end_time;

    // Number of times this task was executed
    int execution_count;
};
