#ifndef TIMER_CORE_H
#define TIMER_CORE_H

#include <stdint.h>

// Define a C function pointer type for our trigger function.
// It matches the signature of mvsdk.CameraSoftTrigger (takes an int, returns an int/void).
typedef void (*trigger_func_ptr_t)(int);

// A struct to hold all the state for our timer.
// This is an "opaque pointer" to the Cython layer.
typedef struct TimerState TimerState;

// Functions exported by our C module
TimerState* timer_core_create(
    uint64_t interval_ns,
    trigger_func_ptr_t trigger_func,
    int hCamera,
    int busy_wait_ns,
    int priority
);

void timer_core_start(TimerState* state);
void timer_core_stop(TimerState* state);
void timer_core_destroy(TimerState* state);

#endif // TIMER_CORE_H
