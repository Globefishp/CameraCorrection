#include "timer_core.h"
#include <windows.h>
#include <process.h> // For _beginthreadex
#include <stdlib.h>  // For malloc/free

// C结构，用于保存一个定时器实例的所有状态
struct TimerState {
    // 定时参数，单位为 QPC "ticks"
    LONGLONG interval_ticks;
    LONGLONG busy_wait_ticks;

    // 用户传入的相机句柄和回调函数指针
    int hCamera;
    trigger_func_ptr_t trigger_func;

    // 线程和控制标志
    HANDLE hThread;
    volatile int stop_flag; // volatile至关重要，防止编译器过度优化
    int priority;           // 线程优先级

    // Windows特有的计时器频率
    LARGE_INTEGER performance_frequency;
};

// 线程的主循环函数 (Windows版本)
// 注意返回值和调用约定
static unsigned __stdcall timer_loop(void* arg) {
    TimerState* state = (TimerState*)arg;

    // --- 线程初始化 ---
    // 1. 提升系统定时器精度到1ms
    timeBeginPeriod(1);

    // 2. 根据传入的参数设置线程优先级
    switch (state->priority) {
        case 1: SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL); break;
        case 2: SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); break;
        case 3: SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL); break;
        default: SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL); break;
    }
    
    LARGE_INTEGER current_ticks;
    LONGLONG next_trigger_ticks;

    // 初始化第一个触发时间点
    QueryPerformanceCounter(&current_ticks);
    next_trigger_ticks = current_ticks.QuadPart + state->interval_ticks;

    while (!state->stop_flag) {
        // --- 1. 粗略等待 ---
        QueryPerformanceCounter(&current_ticks);
        
        // 计算需要等待的ticks数，留出忙等待的余量
        LONGLONG ticks_to_wait = next_trigger_ticks - current_ticks.QuadPart - state->busy_wait_ticks;
        
        if (ticks_to_wait > 0) {
            // 将ticks转换为毫秒以供Sleep()函数使用
            // 乘以1000再除以频率，以保持精度
            DWORD sleep_ms = (DWORD)((ticks_to_wait * 1000) / state->performance_frequency.QuadPart);
            if (sleep_ms > 0) {
                Sleep(sleep_ms);
            }
        }
        
        // --- 2. 精确的忙等待 ---
        // 使用QPC循环检查，直到到达精确的触发时间
        // 此处逻辑上不能优化到上一个函数内部。（等待时间不足以sleep但仍有忙等待）
        while (1) {
            QueryPerformanceCounter(&current_ticks);
            if (current_ticks.QuadPart >= next_trigger_ticks) break;
        }

        // --- 3. 直接调用C函数指针 ---
        // 这是性能的关键，完全在C层面，没有GIL
        state->trigger_func(state->hCamera);
        
        // --- 4. 计算下一次触发时间 ---
        next_trigger_ticks += state->interval_ticks;
    }

    // --- 线程清理 ---
    // 恢复系统默认的定时器精度
    timeEndPeriod(1);
    
    return 0;
}

// --- 公共API实现 ---

// “构造函数”
TimerState* timer_core_create(
    uint64_t interval_ns,
    trigger_func_ptr_t trigger_func,
    int hCamera,
    int busy_wait_ns,
    int priority
) {
    TimerState* state = (TimerState*)malloc(sizeof(TimerState));
    if (!state) return NULL;

    state->hCamera = hCamera;
    state->trigger_func = trigger_func;
    state->priority = priority;
    state->stop_flag = 0;
    state->hThread = NULL;

    // 获取高精度计时器的频率（每秒多少ticks）
    if (!QueryPerformanceFrequency(&state->performance_frequency)) {
        free(state);
        return NULL; // 系统不支持高精度计时器
    }

    // 将用户输入的纳秒单位，转换为内部使用的ticks单位
    state->interval_ticks = (LONGLONG)((double)interval_ns * state->performance_frequency.QuadPart / 1e9);
    state->busy_wait_ticks = (LONGLONG)((double)busy_wait_ns * state->performance_frequency.QuadPart / 1e9);

    return state;
}

// “启动方法”
void timer_core_start(TimerState* state) {
    if (!state) return;
    state->stop_flag = 0;
    // 使用_beginthreadex安全地创建C运行时兼容的线程
    state->hThread = (HANDLE)_beginthreadex(NULL, 0, &timer_loop, state, 0, NULL);
}

// “停止方法” (阻塞式)
void timer_core_stop(TimerState* state) {
    if (!state || !state->hThread) return;
    
    state->stop_flag = 1;
    
    // 等待线程执行完毕
    WaitForSingleObject(state->hThread, INFINITE);
    
    // 清理线程句柄
    CloseHandle(state->hThread);
    state->hThread = NULL;
}

// “析构函数”
void timer_core_destroy(TimerState* state) {
    if (state) {
        // 确保线程已停止，防止内存泄漏或野指针
        if (state->hThread) {
            timer_core_stop(state);
        }
        free(state);
    }
}
