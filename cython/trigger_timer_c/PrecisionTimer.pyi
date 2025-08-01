# cython: language_level=3
# -*- coding: utf-8 -*-

from typing import Any, Callable

class PrecisionTimer:
    """
    A high-precision timer that uses a dedicated C-level thread to trigger
    a callback function at a specified interval.

    This class is a Python wrapper around a C core that leverages
    Windows' high-performance counters (QueryPerformanceCounter) for
    microsecond-level accuracy.
    """
    
    is_running: bool
    """(Read-only) True if the timer thread is currently active, False otherwise."""

    def __init__(
        self,
        interval_s: float,
        hCamera: int,
        c_trigger_func: Callable[[int], None] | Any,
        busy_wait_us: int = 2000,
        priority: int = 0
    ) -> None:
        """
        Initializes the high-precision timer.

        Parameters
        ----------
        interval_s : float
            The desired time interval between triggers, in seconds.
        c_trigger_func : ctypes function object or similar
            The C-level trigger function to call. This should be a function
            pointer from a loaded library, e.g., `mvsdk._sdk.CameraSoftTrigger`.
            The expected signature is `void(int)`.
        hCamera : int
            The integer handle to the camera, which will be passed to the
            trigger function.
        busy_wait_us : int, optional
            The duration in microseconds (µs) for the final busy-wait loop.
            This determines the trade-off between CPU usage and timing precision.
            It should be slightly longer than the system's timer resolution
            (which is set to 1ms by this timer). A value of 2000 (2ms) is a
            good starting point. Default is 2000.
        priority : int, optional
            The priority of the internal timer thread.
            - 0: Normal (default)
            - 1: Above Normal
            - 2: Highest
            - 3: Time Critical (use with caution, can affect system stability)
        """
        ...

    def start(self) -> None:
        """
        Starts the timer thread. The callback function will begin to be
        called at the specified interval.
        """
        ...

    def stop(self) -> None:
        """
        Signals the timer thread to stop and waits for it to terminate.
        This is a blocking call.
        """
        ...

    def join(self) -> None:
        """
        Waits for the timer thread to complete.
        Note: In the current implementation, `stop()` is blocking and already
        performs the join operation, so this method is mainly for API
        compatibility.
        """
        ...

    def __dealloc__(self) -> None: ...
