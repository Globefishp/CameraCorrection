# 假设您的SDK模块和相机类已定义
# from cam_frame_system.camera import mvsdk
# from cam_frame_system.camera.huateng_camera_v2_tc_raw import Camera

# 导入新编译的模块
import PrecisionTimer 
import time
import numpy as np
import matplotlib.pyplot as plt

import mvsdk
from huateng_camera_v2_tc_raw_mod import Camera


# --- Configuration ---
FRAME_RATE = 10    # FPS
TEST_DURATION = 120  # seconds

# --- Main Application ---
# 1. Initialize camera in Python to get the handle
DevList = mvsdk.CameraEnumerateDevice()
mycam = Camera(DevList[0], 10, gain=1, hibitdepth=1)
mycam.open()
hCamera = mycam.hCamera # The integer handle

mvsdk.CameraSetTriggerMode(mycam.hCamera, 1) # Soft Trigger
mvsdk.CameraSetTriggerCount(mycam.hCamera, 1) # 1 frame per trigger

try: 
    # 2. Instantiate our new Cython C-function timer
    interval_s = 1.0 / FRAME_RATE
    timer = PrecisionTimer.PrecisionTimer(
        interval_s=interval_s,
        c_trigger_func=mvsdk._sdk.CameraSoftTrigger, # Pass the ctypes function object
        hCamera=hCamera,
        busy_wait_us=2000, # Recommended: > 1000us to accommodate system timer resolution
        priority=2         # 3: TIME_CRITICAL for maximum precision
    )

    # 3. Main loop for grabbing frames
    imgs = []
    timestamps_cam = []
    timestamps_python = []
    print(f"Starting capture at {FRAME_RATE} FPS...")
    timer.start()
    start_time = time.time()

    frame_count = 0
    last_time = 0
    while time.time() - start_time < TEST_DURATION:
        # This loop is now "paced" by the C timer.
        # The C timer triggers the camera. We just need to grab the frame.
        # For maximum performance, grab() should be as fast as possible.
        try:
            img, timestamp = mycam.grab_raw() # This should ideally just fetch the latest frame
            if img is not None:
                frame_count += 1
                imgs.append(img)
                timestamps_cam.append(timestamp * 100_000) # convert camera timestamp to ns.
                timestamps_python.append(time.perf_counter_ns()) # convert python timestamp to ns.
                if frame_count % 10 == 0:
                    print(f'Frame: {frame_count}, timestamp:{timestamps_cam[-1]}, fps: {10 * 1_000_000_000 / (timestamps_cam[-1] - last_time):.2f}')
                    last_time = timestamp * 100_000
            else:
                print('NullFrame')
        except Exception as e:
            print(f"Error grabbing frame: {e}")
            break
finally:
    # 4. Shutdown
    print("\nStopping timer...")
    timer.stop()
    timer.join()
    mycam.close()

    # 5. Performance Analysis
    if len(timestamps_cam) > 1:
        intervals_cam = np.diff(timestamps_cam) / 1_000 # in microsec.
        target_interval_us = interval_s * 1_000_000
        errors_cam = intervals_cam - target_interval_us
        
        print("\n--- Cython Direct C-Call Timer Performance (Camera Timestamp) ---")
        print(f"  Target Interval: {target_interval_us:.2f} µs")
        print(f"  Mean Interval:   {np.mean(intervals_cam):.2f} µs")
        print(f"  Jitter (Std Dev):{np.std(intervals_cam):.2f} µs")
        print(f"  Max Error:       {np.max(errors_cam):.2f} µs")
        print(f"  Min Error:       {np.min(errors_cam):.2f} µs")
        print('\n--- Cython Direct C-Call Timer Performance (Python grab Timestamp) ---')
        intervals_python = np.diff(timestamps_python) / 1_000 # in microsec.
        errors_python = intervals_python - target_interval_us
        print(f"  Target Interval: {target_interval_us:.2f} µs")
        print(f"  Mean Interval:   {np.mean(intervals_python):.2f} µs")
        print(f"  Jitter (Std Dev):{np.std(intervals_python):.2f} µs")
        print(f"  Max Error:       {np.max(errors_python):.2f} µs")
        print(f"  Min Error:       {np.min(errors_python):.2f} µs")

    print(f"\nCaptured {len(imgs)} frames. Program finished.")

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.hist(intervals_cam, bins=100, density=True)
    plt.xlabel('Interval (µs)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Frame Intervals (Camera Timestamp)')
    plt.subplot(2, 2, 2)
    plt.plot(intervals_cam)
    plt.xlabel('Frame Index')
    plt.ylabel('Interval (µs)')
    plt.title('Frame Intervals (Camera Timestamp)')
    plt.subplot(2, 2, 3)
    plt.hist(intervals_python, bins=100, density=True)
    plt.xlabel('Interval (µs)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Frame Intervals (Python grab Timestamp)')
    plt.subplot(2, 2, 4)
    plt.plot(intervals_python)
    plt.xlabel('Frame Index')
    plt.ylabel('Interval (µs)')
    plt.title('Frame Intervals (Python grab Timestamp)')

    plt.tight_layout()
    plt.show()
