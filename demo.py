import mvsdk
import numpy as np
from huateng_camera_v2_tc_mod import Camera

import matplotlib.pyplot as plt
import cv2
from raw_processing import *


DevList = mvsdk.CameraEnumerateDevice()
mycam = Camera(DevList[0], 9, gain=1, hibitdepth=1)

mycam.open()
raw_img = mycam.grab_raw()
# img最大值最小值
print(f'RAW Grabbed: img.max(): {raw_img.max()}, img.min(): {raw_img.min()}')

# Demo, 单纯先跑流程，不封装
# 定义Bayer模式
BAYER_PATTERN = 'BGGR' # 'RGGB' or 'BGGR'

# 扣黑位BLC
img = raw_img - 32

# print(img[920:960, 1460:1500])
# plt.imshow(img)
# plt.show()

# 白平衡
# 定义两个ROI用于自动白平衡
# ROI 1 (亮区)
x_slice_1 = slice(1460, 1500)
y_slice_1 = slice(920, 960)
awb_roi_1 = (y_slice_1, x_slice_1) # y,x

# ROI 2 (暗区)
x_slice_2 = slice(1460, 1500)
y_slice_2 = slice(1120, 1160)
awb_roi_2 = (y_slice_2, x_slice_2) # y,x

# 调用raw_awb进行白平衡，使用单点或两点
# 单点校准 (正常情况下用这个)
# img, awb_params = raw_awb(img, awb_roi_1, pattern=BAYER_PATTERN, clip_max_level=4095-32)

# 两点校准
img, awb_params = raw_awb(img, awb_roi_1, awb_roi_2, pattern=BAYER_PATTERN, clip_max_level=4095-32)


print(f'After AWB: img.max(): {img.max()}, img.min(): {img.min()}')

# plt.imshow(img)
# plt.show()

# 简单的Debayer
if BAYER_PATTERN == 'RGGB':
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_RGGB2RGB)
elif BAYER_PATTERN == 'BGGR':
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BGGR2RGB)

print(f'After Debayer: img.max(): {img.max()}, img.min(): {img.min()}')

# 保存成Tiff
cv2.imwrite('demo.tiff', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# # 保存成RAW
# import subprocess

# # 1. 保存 Bayer 数据为 .pgm
# height, width = raw_img.shape
# with open("temp.pgm", "wb") as f:
#     f.write(f"P5\n{width} {height}\n65535\n".encode())  # PGM 头（16-bit）
#     f.write(raw_img.tobytes())

# # 2. 使用 dcraw 转换为 DNG（假设 Bayer 模式是 RGGB）
# subprocess.run([
#     "dcraw",
#     "-v",  # verbose
#     "-6",  # 16-bit
#     "-T",  # 输出 TIFF（可用于进一步转 DNG）
#     "-o", "0",  # 原始颜色（不自动白平衡）
#     "-4",  # 线性 RAW（不 gamma 校正）
#     "-r", "1 1 1 1",  # 白平衡系数（默认 1:1:1:1）
#     "temp.pgm"
# ])


# 归一化为了显示
img = img.astype(np.float32) / 65535

plt.imshow(img)
# plt.xlim(-1,200)
# plt.ylim(200,-1)
plt.title("White-balanced, in device RGB space (No transformation)")
plt.show()

mycam.close()

# Segment and extract measurement of 24 patches

# Define 4 corners of the chart in the original image
# Note: The order of corners does not matter for `correct_perspective`
chart_corners = np.array([
    [1134, 702],  # Top-left
    [1528, 702],  # Top-right
    [1134, 1294], # Bottom-left
    [1528, 1294]  # Bottom-right
])

# --- Scheme 1: Correcting the perspective of original image ---

from correction_utils import correct_perspective, calculate_bboxes, extract_colors_bbox, draw_bboxes_matplotlib

# 1. Correct the perspective
# The image `img` is float32 and normalized, which is fine for transformation
corrected_chart = correct_perspective(img, chart_corners)

# 2. Calculate the 24 patch bounding boxes
# Using a 20% margin to avoid patch borders.
# Since orientation is now locked to landscape, we use 4 rows and 6 columns.
bboxes = calculate_bboxes(corrected_chart, rows=4, cols=6, margin_percent=0.2)

# 3. Extract the mean color of each patch
mean_rgb_values = extract_colors_bbox(corrected_chart, bboxes)

# 4. Print the results
print("\n--- Color Patch Extraction Results ---")
for i, rgb in enumerate(mean_rgb_values):
    # Since the image was normalized to 1.0, the values are float.
    # We can scale them to 0-255 for a more familiar representation.
    rgb_255 = tuple(f'{c * 255:.2f}' for c in rgb)
    print(f"Patch {i+1:2d}: Mean RGB = {rgb_255}")
print("------------------------------------")


# 5. Visualize the results
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Display the corrected chart on the first subplot
ax1.imshow(corrected_chart)
ax1.set_title("Perspective Corrected Chart")
ax1.axis('off')

# Display the corrected chart on the second subplot
ax2.imshow(corrected_chart)
# Draw high-resolution bboxes using matplotlib
draw_bboxes_matplotlib(ax2, bboxes)
ax2.set_title("Corrected Chart with BBoxes (Matplotlib)")
ax2.axis('off')

plt.tight_layout()
plt.show()


# --- Scheme 2: Extracting from warped bboxes on original image ---

from correction_utils import calculate_warped_bboxes, extract_colors_warped_bbox, draw_warped_bboxes
print("\n--- Scheme 2: Warped BBox Extraction ---")

# 1. Calculate the warped bboxes on the original image
# Since orientation is now locked to landscape, we use 4 rows and 6 columns.
warped_bboxes = calculate_warped_bboxes(chart_corners, rows=4, cols=6, margin_percent=0.2)

# 2. Extract mean colors directly from the original image `img`
mean_rgb_values_scheme2 = extract_colors_warped_bbox(img, warped_bboxes)

# 3. Print the results for Scheme 2
for i, rgb in enumerate(mean_rgb_values_scheme2):
    rgb_255 = tuple(f'{c * 255:.2f}' for c in rgb)
    print(f"Patch {i+1:2d}: Mean RGB = {rgb_255}")
print("------------------------------------")

# 4. Visualize the results of Scheme 2
# Draw the warped bboxes on the original, uncorrected image
img_with_warped_bboxes = draw_warped_bboxes(img, warped_bboxes)

plt.figure(figsize=(10, 8))
plt.imshow(img_with_warped_bboxes)
plt.title("Scheme 2: Warped BBoxes on Original Image")
plt.axis('off')
plt.show()
