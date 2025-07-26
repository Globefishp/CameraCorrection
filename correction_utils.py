# correction_utils.py
# Providing useful functions during the process of color correction. 
# See Correction.ipynb for details.
# Author: Haiyun Huang & Deepseek V3 & Gemini 2.5 Pro
import io
import cv2 # for perspective correction
import numpy as np
from matplotlib import pyplot as plt
from functools import wraps
from contextlib import redirect_stdout
from scipy.optimize import minimize
import colour

def suppress_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 使用null设备作为输出目标
        with io.StringIO() as fake_stdout:
            with redirect_stdout(fake_stdout):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 可以在这里添加自定义的异常处理逻辑
                    raise e
    return wrapper

# --- Scheme 1: Correcting the perspective of original image ---

def correct_perspective(image, corners):
    """
    Corrects the perspective of a quadrilateral region in an image to a rectangle,
    ensuring the output is always landscape (width > height).
    Make sure the chart on the original image has *longer* pixel width than height, or it
    will produce wrong result.
    This function automatically orders the corners.

    Args:
        image: The source image.
        corners: A numpy array of 4 corner points of the quadrilateral.

    Returns:
        The perspective-corrected rectangular image.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)] # Top-left
    rect[2] = corners[np.argmax(s)] # Bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)] # Top-right
    rect[3] = corners[np.argmax(diff)] # Bottom-left

    (tl, tr, br, bl) = rect

    # Calculate the width and height of the bounding box
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    # The output width should be the long side, height the short side
    outWidth = max(int(widthA), int(widthB), int(heightA), int(heightB))
    outHeight = min(int(widthA), int(widthB), int(heightA), int(heightB))

    # If original was portrait, we need to rotate the corners for the transform
    if max(int(heightA), int(heightB)) > max(int(widthA), int(widthB)):
        # Original was portrait, rotate source rect points
        src_rect = np.array([bl, tl, tr, br], dtype="float32")
    else:
        # Original was landscape
        src_rect = rect

    dst = np.array([
        [0, 0],
        [outWidth - 1, 0],
        [outWidth - 1, outHeight - 1],
        [0, outHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_rect, dst)
    warped = cv2.warpPerspective(image, M, (outWidth, outHeight))
    return warped

def calculate_bboxes(chart_image, rows=4, cols=6, margin_percent=0.2):
    """
    Calculates the square bounding boxes for each patch on a corrected color chart image.

    Args:
        chart_image: The perspective-corrected image of the color chart.
        rows: The number of rows of patches.
        cols: The number of columns of patches.
        margin_percent: The margin to leave around each patch to avoid edges.

    Returns:
        A list of square bounding boxes (x, y, w, h) for each patch.
    """
    height, width, _ = chart_image.shape
    patch_width = width / cols
    patch_height = height / rows
    
    bboxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * patch_width
            y1 = r * patch_height
            
            margin_x = patch_width * margin_percent
            margin_y = patch_height * margin_percent
            
            x_inner = x1 + margin_x
            y_inner = y1 + margin_y
            w_inner = patch_width - (2 * margin_x)
            h_inner = patch_height - (2 * margin_y)
            
            size = min(w_inner, h_inner)
            
            x_center = x_inner + (w_inner - size) / 2
            y_center = y_inner + (h_inner - size) / 2

            bboxes.append((int(x_center), int(y_center), int(size), int(size)))
            
    return bboxes

def extract_colors_bbox(chart_image, bboxes):
    """
    Extracts the mean RGB color for each bounding box from the chart image.

    Args:
        chart_image: The perspective-corrected image of the color chart.
        bboxes: A list of bounding boxes (x, y, w, h).

    Returns:
        A list of mean RGB tuples.
    """
    mean_colors = []
    # Note: OpenCV uses BGR color order. The input image `chart_image` is assumed
    # to be in RGB format from the previous processing steps.
    # We will calculate the mean and keep it in RGB.
    for (x, y, w, h) in bboxes:
        patch = chart_image[y:y+h, x:x+w]
        # Calculate mean over R, G, B channels
        mean_rgb = np.mean(patch, axis=(0, 1))
        mean_colors.append(tuple(mean_rgb))
    return np.array(mean_colors)

def draw_bboxes_matplotlib(ax, bboxes):
    """
    Draws bounding boxes and labels on a matplotlib axes object.

    Args:
        ax: The matplotlib axes to draw on.
        bboxes: A list of bounding boxes (x, y, w, h).
    """
    for i, (x, y, w, h) in enumerate(bboxes):
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 10, str(i + 1), color='lime', fontsize=12, weight='bold')

# --- Scheme 2: Extracting from warped bboxes on original image ---

def get_perspective_transform_matrices(corners):
    """
    Calculates the perspective transform matrix and its inverse, ensuring
    the target rectangle is always landscape (width > height).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    side1 = max(int(widthA), int(widthB))
    side2 = max(int(heightA), int(heightB))

    outWidth = max(side1, side2)
    outHeight = min(side1, side2)

    # If original was portrait, we need to rotate the corners for the transform
    if side2 > side1:
        # Original was portrait, rotate source rect points
        src_rect = np.array([bl, tl, tr, br], dtype="float32")
    else:
        # Original was landscape
        src_rect = rect

    dst_rect = np.array([
        [0, 0],
        [outWidth - 1, 0],
        [outWidth - 1, outHeight - 1],
        [0, outHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    M_inv = cv2.getPerspectiveTransform(dst_rect, src_rect) # Inverse transform
    
    return M, M_inv, outWidth, outHeight

def calculate_warped_bboxes(corners, rows=4, cols=6, margin_percent=0.2):
    """
    Calculates the warped quadrilaterals on the original image by inverse-transforming
    ideal bboxes from the corrected space.
    """
    _, M_inv, maxWidth, maxHeight = get_perspective_transform_matrices(corners)
    
    # Create a dummy chart image to calculate ideal bboxes
    dummy_chart = np.zeros((maxHeight, maxWidth, 3))
    ideal_bboxes = calculate_bboxes(dummy_chart, rows, cols, margin_percent)
    
    warped_quads = []
    for (x, y, w, h) in ideal_bboxes:
        # Get the 4 corners of the ideal bbox
        bbox_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype="float32")
        
        # Reshape for cv2.perspectiveTransform: (1, N, 2)
        bbox_corners = bbox_corners.reshape(1, -1, 2)
        
        # Apply inverse perspective transform
        warped_corners = cv2.perspectiveTransform(bbox_corners, M_inv)
        warped_quads.append(warped_corners.reshape(4, 2).astype(np.int32))
        
    return warped_quads

def extract_colors_warped_bbox(image, warped_quads):
    """
    Extracts mean colors from the original image using warped quadrilateral masks.
    """
    mean_colors = []
    for quad in warped_quads:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [quad], 255)
        
        # Extract pixels using the mask
        # Note: For float images, boolean indexing is more direct
        pixels = image[mask == 255]
        
        # Calculate mean color
        mean_rgb = np.mean(pixels, axis=0)
        mean_colors.append(tuple(mean_rgb))
        
    return np.array(mean_colors)

def draw_warped_bboxes(image, warped_quads):
    """
    Draws warped quadrilaterals on an image for visualization.
    """
    img_with_quads = image.copy()
    # cv2.polylines requires a list of arrays
    pts = [q.reshape((-1, 1, 2)) for q in warped_quads]
    color = (0, 1.0, 0) if image.dtype == np.float32 else (0, 255, 0)
    cv2.polylines(img_with_quads, pts, isClosed=True, color=color, thickness=2)
    
    for i, quad in enumerate(warped_quads):
        # Put text at the first corner of the quadrilateral
        pos = (quad[0][0], quad[0][1] - 10)
        cv2.putText(img_with_quads, str(i + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    return img_with_quads

def linear_to_srgb(c_linear):
    '''
    Apply sRGB gamma curve on linear data, assuming input has cliped to [0, 1]
    '''
    c_srgb = np.where(c_linear <= 0.0031308, 12.92 * c_linear, 1.055 * (c_linear ** (1/2.4)) - 0.055)
    return c_srgb 

def forward_matrix_slover(src, target_xyz, 
                          illuminant='D50', 
                          extra_illuminant_transform_XYZ=np.eye(3),
                          initial_matrix=None, 
                          verbose=True):
    '''
    Calculate a forward matrix to transform source linear RGB values to CIE XYZ
    by minimizing the Delta E 2000 color difference.

    Args:
        src (np.ndarray): Source colors in a linear RGB space, shape (N, 3).
        target_xyz (np.ndarray): Target colors in CIE XYZ space, shape (N, 3).
        illuminant (str, optional): The name of the illuminant to use for XYZ-Lab 
            conversion. Defaults to 'D50'. Options: `D65` and `D50`.  
        extra_illuminant_transform_XYZ (np.ndarray, optional): A 3x3 matrix to transform 
            from src XYZ to target illuminant in XYZ space. Defaults to identity matrix.
            This let you control chromatic adaptation method. `I` matrix is the same with
            `implicitly identical` method applied in XYZ -> Lab conversion.
        initial_matrix (np.ndarray, optional): A 3x3 matrix to use as the
            initial guess. If None, an identity matrix is used. Defaults to None.
        verbose (bool): If True, prints optimization progress.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The optimized 3x3 forward matrix.
            - float: The final mean Delta E 2000 loss.
    '''
    # D50 illuminant is a common standard for colorimetric conversions.
    illuminant = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]

    # Convert the target XYZ colors to Lab* under D50 illuminant once.
    target_lab = colour.XYZ_to_Lab(target_xyz, illuminant=illuminant)

    # Keep track of iterations for the callback
    iteration_count = 0

    def loss_func(matrix_flat):
        """
        Loss function to be minimized. Calculates the mean squared Delta E 2000.
        """
        matrix = matrix_flat.reshape(3, 3)
        estimated_xyz = src @ matrix.T @ extra_illuminant_transform_XYZ.T
        estimated_lab = colour.XYZ_to_Lab(estimated_xyz, illuminant=illuminant)
        delta_E = colour.delta_E(estimated_lab, target_lab, method='CIE 2000')
        return np.mean(delta_E**3)

    def callback_func(xk):
        """
        Callback function to print progress during optimization.
        """
        nonlocal iteration_count
        iteration_count += 1
        if verbose and iteration_count % 100 == 0:
            loss = loss_func(xk)
            print(f"Iteration {iteration_count:4d}, Loss (dE2000**3): {loss:.4f}")

    # Set the initial guess for the matrix
    if initial_matrix is None:
        x0 = np.eye(3).flatten()
    else:
        x0 = initial_matrix.flatten()

    # Run the optimization
    result = minimize(
        loss_func,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6},
        callback=callback_func if verbose else None
    )

    optimized_matrix = result.x.reshape(3, 3)
    final_loss = result.fun
    
    if verbose:
        print(f"\nOptimization finished.")
        print(f"Final Loss (dE2000**3): {final_loss:.4f}")

    return optimized_matrix, final_loss
