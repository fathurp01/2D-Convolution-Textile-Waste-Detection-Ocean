import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def ensure_output_folder():
    os.makedirs("output", exist_ok=True)


def get_output_name(img_path, suffix):
    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    return os.path.join("output", f"{suffix}_{name}.jpg")


def analyze_image_quality(img_gray):
    """Analisis kualitas gambar untuk menentukan preprocessing yang optimal"""
    # Brightness analysis
    brightness = np.mean(img_gray)

    # Contrast analysis
    contrast = img_gray.std()

    # Noise analysis (using Laplacian variance)
    noise_level = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # Blur analysis
    blur_level = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    return {
        "brightness": brightness,
        "contrast": contrast,
        "noise_level": noise_level,
        "blur_level": blur_level,
        "is_dark": brightness < 100,
        "is_low_contrast": contrast < 30,
        "is_noisy": noise_level > 500,
        "is_blurry": blur_level < 100,
    }


def grayscale(img):
    """Convert to grayscale with weighted RGB conversion for better results"""
    if len(img.shape) == 3:
        # Weighted conversion (more accurate than simple average)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    return gray


def adaptive_hist_eq(img_gray, analysis):
    """Adaptive histogram equalization based on image analysis"""
    if analysis["is_dark"] or analysis["is_low_contrast"]:
        # Use CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)
    else:
        # Standard histogram equalization
        return cv2.equalizeHist(img_gray)


def adaptive_filtering(img_gray, analysis):
    """Apply adaptive filtering based on image characteristics"""
    if analysis["is_noisy"]:
        # Bilateral filter for noise reduction while preserving edges
        return cv2.bilateralFilter(img_gray, 9, 75, 75)
    else:
        # Gentle Gaussian blur
        return cv2.GaussianBlur(img_gray, (3, 3), 0)


def adaptive_sharpening(img_gray, analysis):
    """Apply sharpening only when needed"""
    if analysis["is_blurry"]:
        # Strong sharpening for blurry images
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img_gray, -1, kernel)
    else:
        # Mild sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img_gray, -1, kernel)


def enhanced_edge_detection(img_gray, analysis):
    """Enhanced edge detection with adaptive parameters"""
    if analysis["is_low_contrast"]:
        # More sensitive edge detection for low contrast images
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    else:
        # Standard Sobel
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    grad = cv2.magnitude(grad_x, grad_y)
    return np.uint8(np.clip(grad, 0, 255))


def adaptive_threshold(img_gray, analysis):
    """Adaptive thresholding based on image characteristics"""
    if analysis["is_dark"] or analysis["is_low_contrast"]:
        # Adaptive threshold for challenging lighting
        return cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Standard threshold
        _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th


def get_optimal_preprocessing(img_gray, analysis):
    """Return the best preprocessing result for Haar detection"""
    # Apply all preprocessing steps
    hist_eq = adaptive_hist_eq(img_gray, analysis)
    filtered = adaptive_filtering(hist_eq, analysis)

    # Mild sharpening for detection
    if analysis["is_blurry"]:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        optimal = cv2.filter2D(filtered, -1, kernel)
    else:
        optimal = filtered

    return optimal


def show_and_save_all_processes(img_path):
    """Enhanced processing with adaptive methods and optimal output for Haar detection"""
    ensure_output_folder()

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image from {img_path}")

    # Convert to grayscale
    gray = grayscale(img)

    # Analyze image quality
    analysis = analyze_image_quality(gray)

    # Print analysis results
    print(f"\n📊 Image Analysis Results:")
    print(f"   Brightness: {analysis['brightness']:.1f}")
    print(f"   Contrast: {analysis['contrast']:.1f}")
    print(f"   Noise Level: {analysis['noise_level']:.1f}")
    print(f"   Blur Level: {analysis['blur_level']:.1f}")
    print(
        f"   Characteristics: {'Dark' if analysis['is_dark'] else 'Bright'}, "
        f"{'Low Contrast' if analysis['is_low_contrast'] else 'Good Contrast'}, "
        f"{'Noisy' if analysis['is_noisy'] else 'Clean'}, "
        f"{'Blurry' if analysis['is_blurry'] else 'Sharp'}"
    )

    # Apply adaptive preprocessing
    hist_eq = adaptive_hist_eq(gray, analysis)
    filtered = adaptive_filtering(gray, analysis)
    sharpened = adaptive_sharpening(gray, analysis)
    edges = enhanced_edge_detection(gray, analysis)
    threshold = adaptive_threshold(gray, analysis)

    # Get optimal preprocessing for Haar detection
    optimal_for_haar = get_optimal_preprocessing(gray, analysis)

    # Save all outputs
    cv2.imwrite(get_output_name(img_path, "grayscale"), gray)
    cv2.imwrite(get_output_name(img_path, "hist_eq_adaptive"), hist_eq)
    cv2.imwrite(get_output_name(img_path, "filtered_adaptive"), filtered)
    cv2.imwrite(get_output_name(img_path, "sharpened_adaptive"), sharpened)
    cv2.imwrite(get_output_name(img_path, "edges_enhanced"), edges)
    cv2.imwrite(get_output_name(img_path, "threshold_adaptive"), threshold)
    cv2.imwrite(get_output_name(img_path, "optimal_for_haar"), optimal_for_haar)

    # Create comparison visualization
    titles = [
        "Original",
        "Grayscale",
        "Adaptive Hist Eq",
        "Adaptive Filter",
        "Adaptive Sharpen",
        "Enhanced Edges",
        "Adaptive Threshold",
        "Optimal for Haar",
    ]

    images = [
        img,
        gray,
        hist_eq,
        filtered,
        sharpened,
        edges,
        threshold,
        optimal_for_haar,
    ]

    # Create subplot with better layout
    plt.figure(figsize=(16, 10))
    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)

        if len(images[i].shape) == 3:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(images[i], cmap="gray")

        plt.title(titles[i], fontsize=10, fontweight="bold")
        plt.axis("off")

    plt.suptitle(
        f"Adaptive Image Processing Results\n"
        f"Brightness: {analysis['brightness']:.0f} | "
        f"Contrast: {analysis['contrast']:.0f} | "
        f"Noise: {analysis['noise_level']:.0f} | "
        f"Blur: {analysis['blur_level']:.0f}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return optimal_for_haar, analysis


# Legacy function untuk backward compatibility
def hist_eq(img_gray):
    return cv2.equalizeHist(img_gray)


def mean_filter(img_gray):
    return cv2.blur(img_gray, (3, 3))


def sharpen_filter(img_gray):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img_gray, -1, kernel)


def sobel_edge(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    return np.uint8(np.clip(grad, 0, 255))


def threshold(img_gray):
    _, th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    return th
