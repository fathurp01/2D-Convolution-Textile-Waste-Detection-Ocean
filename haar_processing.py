import cv2
import os
import numpy as np
from image_processing import get_optimal_preprocessing, analyze_image_quality


def ensure_output_folder():
    os.makedirs("output", exist_ok=True)


def get_output_name(img_path, suffix="haar"):
    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    return os.path.join("output", f"{suffix}_{name}.jpg")


def validate_cascade(cascade_path):
    """Validate Haar cascade and provide helpful error messages"""
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError(
            f"Failed to load Haar Cascade. File may be corrupted: {cascade_path}"
        )

    return cascade


def calculate_detection_confidence(detections, img_shape):
    """Calculate confidence scores for detections"""
    confidences = []
    h, w = img_shape[:2]
    total_area = h * w

    for x, y, w_box, h_box in detections:
        # Size-based confidence
        box_area = w_box * h_box
        size_confidence = min(100, (box_area / total_area) * 1000)

        # Position-based confidence (center regions get higher scores)
        center_x, center_y = x + w_box // 2, y + h_box // 2
        distance_from_center = np.sqrt(
            (center_x - w // 2) ** 2 + (center_y - h // 2) ** 2
        )
        max_distance = np.sqrt((w // 2) ** 2 + (h // 2) ** 2)
        position_confidence = (1 - distance_from_center / max_distance) * 50

        # Aspect ratio confidence (more square objects get higher scores)
        aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
        aspect_confidence = max(0, 50 - (aspect_ratio - 1) * 10)

        total_confidence = (
            size_confidence + position_confidence + aspect_confidence
        ) / 3
        confidences.append(min(100, max(10, total_confidence)))

    return confidences


def non_maximum_suppression(detections, confidences, overlap_threshold=0.3):
    """Remove overlapping detections using NMS"""
    if len(detections) == 0:
        return [], []

    # Convert to format expected by cv2.dnn.NMSBoxes
    boxes = [[x, y, w, h] for x, y, w, h in detections]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, overlap_threshold)

    if len(indices) > 0:
        indices = indices.flatten()
        filtered_detections = [detections[i] for i in indices]
        filtered_confidences = [confidences[i] for i in indices]
        return filtered_detections, filtered_confidences

    return [], []


def multi_scale_detection(cascade, img_gray, analysis):
    """Perform detection with multiple parameter sets"""
    all_detections = []
    all_confidences = []

    # Parameter sets based on image characteristics
    if analysis["is_dark"] or analysis["is_low_contrast"]:
        # More sensitive parameters for difficult images
        param_sets = [
            {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (15, 15)},
            {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (20, 20)},
            {"scaleFactor": 1.15, "minNeighbors": 4, "minSize": (25, 25)},
        ]
    else:
        # Standard parameters for good quality images
        param_sets = [
            {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (30, 30)},
            {"scaleFactor": 1.05, "minNeighbors": 4, "minSize": (25, 25)},
            {"scaleFactor": 1.2, "minNeighbors": 6, "minSize": (35, 35)},
        ]

    for i, params in enumerate(param_sets):
        print(f"   Trying parameter set {i + 1}: {params}")

        detections = cascade.detectMultiScale(
            img_gray,
            scaleFactor=params["scaleFactor"],
            minNeighbors=params["minNeighbors"],
            minSize=params["minSize"],
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(detections) > 0:
            confidences = calculate_detection_confidence(detections, img_gray.shape)
            all_detections.extend(detections)
            all_confidences.extend(confidences)
            print(f"   Found {len(detections)} detections")

    return all_detections, all_confidences


def draw_enhanced_detections(img, detections, confidences):
    """Draw detections with enhanced visualization"""
    colors = [
        (0, 255, 0),  # Green - High confidence
        (0, 255, 255),  # Yellow - Medium confidence
        (0, 165, 255),  # Orange - Low confidence
    ]

    for i, ((x, y, w, h), confidence) in enumerate(zip(detections, confidences)):
        # Choose color based on confidence
        if confidence > 70:
            color = colors[0]
            thickness = 3
        elif confidence > 40:
            color = colors[1]
            thickness = 2
        else:
            color = colors[2]
            thickness = 2

        # Draw main rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        # Draw confidence text with background
        conf_text = f"Sampah Tekstil {confidence:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(
            conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Background rectangle for text
        cv2.rectangle(img, (x, y - text_h - 15), (x + text_w + 10, y), color, -1)
        cv2.putText(
            img, conf_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        # Draw confidence bar
        bar_width = int(w * confidence / 100)
        cv2.rectangle(img, (x, y + h + 5), (x + bar_width, y + h + 15), color, -1)
        cv2.rectangle(img, (x, y + h + 5), (x + w, y + h + 15), color, 1)

        # Add detection number
        cv2.circle(img, (x + w - 15, y + 15), 12, color, -1)
        cv2.putText(
            img,
            str(i + 1),
            (x + w - 20, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )


def detect_sampah(
    img_path, cascade_path="haarcascade_sampah/cascade.xml", use_preprocessing=True
):
    """Enhanced sampah detection with integrated preprocessing"""
    ensure_output_folder()

    print(f"\n🔍 Starting textile waste detection...")
    print(f"   Image: {os.path.basename(img_path)}")
    print(f"   Cascade: {cascade_path}")
    print(f"   Use Preprocessing: {use_preprocessing}")

    # Validate and load cascade
    cascade = validate_cascade(cascade_path)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Cannot load image from {img_path}")

    img_original = img.copy()
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Choose preprocessing approach
    if use_preprocessing:
        # Use optimal preprocessing from image_processing module
        analysis = analyze_image_quality(gray_original)
        processed_gray = get_optimal_preprocessing(gray_original, analysis)
        print(f"   Using adaptive preprocessing based on image analysis")
    else:
        # Use original grayscale
        processed_gray = gray_original
        analysis = analyze_image_quality(gray_original)
        print(f"   Using original grayscale image")

    # Save preprocessed image
    cv2.imwrite(get_output_name(img_path, "preprocessed_for_detection"), processed_gray)

    # Multi-scale detection
    print(f"   Performing multi-scale detection...")
    all_detections, all_confidences = multi_scale_detection(
        cascade, processed_gray, analysis
    )

    # Apply Non-Maximum Suppression to remove overlaps
    if len(all_detections) > 0:
        print(f"   Applying Non-Maximum Suppression...")
        detections, confidences = non_maximum_suppression(
            all_detections, all_confidences
        )
        print(f"   After NMS: {len(detections)} detections remaining")
    else:
        detections, confidences = [], []

    # Draw results
    result_img = img_original.copy()
    if len(detections) > 0:
        draw_enhanced_detections(result_img, detections, confidences)

        # Print detection summary
        avg_confidence = np.mean(confidences)
        print(f"\n✅ Detection Results:")
        print(f"   Total Objects Found: {len(detections)}")
        print(f"   Average Confidence: {avg_confidence:.1f}%")
        for i, conf in enumerate(confidences):
            print(f"   Object {i + 1}: {conf:.1f}% confidence")
    else:
        print(f"\n❌ No textile waste detected")

    # Create comparison image
    if use_preprocessing:
        # Show original, preprocessed, and result
        comparison_imgs = [
            cv2.resize(img_original, (300, 300)),
            cv2.cvtColor(cv2.resize(processed_gray, (300, 300)), cv2.COLOR_GRAY2BGR),
            cv2.resize(result_img, (300, 300)),
        ]
        comparison = np.hstack(comparison_imgs)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (50, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(
            comparison, "Preprocessed", (350, 30), font, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            comparison, "Detection Result", (650, 30), font, 0.7, (255, 255, 255), 2
        )
    else:
        comparison = np.hstack(
            [cv2.resize(img_original, (400, 400)), cv2.resize(result_img, (400, 400))]
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (100, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(
            comparison, "Detection Result", (500, 30), font, 0.8, (255, 255, 255), 2
        )

    # Save results
    output_path = get_output_name(img_path, "detection_result")
    comparison_path = get_output_name(img_path, "comparison")

    cv2.imwrite(output_path, result_img)
    cv2.imwrite(comparison_path, comparison)

    print(f"   Results saved to: {output_path}")
    print(f"   Comparison saved to: {comparison_path}")

    # Display results
    cv2.imshow("Textile Waste Detection Results", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(detections), confidences
