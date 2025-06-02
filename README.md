---

# Textile Waste Detection System: Digital Image Processing with Haar Cascade

## 📋 Project Description

This project implements a textile waste detection system aimed at assisting the waste sorting process in marine environments. The system utilizes a traditional image processing approach, combining 2D convolution techniques with Haar Cascade Classifiers. Designed without reliance on deep learning, it offers a practical alternative for marine waste management.

### Background
Textile waste in marine environments poses a serious problem affecting coastal ecosystems and marine life. This system was developed to aid the automatic identification and sorting of textile waste using digital image processing technology.

## 🚀 Key Features

-   **Automatic Detection**: Detects textile waste in images using custom-trained Haar Cascades.
-   **Adaptive Preprocessing**: Automatic image quality analysis and adaptive preprocessing adjustments.
-   **Interactive GUI**: User-friendly graphical interface built with PyQt5.
-   **Multi-scale Detection**: Supports detection with various parameters for optimal accuracy.
-   **Result Visualization**: Displays detection results with confidence scores.
-   **Batch Processing**: Ability to process multiple images simultaneously.
-   **Export Results**: Option to save detection and preprocessing outcomes.

## 🛠️ Technologies Used

-   **Python 3.9+**
-   **OpenCV**: For image processing and object detection.
-   **PyQt5**: For the graphical user interface.
-   **NumPy**: For array operations and mathematical computations.
-   **Matplotlib**: For data visualization.

## 📁 Project Structure

```
Tubes/
├── main.py                     # Main GUI application file
├── image_processing.py         # Image preprocessing module
├── haar_processing.py          # Haar Cascade detection module
├── laporan.txt                 # Full project report (in Indonesian)
├── dataset/                    # Dataset for training
│   ├── positives/              # Positive sample images
│   ├── negatives/              # Negative sample images
│   ├── info.lst                # Positive sample information file
│   ├── bg.txt                  # Background image list file
│   └── note.txt                # Dataset notes
├── haarcascade_sampah/         # Trained Haar Cascade model
├── train_haar_cascade/         # Scripts for model training
│   ├── create_annotations.py   # Tool for creating annotations
│   ├── train_cascade.py        # Cascade training script
│   └── test_cascade.py         # Model testing script
├── output/                     # Preprocessing and detection results
├── samples/                    # Sample images for testing
└── test_images/                # Test images
```

## 🔧 Installation

### System Requirements
-   Python 3.9 or newer
-   OpenCV with Haar Cascade support
-   PyQt5
-   NumPy
-   Matplotlib

### Installation Steps

1.  **Clone or download this project**
    ```bash
    cd d:\Src\Latihan\PCD\Tubes # Adjust path as needed
    ```

2.  **Install dependencies**
    ```bash
    pip install opencv-python PyQt5 numpy matplotlib
    ```

3.  **Verify OpenCV installation**
    ```bash
    python -c "import cv2; print(cv2.__version__)"
    ```

## 🎯 How to Use

### Running the GUI Application

```bash
python main.py
```

### Application Features

1.  **Load Image**: Select image(s) for analysis.
2.  **Processing Options**:
    -   Show all preprocessing steps
    -   Use adaptive preprocessing
    -   Save intermediate results
3.  **Detection Settings**: Configure detection parameters.
4.  **Start Processing**: Initiate the detection process.
5.  **View Results**: Display detection results with confidence scores.

### Programmatic Usage

```python
from haar_processing import detect_sampah
from image_processing import show_and_save_all_processes

# Preprocess an image
show_and_save_all_processes('path/to/image.jpg')

# Detect textile waste
detections_count, confidences = detect_sampah(
    'path/to/image.jpg',
    use_preprocessing=True
)

print(f"Found {detections_count} textile waste objects")
```

## 🧠 Methodology

### 1. Adaptive Preprocessing
-   **Image Quality Analysis**: Brightness, contrast, noise, and blur analysis.
-   **Grayscale Conversion**: Weighted RGB to grayscale conversion.
-   **Histogram Equalization**: CLAHE for local contrast enhancement.
-   **Filtering**: Bilateral filter for noise reduction.
-   **Sharpening**: Adaptive sharpening based on blur level.
-   **Edge Detection**: Enhanced Sobel edge detection.

### 2. Haar Cascade Detection
-   **Multi-scale Detection**: Detection performed with various parameters.
-   **Confidence Calculation**: Confidence score based on size, position, and aspect ratio.
-   **Non-Maximum Suppression**: Elimination of overlapping detections.
-   **Enhanced Visualization**: Color-coded visualization of detection results.

### 3. Custom Cascade Training
-   **Dataset Preparation**: Positive and negative samples.
-   **Annotation Tool**: Custom tool for creating bounding box annotations.
-   **Training Process**: OpenCV cascade training with optimized parameters.

## 📊 Results and Evaluation

The system has been tested under various image conditions:
-   ✅ Images with normal lighting
-   ✅ Images with low lighting
-   ✅ Images with noise
-   ✅ Images with blur
-   ✅ Images with low contrast

### Evaluation Metrics
-   **Detection Count**: Number of objects detected.
-   **Confidence Score**: Detection confidence level (10-100%).
-   **Processing Time**: Time taken to process each image.
-   **Accuracy**: Detection accuracy based on ground truth.

## 🔄 Training a New Model

To train a new Haar Cascade model with your own dataset:

1.  **Prepare Dataset**
    ```bash
    # Place positive images in dataset/positives/
    # Place negative images in dataset/negatives/
    ```

2.  **Create Annotations**
    ```bash
    python train_haar_cascade/create_annotations.py
    ```

3.  **Train Cascade**
    ```bash
    python train_haar_cascade/train_cascade.py --num_stages 20
    ```

## 🐛 Troubleshooting

### Error: "Haar cascade file not found"
-   Ensure the `haarcascade_sampah/cascade.xml` file exists.
-   Alternatively, use the "Browse Cascade" option in the GUI to select another cascade file.

### Error: "Cannot load image"
-   Verify that the image format is supported (JPG, PNG, BMP).
-   Check if the image path is correct.

### Inaccurate detection
-   Try enabling "Use adaptive preprocessing".
-   Adjust detection parameters in "Advanced Settings".
-   Ensure the image quality is good.

## 📈 Future Development

-   [ ] Implement deep learning for higher accuracy.
-   [ ] Support for video processing.
-   [ ] Real-time detection using a webcam.
-   [ ] Mobile application implementation.
-   [ ] Cloud-based processing.
-   [ ] Integration with environmental monitoring systems.

## 📚 References

1.  Wikiandy, Rosidah, dan Titin Herawati (2013). "Dampak Pencemaran Limbah Industri Tekstil Terhadap Kerusakan Struktur Organ Ikan di DAS Citarum Bagian Hulu" (Impact of Textile Industry Waste Pollution on Fish Organ Structure Damage in the Upper Citarum River Basin)
2.  OpenCV Documentation - Haar Cascade Classifiers
3.  Digital Image Processing Techniques

## 👥 Contributors

This project was developed as a final assignment for the Digital Image Processing (PCD) course.

## 📄 License

This project was created by Fathurrahman Pratama Putra for academic and research purposes.

---
