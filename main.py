import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QCheckBox,
    QTextEdit,
    QProgressBar,
    QGroupBox,
    QGridLayout,
    QComboBox,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from image_processing import show_and_save_all_processes
from haar_processing import detect_sampah


class ProcessingThread(QThread):
    progress_update = pyqtSignal(str)
    finished_signal = pyqtSignal(int, list)  # detections count, confidences
    error_signal = pyqtSignal(str)

    def __init__(self, img_path, show_processing=True, use_preprocessing=True):
        super().__init__()
        self.img_path = img_path
        self.show_processing = show_processing
        self.use_preprocessing = use_preprocessing

    def run(self):
        try:
            if self.show_processing:
                self.progress_update.emit("🔄 Starting image preprocessing...")
                show_and_save_all_processes(self.img_path)
                self.progress_update.emit("✅ Preprocessing completed!")

            self.progress_update.emit("🔍 Starting textile waste detection...")
            detections_count, confidences = detect_sampah(
                self.img_path, use_preprocessing=self.use_preprocessing
            )

            if detections_count > 0:
                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                self.progress_update.emit(
                    f"✅ Detection completed! Found {detections_count} objects "
                    f"(avg confidence: {avg_conf:.1f}%)"
                )
            else:
                self.progress_update.emit("⚠️ No textile waste detected")

            self.finished_signal.emit(detections_count, confidences)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.error_signal.emit(error_msg)
            self.progress_update.emit(error_msg)


class AdvancedSettingsWidget(QWidget):
    """Advanced settings panel for detection parameters"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Detection Settings Group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QGridLayout()

        # Cascade file selection
        cascade_layout = QHBoxLayout()
        self.cascade_label = QLabel("Default cascade")
        self.cascade_btn = QPushButton("Browse Cascade")
        self.cascade_btn.clicked.connect(self.browse_cascade)
        cascade_layout.addWidget(QLabel("Cascade File:"))
        cascade_layout.addWidget(self.cascade_label)
        cascade_layout.addWidget(self.cascade_btn)

        detection_layout.addLayout(cascade_layout, 0, 0, 1, 2)
        detection_group.setLayout(detection_layout)

        # Processing Options Group
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()

        self.show_processing_cb = QCheckBox("Show all preprocessing steps")
        self.show_processing_cb.setChecked(True)

        self.use_preprocessing_cb = QCheckBox("Use adaptive preprocessing")
        self.use_preprocessing_cb.setChecked(True)

        self.save_intermediate_cb = QCheckBox("Save intermediate results")
        self.save_intermediate_cb.setChecked(True)

        options_layout.addWidget(self.show_processing_cb)
        options_layout.addWidget(self.use_preprocessing_cb)
        options_layout.addWidget(self.save_intermediate_cb)
        options_group.setLayout(options_layout)

        layout.addWidget(detection_group)
        layout.addWidget(options_group)
        layout.addStretch()

        self.setLayout(layout)
        self.cascade_path = "haarcascade_sampah/cascade.xml"

    def browse_cascade(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Haar Cascade File", "", "XML files (*.xml)"
        )
        if file_name:
            self.cascade_path = file_name
            self.cascade_label.setText(os.path.basename(file_name))


class ResultsWidget(QWidget):
    """Results display widget with detailed information"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Summary
        self.summary_label = QLabel("No results yet")
        self.summary_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #f0f0f0;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
        """)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(
            ["Object #", "Confidence", "Status"]
        )
        self.results_table.setAlternatingRowColors(True)

        # Open output folder button
        self.open_folder_btn = QPushButton("📁 Open Output Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)

        layout.addWidget(QLabel("Detection Summary:"))
        layout.addWidget(self.summary_label)
        layout.addWidget(QLabel("Detailed Results:"))
        layout.addWidget(self.results_table)
        layout.addWidget(self.open_folder_btn)

        self.setLayout(layout)

    def update_results(self, detections_count, confidences):
        if detections_count > 0:
            avg_conf = sum(confidences) / len(confidences)
            self.summary_label.setText(
                f"🎯 Found {detections_count} textile waste objects\n"
                f"Average confidence: {avg_conf:.1f}%"
            )
            self.summary_label.setStyleSheet("""
                QLabel {
                    padding: 15px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)

            # Update table
            self.results_table.setRowCount(len(confidences))
            for i, conf in enumerate(confidences):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"Object {i + 1}"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{conf:.1f}%"))

                # Status based on confidence
                if conf > 70:
                    status = "High Confidence"
                elif conf > 40:
                    status = "Medium Confidence"
                else:
                    status = "Low Confidence"

                self.results_table.setItem(i, 2, QTableWidgetItem(status))
        else:
            self.summary_label.setText("🔍 No textile waste objects detected")
            self.summary_label.setStyleSheet("""
                QLabel {
                    padding: 15px;
                    background-color: #fff3cd;
                    color: #856404;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            self.results_table.setRowCount(0)

        self.open_folder_btn.setEnabled(True)

    def open_output_folder(self):
        output_path = os.path.abspath("output")
        if os.path.exists(output_path):
            os.startfile(output_path)  # Windows
            # For Linux/Mac: os.system(f'xdg-open "{output_path}"')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧵 Advanced Textile Waste Detection System")
        self.setGeometry(200, 200, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)

        self.init_ui()
        self.img_path = ""
        self.processing_thread = None

    def init_ui(self):
        # Central widget with tabs
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("🧵 Advanced Textile Waste Detection System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #495057; margin: 10px;")

        # File selection group
        file_group = QGroupBox("Image Selection")
        file_layout = QVBoxLayout()

        file_select_layout = QHBoxLayout()
        self.file_label = QLabel("📁 No file selected")
        self.file_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                border: 2px dashed #dee2e6;
                border-radius: 8px;
                background-color: white;
            }
        """)

        self.btn_browse = QPushButton("📂 Browse Image")
        self.btn_browse.clicked.connect(self.browse_file)

        file_select_layout.addWidget(self.file_label, 3)
        file_select_layout.addWidget(self.btn_browse, 1)
        file_layout.addLayout(file_select_layout)
        file_group.setLayout(file_layout)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Processing tab
        processing_tab = QWidget()
        processing_layout = QVBoxLayout()

        # Quick settings
        quick_settings_group = QGroupBox("Quick Settings")
        quick_layout = QHBoxLayout()

        self.show_steps_cb = QCheckBox("Show preprocessing steps")
        self.show_steps_cb.setChecked(True)

        self.use_adaptive_cb = QCheckBox("Use adaptive preprocessing")
        self.use_adaptive_cb.setChecked(True)

        quick_layout.addWidget(self.show_steps_cb)
        quick_layout.addWidget(self.use_adaptive_cb)
        quick_layout.addStretch()
        quick_settings_group.setLayout(quick_layout)

        # Process button
        self.btn_process = QPushButton("🚀 Start Detection Process")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                font-size: 14px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #1e7e34;
            }
        """)

        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(200)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_area)
        progress_group.setLayout(progress_layout)

        processing_layout.addWidget(quick_settings_group)
        processing_layout.addWidget(self.btn_process)
        processing_layout.addWidget(progress_group)
        processing_layout.addStretch()
        processing_tab.setLayout(processing_layout)

        # Advanced settings tab
        self.advanced_settings = AdvancedSettingsWidget()

        # Results tab
        self.results_widget = ResultsWidget()

        # Add tabs
        self.tab_widget.addTab(processing_tab, "🔧 Processing")
        self.tab_widget.addTab(self.advanced_settings, "⚙️ Advanced")
        self.tab_widget.addTab(self.results_widget, "📊 Results")

        # Add everything to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(file_group)
        main_layout.addWidget(self.tab_widget)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
        )
        if file_name:
            self.img_path = file_name
            file_name_short = os.path.basename(file_name)
            self.file_label.setText(f"📁 Selected: {file_name_short}")
            self.file_label.setStyleSheet("""
                QLabel {
                    padding: 12px;
                    border: 2px solid #28a745;
                    border-radius: 8px;
                    background-color: #d4edda;
                    color: #155724;
                }
            """)
            self.btn_process.setEnabled(True)
            self.log_area.clear()
            self.log_area.append(f"✅ Image loaded: {file_name_short}")

            # Get image info
            try:
                import cv2

                img = cv2.imread(file_name)
                if img is not None:
                    h, w = img.shape[:2]
                    size_mb = os.path.getsize(file_name) / (1024 * 1024)
                    self.log_area.append(f"📐 Dimensions: {w}x{h} pixels")
                    self.log_area.append(f"💾 File size: {size_mb:.2f} MB")
            except:
                pass

    def start_processing(self):
        if not self.img_path:
            return

        # Switch to processing tab
        self.tab_widget.setCurrentIndex(0)

        # Update UI
        self.btn_process.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Clear previous results
        self.log_area.clear()

        # Start processing
        self.processing_thread = ProcessingThread(
            self.img_path,
            show_processing=self.show_steps_cb.isChecked(),
            use_preprocessing=self.use_adaptive_cb.isChecked(),
        )

        self.processing_thread.progress_update.connect(self.update_log)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.error_signal.connect(self.processing_error)
        self.processing_thread.start()

    def update_log(self, message):
        self.log_area.append(f"⏰ {message}")
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum()
        )

    def processing_finished(self, detections_count, confidences):
        # Update UI
        self.progress_bar.setVisible(False)
        self.btn_process.setEnabled(True)

        # Update results
        self.results_widget.update_results(detections_count, confidences)

        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)

        # Final log message
        self.log_area.append("🎉 Processing completed successfully!")
        self.log_area.append("📁 Check the 'output' folder for all generated files")

    def processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.btn_process.setEnabled(True)
        self.log_area.append(error_message)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
