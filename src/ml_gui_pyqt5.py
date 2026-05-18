from PyQt5.QtWidgets import (QApplication, QSizePolicy, QMainWindow, QAbstractItemView,
                              QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                              QPushButton, QLabel, QProgressBar, QCheckBox, QListWidget,
                              QComboBox, QFileDialog, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
from pandas import read_excel
from datetime import datetime
import numpy as np
import os
import sys
from PIL import Image

from processing import makeExcel, process_main
from image_analysis import imdecode, getPlainMean, Reagent, numpy_to_qt_image
from cv2 import cvtColor, COLOR_BGR2RGB
from model_def import ML_Model, DataAxis
from prediction import predict_value, load, download_predictions
from util import open_window, is_float, crop_image
from calibration import ImageMaskApp
from image_processor import ImageProcessor


dropdown_options = ["Auto Detect"] + [r.name for r in Reagent.reagents]

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
APP_STYLESHEET = """
QWidget {
    font-family: "Calibri", "Segoe UI", Arial, sans-serif;
    font-size: 12px;
    color: #1e293b;
    background-color: #f8fafc;
}
QMainWindow { background: #f8fafc; }

/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-top: none;
    background: white;
    border-radius: 0 4px 4px 4px;
}
QTabBar::tab {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    padding: 7px 20px;
    margin-right: 2px;
    border-radius: 4px 4px 0 0;
    color: #64748b;
    font-weight: 600;
}
QTabBar::tab:selected {
    background: white;
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}
QTabBar::tab:hover:!selected {
    background: #e2e8f0;
    color: #334155;
}

/* ── Inputs ── */
QLineEdit {
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 5px 10px;
    background: white;
    min-height: 30px;
    selection-background-color: #bfdbfe;
}
QLineEdit:focus  { border-color: #3b82f6; }
QLineEdit:disabled { background: #f1f5f9; color: #94a3b8; }

QComboBox {
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 5px 10px;
    background: white;
    min-height: 30px;
}
QComboBox:focus { border-color: #3b82f6; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: white;
    selection-background-color: #dbeafe;
}

/* ── Buttons ── */
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 7px 18px;
    font-weight: 600;
    min-height: 32px;
}
QPushButton:hover   { background-color: #1d4ed8; }
QPushButton:pressed { background-color: #1e40af; }
QPushButton:disabled { background-color: #93c5fd; color: #e0f2fe; }

/* ── Progress bar ── */
QProgressBar {
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    background: #f1f5f9;
    text-align: center;
    color: #334155;
    font-weight: 600;
    min-height: 18px;
    max-height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #60a5fa, stop:1 #2563eb);
    border-radius: 4px;
}

/* ── List widget ── */
QListWidget {
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    background: white;
    outline: none;
}
QListWidget::item { padding: 4px 8px; }
QListWidget::item:selected {
    background: #dbeafe;
    color: #1e40af;
    border-radius: 3px;
}
QListWidget::item:hover:!selected { background: #f8fafc; }

/* ── Checkbox ── */
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: white;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
}

/* ── Separator ── */
QFrame[frameShape="4"] { color: #e2e8f0; }

/* ── Status footer ── */
QLabel#statusLabel {
    font-size: 11px;
    padding: 5px 12px;
    border-top: 1px solid #e2e8f0;
    min-height: 26px;
    background: #f8fafc;
    color: #475569;
}
"""

# Reusable inline style for secondary (browse) buttons
_SECONDARY = "background-color: #64748b; padding: 7px 14px; min-width: 80px;"
_SECONDARY += " font-weight: 600;"


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.split(os.path.abspath(__file__))[0]
    return os.path.join(base_path, relative_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.X = DataAxis()
        self.Y = DataAxis()
        self.processor = ImageProcessor(on_complete=self._on_analysis_complete)

        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowTitle("ECL Predictive Analysis Interface")
        self.setWindowIcon(QIcon(resource_path("media/maxresdefault.ico")))
        self.setGeometry(100, 100, 900, 650)
        self.setMinimumSize(480, 360)
        self.setMaximumSize(2400, 1800)

        self.central_widget = QWidget(self)
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.main_font = QFont("Calibri", pointSize=12, weight=30)
        self.init_header()
        self.init_tabs()
        self.init_image_analysis_tab()
        self.init_data_analysis_tab()
        self.init_prediction_tab()
        self.init_about_us_tab()
        self.init_footer()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, message, level="info"):
        """Show a colour-coded message in the status bar."""
        palettes = {
            "info":    ("transparent", "#475569"),
            "success": ("#dcfce7",     "#166534"),
            "error":   ("#fee2e2",     "#991b1b"),
            "working": ("#dbeafe",     "#1e40af"),
            "warning": ("#fef9c3",     "#854d0e"),
        }
        bg, fg = palettes.get(level, palettes["info"])
        self.footer_label.setStyleSheet(
            f"QLabel#statusLabel {{ background:{bg}; color:{fg}; "
            f"padding:5px 12px; border-top:1px solid #e2e8f0; "
            f"font-size:11px; min-height:26px; }}"
        )
        self.footer_label.setText(message)

    @staticmethod
    def _make_separator():
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep

    @staticmethod
    def _secondary(btn, tooltip=""):
        btn.setStyleSheet(_SECONDARY)
        if tooltip:
            btn.setToolTip(tooltip)

    def _on_analysis_complete(self):
        self.perform_analysis_button.setText("Start Analysis")
        self.perform_analysis_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def init_header(self):
        header_widget = QWidget()
        header_widget.setStyleSheet(
            "QWidget { background: white; border-bottom: 2px solid #2563eb; }"
        )
        self.header_bits_image = QLabel()
        i1 = Image.open(resource_path("media/bits_logo.jpg"))
        i1 = i1.resize((46, 46 * i1.height // i1.width))
        self.header_bits_image.setPixmap(QPixmap(numpy_to_qt_image(np.array(i1), swapped=False)))

        self.header_label = QLabel("ECL Predictive Analysis Interface")
        self.header_label.setFont(QFont("Calibri", pointSize=22, weight=600))
        self.header_label.setStyleSheet("color: #1e293b; background: transparent;")
        self.header_label.setAlignment(Qt.AlignCenter)

        self.header_lab_image = QLabel()
        i2 = Image.open(resource_path("media/mmne.jpg"))
        i2 = i2.resize((46, 46 * i2.height // i2.width))
        self.header_lab_image.setPixmap(QPixmap(numpy_to_qt_image(np.array(i2), swapped=False)))
        self.header_lab_image.setAlignment(Qt.AlignRight)

        self.header_layout = QHBoxLayout(header_widget)
        self.header_layout.setContentsMargins(12, 8, 12, 8)
        self.header_layout.addWidget(self.header_bits_image)
        self.header_layout.addWidget(self.header_label, stretch=1)
        self.header_layout.addWidget(self.header_lab_image)
        self.main_layout.addWidget(header_widget)

    def init_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.image_analysis_tab = QWidget()
        self.data_analysis_tab = QWidget()
        self.prediction_tab = QWidget()
        self.calibration_tab = ImageMaskApp()
        self.about_us_tab = QWidget()

        try:
            from cameraApp import CameraApp
            self.real_time_tab = CameraApp()
            self.tab_widget.addTab(self.real_time_tab, "Real-Time")
        except Exception:
            pass
        self.tab_widget.addTab(self.image_analysis_tab, "Image Analysis")
        self.tab_widget.addTab(self.data_analysis_tab, "Data Analysis")
        self.tab_widget.addTab(self.prediction_tab, "Prediction")
        self.tab_widget.addTab(self.calibration_tab, "Calibrate")
        self.tab_widget.addTab(self.about_us_tab, "About")

    def init_image_analysis_tab(self):
        self.image_layout = QVBoxLayout()
        self.image_layout.setContentsMargins(16, 14, 16, 14)
        self.image_layout.setSpacing(10)

        # ── Input row ──
        self.image_folder_input = QLineEdit()
        self.image_folder_input.setPlaceholderText("Folder path (multiple) or image file path (single)…")
        self.browse_folder_btn = QPushButton("Browse")
        self.browse_folder_btn.clicked.connect(lambda: self.tab_one_browse_folder_or_image())
        self._secondary(self.browse_folder_btn, "Browse for an image folder or a single image/GIF file")
        self.multiple_or_single_image_dropdown = QComboBox()
        self.multiple_or_single_image_dropdown.addItems(["Multiple", "Single"])
        self.multiple_or_single_image_dropdown.setToolTip("Process a whole folder or a single image")
        self.hbox1 = QHBoxLayout()
        self.hbox1.setSpacing(8)
        self.hbox1.addWidget(self.image_folder_input)
        self.hbox1.addWidget(self.browse_folder_btn)
        self.hbox1.addWidget(self.multiple_or_single_image_dropdown)

        # ── Reagent row ──
        self.reagent_text_label = QLabel("Reagent:")
        self.reagent_dropdown = QComboBox()
        self.reagent_dropdown.addItems(dropdown_options)
        self.reagent_dropdown.setToolTip("Select the ECL reagent used, or let the app detect it automatically")
        self.detected_reagent_label = QLabel()
        self.detected_reagent_label.setVisible(False)
        self.choose_reagent_hbox = QHBoxLayout()
        self.choose_reagent_hbox.setSpacing(8)
        self.choose_reagent_hbox.addWidget(self.reagent_text_label)
        self.choose_reagent_hbox.addWidget(self.reagent_dropdown)
        self.choose_reagent_hbox.addWidget(self.detected_reagent_label)
        self.choose_reagent_hbox.addStretch()

        self.image_analysis_vbox1 = QVBoxLayout()
        self.image_analysis_vbox1.setSpacing(8)
        self.image_analysis_vbox1.addLayout(self.hbox1)
        self.image_analysis_vbox1.addLayout(self.choose_reagent_hbox)
        self.image_layout.addLayout(self.image_analysis_vbox1)
        self.image_layout.addWidget(self._make_separator())

        # ── Image preview + formula ──
        self.luminol_formula_img_label = QLabel()
        self.ecl_mechanism_image = Image.open(resource_path("media/luminol_formula-min.png"))
        screen_w = QApplication.primaryScreen().size().width()
        self.luminol_formula_img_label.setPixmap(
            QPixmap(resource_path("media/luminol_formula-min.png")).scaled(
                int(screen_w * 0.4),
                int(screen_w * 0.4) * self.ecl_mechanism_image.height // self.ecl_mechanism_image.width,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.luminol_formula_img_label.setAlignment(Qt.AlignCenter)

        self.image_analysis_formulas_vbox = QVBoxLayout()
        self.image_analysis_formulas_vbox.addWidget(self.luminol_formula_img_label)

        self.image_label = QLabel()
        self.image_label.setVisible(False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 1px solid #e2e8f0; border-radius: 6px; background: #f8fafc;"
        )

        self.dynamic_label = QLabel()
        self.dynamic_label.setVisible(False)
        self.dynamic_label.setAlignment(Qt.AlignCenter)
        self.dynamic_label.setStyleSheet("color: #2563eb; font-weight: 600;")

        self.image_placeholders_vbox = QVBoxLayout()
        self.image_placeholders_vbox.addWidget(self.image_label)
        self.image_placeholders_vbox.addWidget(self.dynamic_label)

        self.image_analysis_hbox2 = QHBoxLayout()
        self.image_analysis_hbox2.setSpacing(16)
        self.image_analysis_hbox2.addLayout(self.image_placeholders_vbox)
        self.image_analysis_hbox2.addLayout(self.image_analysis_formulas_vbox)
        self.image_layout.addLayout(self.image_analysis_hbox2)
        self.image_layout.addWidget(self._make_separator())

        # ── Progress ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("0%")
        self.progress_label.setVisible(False)
        self.progress_label.setFixedWidth(50)
        self.hbox2 = QHBoxLayout()
        self.hbox2.setSpacing(8)
        self.hbox2.addWidget(self.progress_bar)
        self.hbox2.addWidget(self.progress_label)
        self.image_layout.addLayout(self.hbox2)

        # ── Action buttons ──
        self.perform_analysis_button = QPushButton("▶  Start Analysis")
        self.perform_analysis_button.setToolTip("Begin image intensity extraction")
        self.pause_resume_analysis_button = QPushButton("⏸  Pause")
        self.pause_resume_analysis_button.clicked.connect(self.pause_resume_analysis)
        self.pause_resume_analysis_button.setVisible(False)
        self.pause_resume_analysis_button.setStyleSheet(
            "background-color: #f59e0b; color: white; font-weight: 600; "
            "border-radius: 5px; padding: 7px 18px; min-height: 32px;"
        )
        self.image_save_data_btn = QPushButton("⬇  Save Data to Excel")
        self.image_save_data_btn.clicked.connect(self.save_image_intensity_data)
        self.image_save_data_btn.setToolTip("Export intensity-concentration pairs to an Excel file")
        self.image_save_data_btn.setStyleSheet(
            "background-color: #16a34a; color: white; font-weight: 600; "
            "border-radius: 5px; padding: 7px 18px; min-height: 32px;"
        )
        self.image_analysis_hbox1 = QHBoxLayout()
        self.image_analysis_hbox1.setSpacing(10)
        self.image_analysis_hbox1.addWidget(self.perform_analysis_button)
        self.image_analysis_hbox1.addWidget(self.pause_resume_analysis_button)
        self.image_analysis_hbox1.addWidget(self.image_save_data_btn)
        self.image_analysis_hbox1.setAlignment(Qt.AlignCenter)
        self.image_layout.addLayout(self.image_analysis_hbox1)
        self.image_layout.addStretch()

        self.image_analysis_tab.setLayout(self.image_layout)

    def init_data_analysis_tab(self):
        self.data_layout = QVBoxLayout()
        self.data_layout.setContentsMargins(16, 14, 16, 14)
        self.data_layout.setSpacing(10)

        # ── File input ──
        self.data_analysis_file_input_bar = QLineEdit()
        self.data_analysis_file_input_bar.setPlaceholderText("Path to Excel file (.xlsx) with intensity-concentration data…")
        self.browse_file_data_analysis = QPushButton("Browse")
        self.browse_file_data_analysis.clicked.connect(lambda: self.browse(self.data_analysis_file_input_bar))
        self._secondary(self.browse_file_data_analysis, "Browse for the Excel data file")
        self.hbox3 = QHBoxLayout()
        self.hbox3.setSpacing(8)
        self.hbox3.addWidget(self.data_analysis_file_input_bar)
        self.hbox3.addWidget(self.browse_file_data_analysis)

        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.setToolTip("Load the selected Excel file and proceed to model selection")
        self.vbox3 = QVBoxLayout()
        self.vbox3.setSpacing(8)
        self.vbox3.addLayout(self.hbox3)
        self.vbox3.addWidget(self.load_data_button)
        self.data_layout.addLayout(self.vbox3)
        self.data_layout.addWidget(self._make_separator())

        # ── Model selection ──
        self.listbox = QListWidget()
        self.listbox.addItems([model.name for model in ML_Model.models])
        self.listbox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listbox.setToolTip("Select one or more regression models to train (Ctrl+click for multi-select)")
        self.listbox.setVisible(False)
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.clicked.connect(lambda: self.listbox.selectAll())
        self.select_all_checkbox.setVisible(False)
        self.hbox4 = QHBoxLayout()
        self.hbox4.setSpacing(8)
        self.hbox4.addWidget(self.listbox)
        self.hbox4.addWidget(self.select_all_checkbox, alignment=Qt.AlignTop)
        self.data_layout.addLayout(self.hbox4)

        self.set_models_btn = QPushButton("Confirm Model Selection")
        self.set_models_btn.clicked.connect(self.load_data_from_file)
        self.set_models_btn.setVisible(False)
        self.data_layout.addWidget(self.set_models_btn)

        # ── Axis selection ──
        self.x_var_label = QLabel("X  (independent variable):")
        self.x_var_dropdown = QComboBox()
        self.x_var_label.setVisible(False)
        self.x_var_dropdown.setVisible(False)
        self.hbox5 = QHBoxLayout()
        self.hbox5.setSpacing(8)
        self.hbox5.addWidget(self.x_var_label)
        self.hbox5.addWidget(self.x_var_dropdown)
        self.hbox5.addStretch()
        self.data_layout.addLayout(self.hbox5)

        self.y_var_label = QLabel("Y  (target / concentration):")
        self.y_var_dropdown = QComboBox()
        self.y_var_label.setVisible(False)
        self.y_var_dropdown.setVisible(False)
        self.hbox6 = QHBoxLayout()
        self.hbox6.setSpacing(8)
        self.hbox6.addWidget(self.y_var_label)
        self.hbox6.addWidget(self.y_var_dropdown)
        self.hbox6.addStretch()
        self.data_layout.addLayout(self.hbox6)

        self.set_labels_btn = QPushButton("Confirm Axis Labels")
        self.set_labels_btn.clicked.connect(self.ask_test_percentage)
        self.set_labels_btn.setVisible(False)
        self.data_layout.addWidget(self.set_labels_btn)

        # ── Test split ──
        self.test_percentage_label = QLabel("Test split % (20 recommended):")
        self.test_percentage_input = QLineEdit()
        self.test_percentage_input.setPlaceholderText("e.g. 20")
        self.test_percentage_input.setMaximumWidth(100)
        self.test_percentage_label.setVisible(False)
        self.test_percentage_input.setVisible(False)
        self.hbox7 = QHBoxLayout()
        self.hbox7.setSpacing(8)
        self.hbox7.addWidget(self.test_percentage_label)
        self.hbox7.addWidget(self.test_percentage_input)
        self.hbox7.addStretch()
        self.data_layout.addLayout(self.hbox7)

        self.set_test_btn = QPushButton("▶  Train Models")
        self.set_test_btn.setToolTip("Set the test split and start training all selected models")
        self.set_test_btn.setVisible(False)
        self.set_test_btn.clicked.connect(self.set_test_percentage_and_run)
        self.data_layout.addWidget(self.set_test_btn)

        self.reset_tab_btn = QPushButton("↺  Train New Data")
        self.reset_tab_btn.setVisible(False)
        self.reset_tab_btn.setStyleSheet(
            "background-color: #64748b; font-weight: 600; border-radius: 5px; "
            "padding: 7px 18px; min-height: 32px;"
        )
        self.reset_tab_btn.clicked.connect(lambda: self.reset_tab(self.data_layout))
        self.data_layout.addWidget(self.reset_tab_btn)
        self.data_layout.addStretch()

        self.data_analysis_tab.setLayout(self.data_layout)
        self.load_data_button.clicked.connect(self.load_listbox_bloc)

    def init_prediction_tab(self):
        self.prediction_layout = QVBoxLayout()
        self.prediction_layout.setContentsMargins(16, 14, 16, 14)
        self.prediction_layout.setSpacing(10)

        # ── Model source ──
        self.prediction_file_input = QLineEdit()
        self.prediction_file_input.setPlaceholderText("Path to trained model file (.xlsx or .pkl)…")
        self.prediction_browse_file_btn = QPushButton("Browse")
        self.prediction_browse_file_btn.clicked.connect(
            lambda: self.browse(self.prediction_file_input,
                                file_types=[("Excel", "*.xlsx"), ("Pickle", "*.pkl")])
        )
        self._secondary(self.prediction_browse_file_btn, "Browse for a trained model (.xlsx result folder or .pkl)")
        self.prediction_hbox1 = QHBoxLayout()
        self.prediction_hbox1.setSpacing(8)
        self.prediction_hbox1.addWidget(self.prediction_file_input)
        self.prediction_hbox1.addWidget(self.prediction_browse_file_btn)
        self.prediction_layout.addLayout(self.prediction_hbox1)

        self.prediction_load_file_btn = QPushButton("Load Model(s)")
        self.prediction_load_file_btn.setToolTip("Load all .pkl models from the selected Excel result folder")
        self.prediction_load_file_btn.clicked.connect(self.load_models)
        self.prediction_layout.addWidget(self.prediction_load_file_btn)
        self.prediction_layout.addWidget(self._make_separator())

        # ── Info images ──
        self.prediction_image_labels_hbox = QHBoxLayout()
        self.luminol_sensor_image_label = QLabel()
        self.luminol_sensor_image_label.setPixmap(QPixmap(resource_path("media/ECL_biosensor.png")))
        self.luminol_sensor_image_label.setAlignment(Qt.AlignCenter)
        self.luminol_working_image_label = QLabel()
        self.luminol_working_image_label.setPixmap(QPixmap(resource_path("media/luminol_working_principle.png")))
        self.luminol_working_image_label.setAlignment(Qt.AlignCenter)
        self.prediction_image_labels_hbox.addWidget(self.luminol_sensor_image_label)
        self.prediction_image_labels_hbox.addWidget(self.luminol_working_image_label)
        self.prediction_layout.addLayout(self.prediction_image_labels_hbox)
        self.prediction_layout.addWidget(self._make_separator())

        # ── Reagent ──
        self.prediction_reagent_label = QLabel("Reagent:")
        self.prediction_reagent_dropdown = QComboBox()
        self.prediction_reagent_dropdown.addItems(dropdown_options)
        self.prediction_reagent_dropdown.setToolTip("Reagent used for the image being predicted")
        self.prediction_hbox3 = QHBoxLayout()
        self.prediction_hbox3.setSpacing(8)
        self.prediction_hbox3.addWidget(self.prediction_reagent_label)
        self.prediction_hbox3.addWidget(self.prediction_reagent_dropdown)
        self.prediction_hbox3.addStretch()
        self.prediction_layout.addLayout(self.prediction_hbox3)

        # ── Input method ──
        self.select_input_method_label = QLabel("Input method:")
        self.select_input_method = QComboBox()
        self.select_input_method.addItems(["Image or GIF file", "Manual value"])
        self.select_input_method.setToolTip("Choose whether to measure intensity from an image or enter it directly")
        self.set_input_method_btn = QPushButton("Select")
        self.set_input_method_btn.clicked.connect(self.set_prediction_input_method)
        self._secondary(self.set_input_method_btn)
        self.prediction_hbox4 = QHBoxLayout()
        self.prediction_hbox4.setSpacing(8)
        self.prediction_hbox4.addWidget(self.select_input_method_label)
        self.prediction_hbox4.addWidget(self.select_input_method)
        self.prediction_hbox4.addWidget(self.set_input_method_btn)
        self.prediction_hbox4.addStretch()
        self.prediction_layout.addLayout(self.prediction_hbox4)

        # ── Manual entry ──
        self.prediction_enter_manually_label = QLabel("Intensity value:")
        self.prediction_x_val_entry = QLineEdit()
        self.prediction_x_val_entry.setPlaceholderText("Enter numeric intensity…")
        self.prediction_x_val_entry.setMaximumWidth(180)
        self.prediction_hbox5 = QHBoxLayout()
        self.prediction_hbox5.setSpacing(8)
        self.prediction_hbox5.addWidget(self.prediction_enter_manually_label)
        self.prediction_hbox5.addWidget(self.prediction_x_val_entry)
        self.prediction_hbox5.addStretch()
        self.prediction_layout.addLayout(self.prediction_hbox5)

        # ── Image path entry ──
        self.prediction_image_input = QLineEdit()
        self.prediction_image_input.setPlaceholderText("Path to image or GIF file…")
        self.prediction_image_input.setVisible(False)
        self.prediction_image_browse_btn = QPushButton("Browse")
        self.prediction_image_browse_btn.clicked.connect(
            lambda: self.browse(self.prediction_image_input, file_types=[
                ("Images", "*.jpg"), ("Images", "*.png"),
                ("Images", "*.jpeg"), ("GIF", "*.gif"),
            ])
        )
        self._secondary(self.prediction_image_browse_btn, "Browse for the image to predict")
        self.prediction_image_browse_btn.setVisible(False)
        self.prediction_hbox2 = QHBoxLayout()
        self.prediction_hbox2.setSpacing(8)
        self.prediction_hbox2.addWidget(self.prediction_image_input)
        self.prediction_hbox2.addWidget(self.prediction_image_browse_btn)
        self.prediction_layout.addLayout(self.prediction_hbox2)

        # ── Predict button ──
        self.prediction_load_and_predict_btn = QPushButton("🔬  Predict Concentration")
        self.prediction_load_and_predict_btn.setVisible(False)
        self.prediction_load_and_predict_btn.clicked.connect(self.load_and_predict)
        self.prediction_layout.addWidget(self.prediction_load_and_predict_btn)
        self.prediction_layout.addWidget(self._make_separator())

        # ── Results ──
        self.results_label = QLabel("")
        self.results_label.setVisible(False)
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet(
            "background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; "
            "padding: 10px 14px; color: #166534; font-size: 12px; line-height: 1.5;"
        )
        self.download_results_tn = QPushButton("⬇  Download Results")
        self.download_results_tn.setVisible(False)
        self.download_results_tn.setStyleSheet(
            "background-color: #16a34a; color: white; font-weight: 600; "
            "border-radius: 5px; padding: 7px 18px; min-height: 32px;"
        )
        self.reset_button = QPushButton("↺  Reset")
        self.reset_button.clicked.connect(lambda: self.reset_tab(self.prediction_layout))
        self.reset_button.setVisible(False)
        self.reset_button.setStyleSheet(
            "background-color: #64748b; font-weight: 600; border-radius: 5px; "
            "padding: 7px 18px; min-height: 32px;"
        )
        self.prediction_vbox3 = QVBoxLayout()
        self.prediction_vbox3.setSpacing(8)
        self.prediction_vbox3.addWidget(self.results_label)
        self.prediction_vbox3.addWidget(self.download_results_tn)
        self.prediction_vbox3.addWidget(self.reset_button)
        self.prediction_layout.addLayout(self.prediction_vbox3)
        self.prediction_layout.addStretch()

        self.prediction_tab.setLayout(self.prediction_layout)

    def init_about_us_tab(self):
        self.about_layout = QVBoxLayout()
        self.about_layout.setContentsMargins(32, 24, 32, 24)
        self.about_layout.setSpacing(20)
        self.about_layout.setAlignment(Qt.AlignCenter)

        self.organisation_logo_hbox = QHBoxLayout()
        self.organisation_logo_hbox.setSpacing(40)
        self.organisation_logo_hbox.setAlignment(Qt.AlignCenter)
        self.bits_image_about = QLabel()
        self.mmne_image_about = QLabel()

        i1 = Image.open(resource_path("media/bits_logo.jpg"))
        i1 = i1.resize((200, 200 * i1.height // i1.width))
        self.bits_image_about.setPixmap(QPixmap(numpy_to_qt_image(np.array(i1), swapped=False)))
        self.bits_image_about.setAlignment(Qt.AlignCenter)
        i2 = Image.open(resource_path("media/mmne.jpg"))
        i2 = i2.resize((200, 200 * i2.height // i2.width))
        self.mmne_image_about.setPixmap(QPixmap(numpy_to_qt_image(np.array(i2), swapped=False)))
        self.mmne_image_about.setAlignment(Qt.AlignCenter)
        self.organisation_logo_hbox.addWidget(self.bits_image_about)
        self.organisation_logo_hbox.addWidget(self.mmne_image_about)
        self.about_layout.addLayout(self.organisation_logo_hbox)

        self.about_layout.addWidget(self._make_separator())

        self.about_text = QLabel(
            "MEMS, Microfluidics and Nanoelectronics Lab is a collaborative effort\n"
            "across the departments at BITS-Pilani, Hyderabad Campus.\n"
            "The lab is spread across 2500 sqft with various fabrication,\n"
            "characterization and testing facilities.\n\n"
            "The lab focuses on miniaturized sensing/monitoring devices for\n"
            "Energy, Biomedical, and Biochemical applications.\n\n"
            "www.mmne.in"
        )
        self.about_text.setFont(QFont("Calibri", pointSize=14, weight=30))
        self.about_text.setAlignment(Qt.AlignCenter)
        self.about_text.setStyleSheet("color: #334155; line-height: 1.6;")
        self.about_layout.addWidget(self.about_text)

        self.about_us_tab.setLayout(self.about_layout)
        self.main_layout.addWidget(self.tab_widget, stretch=1)

    def init_footer(self):
        self.footer_label = QLabel(" ")
        self.footer_label.setObjectName("statusLabel")
        self.footer_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.main_layout.addWidget(self.footer_label)

        self.perform_analysis_button.clicked.connect(self.check_path_image_input)
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)
        self.hide_elements(self.prediction_hbox5, footer=False)
        self.hide_elements(self.prediction_hbox2, footer=False)
        self.hide_elements(self.prediction_hbox4)
        self.startup()

    def startup(self):
        """Apply final size policies and alignment after all widgets are built."""
        image_labels = {
            self.luminol_formula_img_label, self.luminol_sensor_image_label,
            self.luminol_working_image_label, self.image_label, self.dynamic_label,
        }
        skip_policy = image_labels | {self.about_text}

        elements = (self.getElements(self.prediction_layout)
                    + self.getElements(self.data_layout)
                    + self.getElements(self.image_layout))

        for el in elements:
            if isinstance(el, (QLabel, QPushButton, QLineEdit, QComboBox)):
                el.setFont(self.main_font)
            if isinstance(el, (QPushButton,)) and el not in skip_policy:
                el.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            if isinstance(el, QLabel):
                if el in image_labels:
                    el.setAlignment(Qt.AlignCenter)
                elif el not in skip_policy:
                    el.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def browse(self, input_element, is_file=True, file_types=[("Excel Files", "*.xlsx")]):
        if is_file:
            path, _ = QFileDialog.getOpenFileName(
                self, filter=";;".join([f"{desc} ({ext})" for desc, ext in file_types])
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            input_element.setText(path)

    def tab_one_browse_folder_or_image(self):
        if self.multiple_or_single_image_dropdown.currentText().strip() == "Single":
            path, _ = QFileDialog.getOpenFileName(self, filter=";;".join([
                f"{d} ({e})" for d, e in [
                    ("Image", "*.jpg"), ("GIF", "*.gif"),
                    ("Image", "*.jpeg"), ("Image", "*.png"),
                ]
            ]))
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            self.image_folder_input.setText(path)

    def save_image_intensity_data(self):
        data = self.processor.data
        folder_path = self.processor.folder_path
        if len(data) == 0:
            self._set_status("No data yet — run analysis first.", "warning")
            return
        if data is None:
            self._set_status("Please calculate image intensities before saving.", "warning")
            return
        if type(data) == str:
            filename = f"{os.path.splitext(folder_path)[0]}_calculatedIntensity.txt"
            with open(filename, "w+") as f:
                f.write(data)
        else:
            filename = os.path.join(folder_path, "data.xlsx")
            makeExcel(path=filename, data=data, sortby="Concentration")
        self._set_status(f"Saved → {filename}", "success")
        open_window(os.path.split(filename)[0])

    def pause_resume_analysis(self):
        if "pause" in self.pause_resume_analysis_button.text().lower():
            self.processor.pause()
            self.pause_resume_analysis_button.setText("▶  Resume")
        else:
            self.processor.resume()
            self.pause_resume_analysis_button.setText("⏸  Pause")

    def check_path_image_input(self):
        if not os.path.exists(self.image_folder_input.text()):
            self._set_status("Please enter a valid path.", "error")
            return

        if self.multiple_or_single_image_dropdown.currentText().lower() == "multiple":
            self.perform_analysis_button.setText("Analysing…")
            self.perform_analysis_button.setEnabled(False)
            self.pause_resume_analysis_button.setVisible(True)
            ui_refs = {
                'progress_bar':        self.progress_bar,
                'progress_status':     self.progress_label,
                'status_label':        self.footer_label,
                'image_placeholder':   self.image_label,
                'mean_label':          self.dynamic_label,
                'pause_resume_button': self.pause_resume_analysis_button,
            }
            if self.processor.initialize(
                self.image_folder_input.text(),
                self.reagent_dropdown.currentText().lower(),
                ui_refs,
            ):
                self.processor.start()

        elif self.image_folder_input.text().lower().strip().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            self.image_folder_input.setDisabled(True)
            image_path = self.image_folder_input.text()
            self._set_status("Processing…", "working")

            if image_path.endswith(".gif"):
                from image_analysis import getFrame
                image = getFrame(image_path)
            else:
                image = imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

            reagent = self.reagent_dropdown.currentText().lower()
            x_val, crop_cords = 0, {}
            if "auto" in reagent:
                for r in Reagent.reagents:
                    x_val, area, crop_cords = getPlainMean(image, r.name)
                    if x_val > 0 and area > 1000:
                        reagent = r.name
                        break
            else:
                x_val, _, crop_cords = getPlainMean(image, reagent)

            if not crop_cords:
                self._set_status("No luminescent region detected. Check reagent selection.", "error")
                self.image_folder_input.setDisabled(False)
                return

            i5 = crop_image(image, crop_cords, pad=10)
            from numpy import uint8
            i5 = Image.fromarray(uint8(cvtColor(i5, COLOR_BGR2RGB)))
            i5 = i5.resize((200, 200 * i5.height // i5.width))
            self.image_label.setPixmap(QPixmap(numpy_to_qt_image(np.array(i5), swapped=False)))
            self.image_label.setVisible(True)
            self.dynamic_label.setVisible(True)
            self.dynamic_label.setText(f"Intensity: {round(x_val, 2)}")
            self.processor.folder_path = image_path
            self.processor.data = f"{image_path} -> Intensity: {x_val}"
            self._set_status("Done. Use 'Save Data to Excel' to export.", "success")
        else:
            self._set_status("Please enter a valid image or GIF path.", "error")

    def load_listbox_bloc(self):
        path = self.data_analysis_file_input_bar.text().strip()
        if os.path.exists(path) and path.endswith((".xlsx", ".xls")):
            self.data_analysis_file_input_bar.setDisabled(True)
            self.listbox.setVisible(True)
            self.select_all_checkbox.setVisible(True)
            self.set_models_btn.setVisible(True)
            self._set_status("Select the models to train, then click 'Confirm Model Selection'.", "working")
        else:
            self._set_status("Please enter a valid .xlsx file path.", "error")

    def load_data_from_file(self):
        filepath = self.data_analysis_file_input_bar.text()
        selected_items = self.listbox.selectedItems()
        if not selected_items:
            self._set_status("Please select at least one model.", "error")
            return
        selected_models = [m for m in ML_Model.models if m.name in [i.text() for i in selected_items]]
        self.listbox.setDisabled(True)
        self.select_all_checkbox.setDisabled(True)
        df = read_excel(filepath)
        labels = df.columns.tolist()
        self.x_var_dropdown.addItems(labels)
        self.y_var_dropdown.addItems(labels)
        self.x_var_dropdown.setVisible(True)
        self.y_var_dropdown.setVisible(True)
        self.x_var_label.setVisible(True)
        self.y_var_label.setVisible(True)
        self.set_labels_btn.setVisible(True)
        self._set_status("Choose X and Y columns, then confirm.", "working")

    def ask_test_percentage(self):
        x_label = self.x_var_dropdown.currentText()
        y_label = self.y_var_dropdown.currentText()
        if x_label == y_label:
            self._set_status("X and Y labels must be different.", "error")
            return
        self.X.label = x_label
        self.Y.label = y_label
        self.set_test_btn.setVisible(True)
        self.test_percentage_input.setVisible(True)
        self.test_percentage_label.setVisible(True)
        self.x_var_dropdown.setDisabled(True)
        self.y_var_dropdown.setDisabled(True)
        self._set_status("Labels set. Enter test split % then click 'Train Models'.", "working")

    def set_test_percentage_and_run(self):
        test_percentage = self.test_percentage_input.text().strip()
        if not is_float(test_percentage):
            self._set_status("Enter a number between 1 and 100 for the test split.", "error")
            return
        test_percentage = float(test_percentage)
        filepath = self.data_analysis_file_input_bar.text()
        selected_models = [m for m in ML_Model.models if m.name in [i.text() for i in self.listbox.selectedItems()]]
        df = read_excel(filepath)
        parentPath = os.path.join(
            os.path.split(filepath)[0],
            os.path.splitext(os.path.split(filepath)[-1])[0],
        )
        os.makedirs(parentPath, exist_ok=True)
        self.test_percentage_input.setDisabled(True)
        self.set_test_btn.setEnabled(False)
        self._set_status("Training models… this may take a moment.", "working")
        try:
            process_main(self.X, self.Y, df, int(test_percentage) / 100, parentPath, selected_models)
        except Exception as e:
            self._set_status(f"Error during training: {e}", "error")
            self.set_test_btn.setEnabled(True)
            return
        self._set_status("Training complete. Click a model button to view its scatter plot.", "success")

        def open_graph_image(i):
            path = f"{selected_models[i].name.strip().replace(' ', '_').lower()}.jpg"
            open_window(os.path.join(parentPath, path))

        self.hbox8 = QHBoxLayout()
        self.hbox8.setSpacing(6)
        for i, m in enumerate(selected_models):
            btn = QPushButton(m.name)
            btn.setToolTip(f"Open scatter plot for {m.name}")
            btn.clicked.connect(lambda checked, n=i: open_graph_image(n))
            self.hbox8.addWidget(btn)
        self.data_layout.insertLayout(self.data_layout.count() - 1, self.hbox8)
        open_window(parentPath)
        self.reset_tab_btn.setVisible(True)

    # ------------------------------------------------------------------
    # Layout utilities (unchanged structure for hide/show compatibility)
    # ------------------------------------------------------------------

    def hide_elements(self, layout, exempt_list=[], footer=True):
        exempt_list = (
            [self.load_data_button, self.browse_file_data_analysis, self.data_analysis_file_input_bar]
            if len(exempt_list) == 0 else exempt_list
        )
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                if widget not in exempt_list:
                    widget.setVisible(False)
                widget.setDisabled(False)
            elif item.layout() is not None:
                self.hide_elements(item.layout(), exempt_list, footer=False)
        if footer:
            self.footer_label.setText("")

    def load_elements(self, layout, exempt_list=[]):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                if widget not in exempt_list:
                    widget.setVisible(True)
                    widget.setDisabled(False)
            elif item.layout() is not None and item.layout() not in exempt_list:
                self.load_elements(item.layout(), exempt_list)

    def getElements(self, layout):
        elements = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                elements.append(item.widget())
            elif item.layout() is not None:
                elements += self.getElements(item.layout())
        return elements

    def reset_tab(self, layout=None):
        layout = self.data_layout if layout is None else layout
        if layout == self.data_layout:
            self.X.reset()
            self.Y.reset()
            self.processor.reset()
            self.hide_elements(self.data_layout)
        elif layout == self.prediction_layout:
            self.hide_elements(
                self.prediction_layout,
                exempt_list=[
                    self.luminol_working_image_label, self.luminol_sensor_image_label,
                    self.prediction_load_file_btn, self.prediction_reagent_dropdown,
                ] + self.getElements(self.prediction_hbox1) + self.getElements(self.prediction_hbox3),
            )

    def load_models(self):
        path = self.prediction_file_input.text().strip()
        if os.path.exists(path) and path.endswith(".xlsx"):
            self.prediction_file_input.setDisabled(True)
            self.load_elements(self.prediction_hbox4)
            self._set_status("Model file loaded. Choose input method.", "working")
        else:
            self._set_status("Please enter a valid .xlsx file path.", "error")

    def set_prediction_input_method(self):
        self.select_input_method.setDisabled(True)
        if "image" in self.select_input_method.currentText().lower():
            self.load_elements(self.prediction_hbox2)
        else:
            self.load_elements(self.prediction_hbox5)
        self.prediction_load_and_predict_btn.setVisible(True)

    def load_and_predict(self):
        x_val = None
        if "image" in self.select_input_method.currentText().lower():
            img_path = self.prediction_image_input.text()
            if not os.path.exists(img_path) or not img_path.lower().endswith((".gif", ".jpg", ".jpeg", ".png")):
                self._set_status("Please enter a valid image path.", "error")
                return
            self.prediction_image_input.setDisabled(True)
            reagent = self.prediction_reagent_dropdown.currentText()
            self.prediction_reagent_dropdown.setDisabled(True)
            if img_path.endswith(".gif"):
                from image_analysis import getFrame
                image = getFrame(img_path)
            else:
                image = imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if "auto" in reagent.lower():
                for r in Reagent.reagents:
                    x_val, area, _ = getPlainMean(image, r.name)
                    if x_val > 0 and area > 1000:
                        break
            else:
                x_val, _, _ = getPlainMean(image, reagent)
        else:
            if not is_float(self.prediction_x_val_entry.text()):
                self._set_status("Please enter a valid numeric intensity value.", "error")
                return
            x_val = float(self.prediction_x_val_entry.text())

        if x_val is None:
            self._set_status("Could not determine intensity value.", "error")
            return

        loaded_models = load(self.prediction_file_input.text().strip())
        predictions, label_text = predict_value(x_val, loaded_models)
        self.results_label.setText(
            f"At intensity {round(x_val, 3)}, predicted concentrations:\n\n{label_text}"
        )
        self.download_results_tn.clicked.connect(
            lambda: (
                download_predictions(x_val, predictions, parentPath=self.prediction_file_input.text().strip()),
                self._set_status("Results downloaded.", "success"),
            )
        )
        self.load_elements(self.prediction_vbox3)
        self._set_status("Prediction complete.", "success")
