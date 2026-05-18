import os
from time import time

import numpy as np
from cv2 import imdecode
from pandas import DataFrame
from PyQt5.QtCore import QTimer

from image_analysis import getPlainMean, debug, processImage
from model_def import Reagent
from util import is_float


class ImageProcessor:
    """Owns all batch-processing state and the drive timer, keeping it off MainWindow."""

    def __init__(self, on_complete=None):
        self.current_index = 0
        self.total_images = []
        self.data = DataFrame()
        self.folder_path = ''
        self.reagent = ''
        self.start_time = 0.0
        self.on_complete = on_complete

        self._ui = {}
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, folder_path, reagent, ui_refs):
        """Set up state for a new batch run.

        ui_refs keys: progress_bar, progress_status, status_label,
                      image_placeholder, mean_label, pause_resume_button
        Returns True on success, False if folder_path does not exist.
        """
        if not os.path.exists(folder_path):
            ui_refs['status_label'].setText("Enter a Valid Folderpath")
            return False

        self._ui = ui_refs
        self.folder_path = folder_path
        self.total_images = self._collect_images(folder_path)
        self.reagent = self._resolve_reagent(reagent)
        self.current_index = 0
        self.data = DataFrame(columns=["Concentration", "Intensity"])
        self.start_time = time()
        return True

    def start(self):
        self._timer.start(1)

    def pause(self):
        self._timer.stop()

    def resume(self):
        self._timer.start()

    def reset(self):
        self._timer.stop()
        self.current_index = 0
        self.total_images = []
        self.data = DataFrame()
        self.folder_path = ''
        self.reagent = ''
        self._ui = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_images(folder_path):
        def numeric_key(path):
            for part in os.path.split(path)[1].split(" "):
                if is_float(part):
                    return float(part)
            return 0.0

        subfolders = sorted(
            [
                os.path.join(folder_path, name)
                for name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, name))
                and any(is_float(p) for p in name.split(" "))
            ],
            key=numeric_key,
        )
        return [
            os.path.join(sub, img)
            for sub in subfolders
            for img in os.listdir(sub)
            if img.lower().endswith((".jpg", ".png", ".jpeg", ".gif"))
        ]

    def _resolve_reagent(self, reagent):
        if "auto" not in reagent.lower():
            return reagent
        if not self.total_images:
            return reagent
        mid = self.total_images[len(self.total_images) // 2]
        image = imdecode(np.fromfile(mid, dtype=np.uint8), -1)
        for r in Reagent.reagents:
            if getPlainMean(image, r.name)[1] > 0:
                debug(f"Auto-detected reagent: {r.name}")
                return r.name
        return reagent

    def _process_next(self, n=1):
        ui = self._ui
        for i in range(self.current_index, min(self.current_index + n, len(self.total_images))):
            image_path = self.total_images[i]
            result = processImage(
                ui['progress_bar'], ui['progress_status'], ui['status_label'],
                ui['image_placeholder'], ui['mean_label'],
                self.total_images, i, image_path,
                reagent=self.reagent, data=self.data,
            )
            if isinstance(result, DataFrame):
                self.data = result
            else:
                return
        self.current_index += n

    def _on_timeout(self):
        self._process_next()
        if self.current_index >= len(self.total_images):
            self._timer.stop()
            btn = self._ui.get('pause_resume_button')
            if btn:
                btn.setVisible(False)
            elapsed = round(time() - self.start_time, 2)
            self._ui['status_label'].setText(
                f"Done within {elapsed} seconds. Data is ready to be saved."
            )
            if self.on_complete:
                self.on_complete()
        else:
            self._timer.start(0)
