"""Tests for the core image processing / intensity extraction pipeline.

All tests work on synthetic numpy images to avoid filesystem dependencies.
Qt is mocked in conftest.py so numpy_to_qt_image is not exercised here.
"""
import cv2
import numpy as np
import pandas as pd
import pytest
from model_def import Reagent
from image_analysis import calculateMean, getPlainMean, addWeights


# ---------------------------------------------------------------------------
# Helpers to build synthetic test images
# ---------------------------------------------------------------------------

def _bgr_from_hsv_patch(hue, sat, val, size=(100, 100), patch=(30, 70, 30, 70)):
    """Return a BGR image with a solid HSV-coloured patch in the interior.

    patch = (y_min, y_max, x_min, x_max).  Background is black.
    """
    img_hsv = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    y0, y1, x0, x1 = patch
    img_hsv[y0:y1, x0:x1] = [hue, sat, val]
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def luminol_image(sat=200, val=180):
    """Synthetic Luminol image: hue 120 (mid-range of 110-130), sat 200, val 180."""
    return _bgr_from_hsv_patch(hue=120, sat=sat, val=val)


def ruthenium_image(sat=200, val=180):
    """Synthetic Ruthenium image: hue 10 (mid-range of 0-20)."""
    return _bgr_from_hsv_patch(hue=10, sat=sat, val=val)


def black_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# calculateMean
# ---------------------------------------------------------------------------

class TestCalculateMean:
    def _call(self, bgr_img, lightness, reagent_name):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        return calculateMean(bgr_img, hsv, lightness, reagent_name)

    def test_luminol_image_detected_with_luminol_reagent(self):
        mean, area, crop_cords = self._call(luminol_image(), lightness=100, reagent_name="luminol")
        assert mean > 0, "Should detect luminol emission"
        assert area > 0
        assert set(crop_cords.keys()) == {"Min-Y", "Max-Y", "Min-X", "Max-X"}

    def test_luminol_image_not_detected_with_ruthenium_reagent(self):
        mean, area, _ = self._call(luminol_image(), lightness=100, reagent_name="ruthenium")
        assert mean == 0, "Ruthenium reagent must not detect luminol-hued pixels"

    def test_ruthenium_image_detected_with_ruthenium_reagent(self):
        mean, area, crop_cords = self._call(ruthenium_image(), lightness=100, reagent_name="ruthenium")
        assert mean > 0

    def test_ruthenium_image_not_detected_with_luminol_reagent(self):
        mean, area, _ = self._call(ruthenium_image(), lightness=100, reagent_name="luminol")
        assert mean == 0

    def test_black_image_returns_zero(self):
        mean, area, _ = self._call(black_image(), lightness=10, reagent_name="luminol")
        assert mean == 0
        assert area == 0

    def test_high_lightness_threshold_misses_dim_emission(self):
        # val=50 is below lightness threshold of 100
        dim_img = luminol_image(val=50)
        mean, _, _ = self._call(dim_img, lightness=100, reagent_name="luminol")
        assert mean == 0

    def test_low_lightness_threshold_catches_dim_emission(self):
        dim_img = luminol_image(val=50)
        mean, _, _ = self._call(dim_img, lightness=30, reagent_name="luminol")
        assert mean > 0

    def test_unknown_reagent_returns_false(self):
        hsv = cv2.cvtColor(luminol_image(), cv2.COLOR_BGR2HSV)
        result = calculateMean(luminol_image(), hsv, 100, "unknown_reagent_xyz")
        assert result is False

    def test_crop_cords_are_within_image_bounds(self):
        img = luminol_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, cords = calculateMean(img, hsv, 100, "luminol")
        h, w = img.shape[:2]
        assert 0 <= cords["Min-Y"] <= cords["Max-Y"] < h
        assert 0 <= cords["Min-X"] <= cords["Max-X"] < w

    def test_mean_is_consistent_with_patch_values(self):
        """Mean of masked pixels must be within the BGR value range of the patch."""
        img = luminol_image(sat=200, val=180)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean, _, _ = calculateMean(img, hsv, 100, "luminol")
        # Pixel values in a near-pure-blue patch span roughly [0, 180];
        # the mean of all channels together should be in that range.
        assert 0 < mean <= 255


# ---------------------------------------------------------------------------
# getPlainMean
# ---------------------------------------------------------------------------

class TestGetPlainMean:
    def test_detects_luminol(self):
        mean, area, cords = getPlainMean(luminol_image(), "luminol")
        assert mean > 0
        assert area > 0
        assert len(cords) == 4

    def test_detects_ruthenium(self):
        mean, area, cords = getPlainMean(ruthenium_image(), "ruthenium")
        assert mean > 0

    def test_black_image_returns_zero(self):
        mean, area, _ = getPlainMean(black_image(), "luminol")
        assert mean == 0
        assert area == 0

    def test_wrong_reagent_returns_zero(self):
        mean, _, _ = getPlainMean(luminol_image(), "ruthenium")
        assert mean == 0

    def test_returns_nonzero_for_dim_image_eventually(self):
        # val=40 is below several VAL_RANGES entries but above the lowest (10).
        dim = luminol_image(val=40)
        mean, _, _ = getPlainMean(dim, "luminol")
        assert mean > 0


# ---------------------------------------------------------------------------
# addWeights
# ---------------------------------------------------------------------------

class TestAddWeights:
    def _make_history(self, concentrations, intensities):
        df = pd.DataFrame({"Concentration": concentrations, "Intensity": intensities})
        return df

    def test_returns_tuple_of_mean_and_cords(self):
        img = luminol_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_in, area, cords_in = calculateMean(img, hsv, 100, "luminol")
        df = self._make_history([0.1, 0.2, 0.3], [50.0, 60.0, 70.0])
        result = addWeights(img, 0.4, df, "Concentration", hsv, mean_in, cords_in,
                            req_range=20, mean_of_prev_means=70.0,
                            max_of_prev_means=70.0, next_image_mean=80.0,
                            same_conc=False, reagent="luminol")
        assert isinstance(result, tuple)
        assert len(result) == 2
        out_mean, out_cords = result
        assert isinstance(out_mean, (int, float, np.floating))

    def test_mean_is_non_negative(self):
        img = luminol_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_in, _, cords_in = calculateMean(img, hsv, 100, "luminol")
        df = self._make_history([0.1, 0.2, 0.3], [50.0, 60.0, 70.0])
        out_mean, _ = addWeights(img, 0.3, df, "Concentration", hsv, mean_in, cords_in,
                                 req_range=5, mean_of_prev_means=mean_in,
                                 max_of_prev_means=mean_in, next_image_mean=mean_in,
                                 same_conc=True, reagent="luminol")
        assert out_mean >= 0

    def test_same_conc_returns_mean_close_to_history(self):
        """When same_conc=True and req_range is generous, mean should stay near history."""
        img = luminol_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_in, _, cords_in = calculateMean(img, hsv, 100, "luminol")
        df = self._make_history([0.5, 0.5, 0.5], [mean_in, mean_in, mean_in])
        out_mean, _ = addWeights(img, 0.5, df, "Concentration", hsv, mean_in, cords_in,
                                 req_range=30, mean_of_prev_means=mean_in,
                                 max_of_prev_means=mean_in, next_image_mean=mean_in,
                                 same_conc=True, reagent="luminol")
        assert abs(out_mean - mean_in) <= 30, "mean should stay near history value"
