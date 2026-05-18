import numpy as np
import pytest
from util import is_float, crop_image, get_image_array, getFrame


class TestIsFloat:
    def test_integer_string(self):
        assert is_float("42") is True

    def test_float_string(self):
        assert is_float("3.14") is True

    def test_negative(self):
        assert is_float("-1.5") is True

    def test_scientific_notation(self):
        assert is_float("1e-3") is True

    def test_word(self):
        assert is_float("hello") is False

    def test_empty_string(self):
        assert is_float("") is False

    def test_partial_number(self):
        assert is_float("1.2.3") is False


class TestCropImage:
    def setup_method(self):
        self.image = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)

    def test_basic_crop(self):
        cords = {"Min-Y": 2, "Max-Y": 5, "Min-X": 2, "Max-X": 5}
        result = crop_image(self.image, cords, pad=0)
        assert result.shape == (3, 3, 3)

    def test_pad_adds_context(self):
        cords = {"Min-Y": 3, "Max-Y": 6, "Min-X": 3, "Max-X": 6}
        result = crop_image(self.image, cords, pad=2)
        # with pad=2 and image bounds [0,10): y=[1,8], x=[1,8] → 7x7
        assert result.shape[0] == 7
        assert result.shape[1] == 7

    def test_clamps_to_image_bounds(self):
        # Crop that would extend beyond image if not clamped
        cords = {"Min-Y": 0, "Max-Y": 9, "Min-X": 0, "Max-X": 9}
        result = crop_image(self.image, cords, pad=5)
        assert result.shape[0] == 10
        assert result.shape[1] == 10

    def test_default_pad_is_10(self):
        cords = {"Min-Y": 3, "Max-Y": 6, "Min-X": 3, "Max-X": 6}
        result_default = crop_image(self.image, cords)
        result_explicit = crop_image(self.image, cords, pad=10)
        np.testing.assert_array_equal(result_default, result_explicit)


class TestGetImageArray:
    def test_returns_array_unchanged(self):
        arr = np.zeros((5, 5, 3), dtype=np.uint8)
        result = get_image_array(arr)
        np.testing.assert_array_equal(result, arr)
