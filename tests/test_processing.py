"""Tests for the ML training pipeline and Excel export."""
import os
import numpy as np
import pandas as pd
import pytest
from model_def import DataAxis, ML_Model
from processing import makeExcel, process_main


# Use only the fastest model so training completes quickly in tests
_FAST_MODEL = next(m for m in ML_Model.models if m.name == "Linear Regression")


@pytest.fixture()
def linear_dataset():
    """Simple y = 2x + 5 with small Gaussian noise."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 100, 60)
    y = 2 * x + 5 + rng.normal(0, 3, len(x))
    return pd.DataFrame({"Intensity": x, "Concentration": y})


@pytest.fixture()
def fresh_axes():
    return DataAxis(label="Intensity"), DataAxis(label="Concentration")


# ---------------------------------------------------------------------------
# makeExcel
# ---------------------------------------------------------------------------

class TestMakeExcel:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "out.xlsx")
        makeExcel(path, {"A": [1, 2], "B": [3, 4]})
        assert os.path.exists(path)

    def test_round_trip_values(self, tmp_path):
        path = str(tmp_path / "data.xlsx")
        data = {"X": [10, 20, 30], "Y": [100, 200, 300]}
        makeExcel(path, data)
        df = pd.read_excel(path)
        assert list(df["X"]) == [10, 20, 30]
        assert list(df["Y"]) == [100, 200, 300]

    def test_sortby_orders_rows(self, tmp_path):
        path = str(tmp_path / "sorted.xlsx")
        data = {"Val": [3, 1, 2]}
        makeExcel(path, data, sortby="Val")
        df = pd.read_excel(path)
        assert list(df["Val"]) == [1, 2, 3]

    def test_accepts_dataframe_input(self, tmp_path):
        path = str(tmp_path / "df.xlsx")
        df_in = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
        makeExcel(path, df_in)
        df_out = pd.read_excel(path)
        assert list(df_out["A"]) == [5, 6]


# ---------------------------------------------------------------------------
# process_main
# ---------------------------------------------------------------------------

class TestProcessMain:
    def test_creates_model_pkl(self, tmp_path, linear_dataset, fresh_axes):
        X, Y = fresh_axes
        X.label, Y.label = "Intensity", "Concentration"
        process_main(X, Y, linear_dataset, test_size=0.2, parentPath=str(tmp_path), models=[_FAST_MODEL])
        pkl_files = list((tmp_path / "models").glob("*.pkl"))
        assert len(pkl_files) == 1

    def test_creates_error_metrics_sheet(self, tmp_path, linear_dataset, fresh_axes):
        X, Y = fresh_axes
        X.label, Y.label = "Intensity", "Concentration"
        process_main(X, Y, linear_dataset, test_size=0.2, parentPath=str(tmp_path), models=[_FAST_MODEL])
        assert (tmp_path / "error-metrics.xlsx").exists()

    def test_creates_xy_data_sheet(self, tmp_path, linear_dataset, fresh_axes):
        X, Y = fresh_axes
        X.label, Y.label = "Intensity", "Concentration"
        process_main(X, Y, linear_dataset, test_size=0.2, parentPath=str(tmp_path), models=[_FAST_MODEL])
        assert (tmp_path / "xy-data.xlsx").exists()

    def test_r2_is_high_for_linear_data(self, tmp_path, linear_dataset, fresh_axes):
        X, Y = fresh_axes
        X.label, Y.label = "Intensity", "Concentration"
        process_main(X, Y, linear_dataset, test_size=0.2, parentPath=str(tmp_path), models=[_FAST_MODEL])
        assert _FAST_MODEL.r2 > 0.95, f"R² should be >0.95 for near-linear data, got {_FAST_MODEL.r2:.3f}"

    def test_populates_axis_train_test_splits(self, tmp_path, linear_dataset, fresh_axes):
        X, Y = fresh_axes
        X.label, Y.label = "Intensity", "Concentration"
        process_main(X, Y, linear_dataset, test_size=0.2, parentPath=str(tmp_path), models=[_FAST_MODEL])
        assert len(X.train) > 0
        assert len(X.test) > 0
        assert len(Y.train) > 0
        assert len(Y.test) > 0
        # 20% test split of 60 samples
        assert len(X.test) == 12
        assert len(X.train) == 48

    def test_separate_axes_dont_interfere(self, tmp_path, linear_dataset):
        X1, Y1 = DataAxis(label="Intensity"), DataAxis(label="Concentration")
        X2, Y2 = DataAxis(label="Intensity"), DataAxis(label="Concentration")
        process_main(X1, Y1, linear_dataset, 0.2, str(tmp_path / "run1"), [_FAST_MODEL])
        r2_run1 = _FAST_MODEL.r2
        process_main(X2, Y2, linear_dataset, 0.2, str(tmp_path / "run2"), [_FAST_MODEL])
        assert _FAST_MODEL.r2 == r2_run1  # deterministic with same random_state
