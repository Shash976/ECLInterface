"""Tests for model loading and prediction."""
import os
import numpy as np
import pandas as pd
import joblib
import pytest
from model_def import DataAxis, ML_Model
from processing import process_main, makeExcel
from prediction import load, predict_value, download_predictions


_FAST_MODEL = next(m for m in ML_Model.models if m.name == "Linear Regression")


@pytest.fixture()
def trained_xlsx(tmp_path):
    """Train a linear model on y = 2x + 5, save everything, return xlsx path."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 60)
    y = 2 * x + 5 + rng.normal(0, 1, len(x))
    df = pd.DataFrame({"Intensity": x, "Concentration": y})

    xlsx = str(tmp_path / "data.xlsx")
    df.to_excel(xlsx, index=False)

    X = DataAxis(label="Intensity")
    Y = DataAxis(label="Concentration")
    parent = str(tmp_path / "data")
    process_main(X, Y, df, test_size=0.2, parentPath=parent, models=[_FAST_MODEL])
    return xlsx


class TestLoad:
    def test_loads_from_xlsx_returns_dict(self, trained_xlsx):
        models = load(trained_xlsx)
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_loaded_model_has_predict(self, trained_xlsx):
        models = load(trained_xlsx)
        for _, m in models.items():
            assert hasattr(m, 'model')
            assert hasattr(m.model, 'predict')

    def test_load_single_pkl(self, trained_xlsx):
        parent = os.path.join(os.path.splitext(trained_xlsx)[0], "models")
        pkl = next(f for f in os.listdir(parent) if f.endswith(".pkl"))
        models = load(os.path.join(parent, pkl))
        assert len(models) == 1


class TestPredictValue:
    def test_returns_prediction_in_reasonable_range(self, trained_xlsx):
        """At x=50 and y=2x+5, prediction should be near 105."""
        models = load(trained_xlsx)
        predictions, label_text = predict_value(50.0, models)
        assert len(predictions) > 0
        for model_obj, pred in predictions.items():
            assert 80 < pred < 130, f"{model_obj.name}: {pred} outside expected range"

    def test_label_text_contains_model_name(self, trained_xlsx):
        models = load(trained_xlsx)
        _, label_text = predict_value(50.0, models)
        assert "Linear Regression" in label_text

    def test_prediction_is_float(self, trained_xlsx):
        models = load(trained_xlsx)
        predictions, _ = predict_value(10.0, models)
        for _, pred in predictions.items():
            assert isinstance(pred, float)

    def test_higher_input_gives_higher_prediction(self, trained_xlsx):
        """Monotonicity check: for an increasing linear function, higher x → higher y."""
        models = load(trained_xlsx)
        pred_low = list(predict_value(10.0, models)[0].values())[0]
        pred_high = list(predict_value(90.0, models)[0].values())[0]
        assert pred_high > pred_low


class TestDownloadPredictions:
    def test_creates_xlsx_file(self, trained_xlsx, tmp_path):
        models = load(trained_xlsx)
        predictions, _ = predict_value(50.0, models)
        download_predictions(50.0, predictions, parentPath=trained_xlsx)
        pred_dir = os.path.join(os.path.splitext(trained_xlsx)[0], "predictions")
        assert os.path.exists(pred_dir)
        files = os.listdir(pred_dir)
        assert any(f.endswith(".xlsx") for f in files)

    def test_output_contains_x_val(self, trained_xlsx):
        models = load(trained_xlsx)
        predictions, _ = predict_value(42.0, models)
        download_predictions(42.0, predictions, parentPath=trained_xlsx)
        pred_dir = os.path.join(os.path.splitext(trained_xlsx)[0], "predictions")
        xlsx = os.path.join(pred_dir, os.listdir(pred_dir)[0])
        df = pd.read_excel(xlsx)
        assert 42.0 in df["Prediction"].values
