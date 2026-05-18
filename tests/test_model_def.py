import pytest
from model_def import DataAxis, ML_Model, Reagent


class TestDataAxis:
    def test_default_construction(self):
        axis = DataAxis()
        assert axis.label == ""
        assert axis.original == []
        assert axis.test == []
        assert axis.train == []

    def test_construction_with_values(self):
        axis = DataAxis(label="Intensity", original=[1, 2], test=[3], train=[4])
        assert axis.label == "Intensity"
        assert axis.original == [1, 2]

    def test_reset_clears_all_fields(self):
        axis = DataAxis(label="X", original=[1, 2, 3], test=[4], train=[5, 6])
        axis.reset()
        assert axis.label == ""
        assert axis.original == []
        assert axis.test == []
        assert axis.train == []

    def test_reset_is_independent_per_instance(self):
        a = DataAxis(label="A")
        b = DataAxis(label="B")
        a.reset()
        assert b.label == "B"


class TestMLModel:
    def test_models_list_is_populated(self):
        names = ML_Model.get_model_names()
        assert "Linear Regression" in names
        assert "Random Forest" in names
        assert "Support Vector Machine" in names

    def test_get_results_returns_all_metrics(self):
        model = ML_Model.models[0]
        result = model.get_results()
        assert set(result.keys()) == {"R2 Score", "MAE", "MSE", "RMSE", "Y-Pred"}

    def test_get_error_metrics_shape(self):
        metrics = ML_Model.get_error_metrics(ML_Model.models[:3])
        assert len(metrics["Model"]) == 3
        assert len(metrics["R2 Score"]) == 3


class TestReagent:
    def test_luminol_is_registered(self):
        r = Reagent.get_reagent("luminol")
        assert r is not None
        assert r.name == "Luminol"
        assert r.min_hue == 110
        assert r.max_hue == 130

    def test_ruthenium_is_registered(self):
        r = Reagent.get_reagent("ruthenium")
        assert r is not None
        assert r.name == "Ruthenium"

    def test_get_reagent_case_insensitive(self):
        assert Reagent.get_reagent("LUMINOL") is not None
        assert Reagent.get_reagent("Luminol") is not None

    def test_get_reagent_unknown_returns_none(self):
        assert Reagent.get_reagent("unknown_reagent_xyz") is None

    def test_hue_range_update_persists(self):
        r = Reagent.get_reagent("luminol")
        original_min, original_max = r.min_hue, r.max_hue
        r.min_hue = 105
        r.max_hue = 135
        assert Reagent.get_reagent("luminol").min_hue == 105
        # restore
        r.min_hue, r.max_hue = original_min, original_max
