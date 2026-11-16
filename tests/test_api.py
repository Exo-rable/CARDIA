import json
import threading
import time
import unittest
from urllib import error, request

import joblib
import numpy as np
import pandas as pd
import uvicorn

from backend.main import app
from common import (CLASSIFICATION_THRESHOLD, CLEAN, FEATURES, LABELS, MODEL, POSITIVE_CLASS,
                    positive_class_probability, probabilities_in_class_order)

HOST = "127.0.0.1"
PORT = 8123

POSITIVE_CASE = {
    "age": 59, "sex": 1, "cp": 0, "trestbps": 135, "chol": 234, "fbs": 0,
    "restecg": 1, "thalach": 161, "exang": 0, "oldpeak": 0.5, "slope": 1,
    "ca": 0, "thal": 3,
}
NEGATIVE_CASE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1,
}


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 20
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.05)
        if not cls.server.started:
            raise RuntimeError("FastAPI test server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    def fetch_json(self, path, payload=None, method="GET"):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(f"http://{HOST}:{PORT}{path}", data=data, method=method)
        if payload is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise AssertionError(f"HTTP {exc.code} for {path}: {body}") from exc

    def fetch_status(self, path, payload, method="POST"):
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(f"http://{HOST}:{PORT}{path}", data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with request.urlopen(req, timeout=20) as resp:
                return resp.status
        except error.HTTPError as exc:
            return exc.code

    def test_health_endpoint(self):
        resp = self.fetch_json("/api/health")
        self.assertEqual(resp["status"], "ok")
        self.assertTrue(resp["model_loaded"])

    def test_model_metadata(self):
        resp = self.fetch_json("/api/model")
        self.assertEqual(resp["selected_model"], "Logistic Regression")
        self.assertIn("test_metrics", resp)
        self.assertIn("features", resp)
        self.assertEqual(len(resp["features"]), 13)

    def test_input_schema_matches_supported_features_and_limits(self):
        schema = self.fetch_json("/api/input-schema")
        self.assertEqual(set(schema), set(FEATURES))
        self.assertEqual(schema["age"]["minimum"], 20)
        self.assertEqual(schema["age"]["maximum"], 90)
        self.assertEqual(schema["oldpeak"]["minimum"], 0)
        self.assertEqual(schema["oldpeak"]["maximum"], 7)
        self.assertEqual(schema["cp"]["enum"], [0, 1, 2, 3])
        self.assertEqual(schema["thal"]["enum"], [1, 2, 3])

    def test_prediction_probability_class_and_shap_additivity(self):
        pipe = joblib.load(MODEL)
        prep = pipe.named_steps["prep"]
        estimator = pipe.named_steps["model"]
        observed_classes = set()

        for payload in (POSITIVE_CASE, NEGATIVE_CASE):
            prediction = self.fetch_json("/api/predict", payload, method="POST")
            row = pd.DataFrame([payload], columns=FEATURES)
            expected_probability = float(positive_class_probability(pipe, row)[0])
            self.assertAlmostEqual(prediction["probability"], expected_probability, places=7)
            self.assertEqual(set(prediction), {
                "probability", "model_name", "model_version", "disclaimer",
            })
            self.assertEqual(prediction["model_name"], "Logistic Regression")
            predicted_class = int(expected_probability >= CLASSIFICATION_THRESHOLD)
            observed_classes.add(predicted_class)
            explanation = self.fetch_json("/api/explain", payload, method="POST")
            self.assertEqual(set(explanation), {
                "output_space", "base_value", "contributions", "note",
            })
            self.assertEqual(explanation["output_space"], "log_odds")
            self.assertEqual(
                {item["feature"] for item in explanation["contributions"]},
                set(FEATURES),
            )
            decision = float(estimator.decision_function(prep.transform(row))[0])
            reconstructed = explanation["base_value"] + sum(
                item["value"] for item in explanation["contributions"]
            )
            self.assertAlmostEqual(reconstructed, decision, places=5)

        self.assertEqual(observed_classes, {0, 1})

    def test_probability_uses_positive_class_label_not_column_position(self):
        class ReverseClassEstimator:
            classes_ = np.array([POSITIVE_CLASS, 0])

            @staticmethod
            def predict_proba(_features):
                return np.array([[0.73, 0.27]])

        actual = positive_class_probability(ReverseClassEstimator(), pd.DataFrame([{}]))
        self.assertEqual(float(actual[0]), 0.73)
        ordered = probabilities_in_class_order(
            ReverseClassEstimator(), pd.DataFrame([{}]), [0, 1]
        )
        np.testing.assert_array_equal(ordered, np.array([[0.27, 0.73]]))

    def test_cleveland_category_mappings(self):
        self.assertEqual(LABELS["cp"], {
            0: "Typical angina", 1: "Atypical angina",
            2: "Non-anginal pain", 3: "Asymptomatic",
        })
        self.assertEqual(LABELS["restecg"], {
            0: "Normal", 1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy",
        })
        self.assertEqual(LABELS["slope"], {
            0: "Upsloping", 1: "Flat", 2: "Downsloping",
        })
        data = pd.read_csv(CLEAN)
        for feature, value_labels in LABELS.items():
            with self.subTest(feature=feature):
                observed = set(data[feature].dropna().astype(int))
                self.assertEqual(observed, set(value_labels))

    def test_missing_and_invalid_input_is_rejected(self):
        missing = dict(POSITIVE_CASE)
        del missing["age"]
        invalid_range = {**POSITIVE_CASE, "age": 121}
        invalid_category = {**POSITIVE_CASE, "cp": 4}
        unexpected = {**POSITIVE_CASE, "new_feature": 1}
        for payload in (missing, invalid_range, invalid_category, unexpected):
            with self.subTest(payload=payload):
                self.assertEqual(self.fetch_status("/api/predict", payload), 422)


if __name__ == "__main__":
    unittest.main()
