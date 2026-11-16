"""Headless test of the Streamlit app. Run from the project root: python tests/smoke_test.py"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PROBABILITY_LABEL = "Model-estimated probability for the predicted dataset class"


def predicted_class_probability(at):
    for m in at.metric:
        if m.label == PROBABILITY_LABEL:
            return float(m.value.strip("%")) / 100
    raise AssertionError("model-estimated class probability metric not found")


def visible_text(at):
    elements = ("markdown", "caption", "info", "warning", "subheader")
    return "\n".join(
        str(element.value)
        for name in elements
        for element in getattr(at, name, [])
    )


def run_preset(name):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="preset").set_value(name).run()
    assert not at.exception, at.exception
    return at


def main():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    default_probability = predicted_class_probability(at)
    assert 0.5 <= default_probability <= 1, "predicted-class probability is outside [0.5, 1]"

    negative = run_preset("Example: lower-risk profile")
    negative_probability = predicted_class_probability(negative)
    assert 0.5 <= negative_probability <= 1
    assert "did not identify the recorded heart-disease pattern" in visible_text(negative)
    assert "not a clinically calibrated personal risk estimate" in visible_text(negative)

    positive = run_preset("Example: higher-risk profile")
    positive_probability = predicted_class_probability(positive)
    assert 0.5 <= positive_probability <= 1
    assert "found a pattern associated with the heart-disease dataset category" in visible_text(positive)
    assert "not a clinically calibrated personal risk estimate" in visible_text(positive)
    assert "Information moving the positive-class model output" in visible_text(positive)
    print(
        f"default {default_probability:.1%} | negative-class preset "
        f"{negative_probability:.1%} | positive-class preset {positive_probability:.1%}"
    )

    # out-of-range input shows a warning, extreme inputs do not crash
    positive.slider(key="w_age").set_value(20).run()
    assert not positive.exception, positive.exception
    assert any("outside the range" in w.value for w in positive.warning), "range warning missing"
    print("All app checks passed.")


if __name__ == "__main__":
    main()
