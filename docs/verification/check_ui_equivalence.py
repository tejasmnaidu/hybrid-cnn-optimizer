import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs" / "verification" / "ui_baseline.json"
WIDGET_TYPES = (
    "button",
    "slider",
    "selectbox",
    "number_input",
    "download_button",
    "expander",
    "checkbox",
    "radio",
    "text_input",
    "file_uploader",
    "text_area",
    "form_submit_button",
    "multiselect",
    "toggle",
    "date_input",
    "time_input",
    "chat_input",
)
ELEMENT_TYPES = (
    *WIDGET_TYPES,
    "tabs",
    "metric",
    "dataframe",
)


def _elements(app_test: AppTest, element_type: str) -> list:
    try:
        return list(getattr(app_test, element_type))
    except AttributeError:
        return []


def _snapshot(app_test: AppTest) -> dict:
    metrics = [
        {
            "label": item.label,
            "value": str(item.value),
            "delta": str(item.delta or ""),
        }
        for item in _elements(app_test, "metric")
    ]
    widget_keys = sorted({
        str(item.key)
        for element_type in WIDGET_TYPES
        for item in _elements(app_test, element_type)
        if getattr(item, "key", None) is not None
    })
    counts = {
        element_type: len(_elements(app_test, element_type))
        for element_type in ELEMENT_TYPES
    }
    counts["pyplot_charts"] = len(_elements(app_test, "image"))
    return {
        "widget_keys": widget_keys,
        "session_state_keys": sorted(str(key) for key in app_test.session_state),
        "element_counts": counts,
        "metric_values": metrics,
        "thd_values": [
            item["value"] for item in metrics if "thd" in item["label"].lower()
        ],
        "tabs": [item.label for item in _elements(app_test, "tabs")],
    }


def _assert_matches(stage: str, actual: dict, expected: dict) -> None:
    for field in ("widget_keys", "session_state_keys", "element_counts", "tabs"):
        if actual[field] != expected[field]:
            raise AssertionError(
                f"{stage}: {field} differs from ui_baseline.json; "
                f"actual={actual[field]!r}, expected={expected[field]!r}"
            )
    actual_metric_values = [
        (item["value"], item["delta"]) for item in actual["metric_values"]
    ]
    expected_metric_values = [
        (item["value"], item["delta"]) for item in expected["metric_values"]
    ]
    if actual_metric_values != expected_metric_values:
        raise AssertionError(f"{stage}: metric values differ from ui_baseline.json")
    if actual["thd_values"] != [
        item["value"] for item in expected["thd_values"]
    ]:
        raise AssertionError(f"{stage}: THD values differ from ui_baseline.json")


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["apptest"]
    app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app_test.run()
    if app_test.exception:
        raise AssertionError(f"first_load raised: {app_test.exception}")
    _assert_matches(
        "first_load",
        _snapshot(app_test),
        baseline["first_load"],
    )

    next(
        item for item in app_test.button if item.label == "Run New Simulation"
    ).click()
    app_test.run()
    if app_test.exception:
        raise AssertionError(f"after_run_new_simulation raised: {app_test.exception}")
    _assert_matches(
        "after_run_new_simulation",
        _snapshot(app_test),
        baseline["after_run_new_simulation"],
    )

    next(
        item
        for item in app_test.slider
        if item.key == "live_demo_modulation_index_0"
    ).set_value(0.80)
    app_test.run()
    if app_test.exception:
        raise AssertionError(f"after_slider_change_to_0_80 raised: {app_test.exception}")
    _assert_matches(
        "after_slider_change_to_0_80",
        _snapshot(app_test),
        baseline["after_slider_change_to_0_80"],
    )

    print("AppTest equivalence passed: keys, counts, tabs, metric values, THD.")


if __name__ == "__main__":
    main()
