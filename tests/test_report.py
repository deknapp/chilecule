"""Tests for report rendering."""

import json

import pandas as pd

from chilecule.workflows.report import Report, Section, dataframe_to_markdown


def test_markdown_table_renders_without_tabulate():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    rendered = dataframe_to_markdown(frame)
    assert rendered.count("\n") == 3  # header, separator, two rows
    assert "| a" in rendered and "| b" in rendered


def test_nan_renders_as_absence_not_as_a_value():
    """REGRESSION: NaN reached the renderer as a float and printed as 'nan',
    which reads like a measurement rather than a missing one."""
    frame = pd.DataFrame({"score": [1.0, float("nan")]})
    assert "nan" not in dataframe_to_markdown(frame)
    assert "--" in dataframe_to_markdown(frame)


def test_long_cells_are_truncated():
    frame = pd.DataFrame({"smiles": ["C" * 200]})
    rendered = dataframe_to_markdown(frame)
    assert max(len(line) for line in rendered.splitlines()) < 100


def test_pipe_characters_are_escaped():
    frame = pd.DataFrame({"x": ["a|b"]})
    assert "\\|" in dataframe_to_markdown(frame)


def test_warnings_appear_before_findings():
    """A caveat below the conclusion is a caveat nobody reads."""
    report = Report(workflow="Test", subject="X")
    report.warn("the negative set is biased")
    report.add(Section(title="Results", body="an enrichment factor"))
    markdown = report.to_markdown()
    assert markdown.index("negative set is biased") < markdown.index("an enrichment factor")


def test_report_round_trips_to_json():
    report = Report(workflow="Test", subject="X")
    report.add(Section(title="S", body="b", table=pd.DataFrame({"a": [1]}), data={"k": "v"}))
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert payload["sections"][0]["k"] == "v"
    assert payload["sections"][0]["table"] == [{"a": 1}]


def test_report_saves_both_formats(tmp_path):
    report = Report(workflow="Test", subject="X")
    report.add(Section(title="S", body="b"))
    paths = report.save(tmp_path)
    assert paths["markdown"].exists() and paths["json"].exists()
    assert json.loads(paths["json"].read_text())["workflow"] == "Test"


def test_large_tables_report_omitted_rows():
    frame = pd.DataFrame({"a": range(100)})
    section = Section(title="S", table=frame, max_rows=5)
    assert "95 further rows omitted" in section.to_markdown()
