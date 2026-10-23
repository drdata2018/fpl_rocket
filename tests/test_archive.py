"""Archive normaliser on a small recorded vaastav sample (tests/fixtures/archive_sample)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_rocket.archive import ARCHIVE_COLUMNS, DEFCON_STATS, load_season_dir, normalise_season, null_columns

SAMPLE = Path(__file__).parent / "fixtures" / "archive_sample"


def _all() -> pd.DataFrame:
    return pd.concat([load_season_dir(s, SAMPLE / s) for s in ("2024-25", "2025-26")], ignore_index=True)


def test_schema_is_stable_and_complete():
    df = _all()
    assert tuple(df.columns) == ARCHIVE_COLUMNS
    raw = [pd.read_csv(SAMPLE / s / "merged_gw.csv") for s in ("2024-25", "2025-26")]
    # 2025-26 merged_gw.csv repeats some rows verbatim (10 pairs in the full file): dropped, nothing else
    assert sum(r.duplicated().sum() for r in raw) == 2
    assert len(df) == sum(len(r.drop_duplicates()) for r in raw)
    keys = ["gw", "element", "code", "element_type", "team_id", "opponent_id", "fixture"]
    assert not df[keys].isna().any().any()
    assert set(df["element_type"]) <= {1, 2, 3, 4, 5}
    assert df["was_home"].dtype == "boolean" and str(df["kickoff_time"].dt.tz) == "UTC"
    assert not df.duplicated(["season", "gw", "element", "fixture"]).any()


def test_teams_resolved_from_fixtures_and_match_the_merged_names():
    for season in ("2024-25", "2025-26"):
        raw = pd.read_csv(SAMPLE / season / "merged_gw.csv")
        norm = load_season_dir(season, SAMPLE / season)
        j = raw.merge(norm, left_on=["GW", "element", "fixture"], right_on=["gw", "element", "fixture"])
        assert len(j) == len(raw)  # every raw row (verbatim repeats included) maps to one normalised row
        assert (j["team_x"] == j["team_y"]).all()
        assert (j["opponent_team"] == j["opponent_id"]).all()
        assert (j["team_id"] != j["opponent_id"]).all()


def test_defensive_columns_null_only_before_2025_26():
    nulls = null_columns(_all())
    assert set(nulls["2024-25"]) == set(DEFCON_STATS)
    assert nulls["2025-26"] == []


def test_values_parsed_as_numbers_not_strings():
    df = _all()
    assert str(df["expected_goals"].dtype) == "Float64" and str(df["value"].dtype) == "Int64"
    assert df["value"].between(35, 160).all()  # tenths


def test_position_fallback_and_assistant_manager_type():
    merged = pd.DataFrame(
        {
            "GW": [24, 24],
            "element": [700, 701],
            "fixture": [1, 1],
            "name": ["Coach", "Unknown"],
            "position": ["AM", "FWD"],
            "team": ["A", "B"],
            "was_home": ["True", "False"],
            "opponent_team": [2, 1],
            "minutes": [0, 90],
            "total_points": [3, 2],
            "value": [5, 45],
            "kickoff_time": ["2025-02-01T15:00:00Z"] * 2,
        }
    )
    players = pd.DataFrame({"id": [700], "code": [9], "element_type": [5]})
    teams = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    fixtures = pd.DataFrame({"id": [1], "event": [24], "team_h": [1], "team_a": [2]})
    out = normalise_season("2024-25", merged, players, teams, fixtures)
    assert list(out["element_type"]) == [5, 4]  # players_raw first, merged position as fallback
    assert list(out["team"]) == ["A", "B"] and list(out["opponent"]) == ["B", "A"]
    assert out["code"].isna().tolist() == [False, True]
    assert out["starts"].isna().all() and out["defensive_contribution"].isna().all()
