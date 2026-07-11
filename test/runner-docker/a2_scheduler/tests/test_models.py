import copy
import json
from pathlib import Path

import pytest
from a2_scheduler.models import Phase, SchedulerState, load_config

CONFIG_EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.json"


def valid_config_dict():
    return copy.deepcopy(json.loads(CONFIG_EXAMPLE.read_text()))


def test_load_config_has_four_runner_slots(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config_dict()))
    config = load_config(path)
    assert config.card_id == 7
    assert config.window_end_minute == 20
    assert [slot.name for slot in config.runner_slots] == [
        "ucm-a2-vnpu-runner-212",
        "ucm-a2-vnpu-runner-213",
        "ucm-a2-vnpu-runner-214",
        "ucm-a2-vnpu-runner-215",
    ]


def test_config_rejects_duplicate_runner_names(tmp_path):
    data = valid_config_dict()
    data["runner_slots"][1]["name"] = data["runner_slots"][0]["name"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runner names must be unique"):
        load_config(path)


def test_config_rejects_non_forward_allocation_window(tmp_path):
    data = valid_config_dict()
    data["window_start_minute"] = data["window_end_minute"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(
        ValueError, match="window_start_minute must be before window_end_minute"
    ):
        load_config(path)


def test_state_json_excludes_credentials():
    state = SchedulerState(
        phase=Phase.SERVING,
        vnpu_ids=(212, 213),
        original_device_mode=0o660,
        transition_timestamp="2026-07-11T08:00:00+08:00",
        last_error="request failed with token secret-token",
    )
    assert json.loads(state.to_json()) == {
        "phase": "SERVING",
        "vnpu_ids": [212, 213],
        "original_device_mode": 0o660,
        "transition_timestamp": "2026-07-11T08:00:00+08:00",
        "last_error": "request failed with token [REDACTED]",
    }
