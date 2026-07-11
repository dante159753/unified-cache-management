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


def test_config_rejects_non_pinned_allocation_window(tmp_path):
    data = valid_config_dict()
    data["window_start_minute"] = data["window_end_minute"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="window_start_minute must be 0"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "repository",
            "ModelEngine-Group/other-repository",
            "repository must be ModelEngine-Group/unified-cache-management",
        ),
        (
            "required_labels",
            ["self-hosted", "npu"],
            "required_labels must be exactly self-hosted,npu,a2",
        ),
        ("card_id", 6, "card_id must be 7"),
        ("chip_id", 1, "chip_id must be 0"),
        ("window_start_minute", 1, "window_start_minute must be 0"),
        ("window_end_minute", 19, "window_end_minute must be 20"),
        ("eviction_grace_seconds", 29, "eviction_grace_seconds must be 30"),
        ("vnpu_count", 3, "vnpu_count must be 4"),
        ("vnpu_template", "vir04_1c_16g", "vnpu_template must be vir05_1c_16g"),
    ],
)
def test_config_rejects_pinned_binding_drift(tmp_path, field, value, message):
    data = valid_config_dict()
    data[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_config_rejects_github_token_shaped_environment_value(tmp_path):
    data = valid_config_dict()
    data["environment"]["RUNNER_NOTE"] = "ghp_012345678901234567890123456789012345"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(
        ValueError, match="environment must not contain GitHub token-shaped values"
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


def test_state_json_redacts_github_token_shaped_error_text():
    state = SchedulerState(
        last_error="GitHub request failed for ghp_012345678901234567890123456789012345"
    )
    assert json.loads(state.to_json())["last_error"] == (
        "GitHub request failed for [REDACTED]"
    )


def state_document(phase="WHOLE", original_device_mode=None):
    return json.dumps(
        {
            "phase": phase,
            "vnpu_ids": [],
            "original_device_mode": original_device_mode,
            "transition_timestamp": None,
            "last_error": None,
        }
    )


@pytest.mark.parametrize(
    "mode",
    [0, -1, 0o20000, 0o24000, 0o22000, 0o21000, 0o27000],
)
def test_state_rejects_invalid_original_device_mode(mode):
    with pytest.raises(ValueError, match="original_device_mode"):
        SchedulerState.from_json(state_document(original_device_mode=mode))


@pytest.mark.parametrize(
    "phase",
    [
        Phase.RESERVING,
        Phase.SPLITTING,
        Phase.SERVING,
        Phase.DRAINING,
        Phase.RESTORING,
        Phase.ERROR,
    ],
)
def test_lifecycle_state_requires_original_device_mode(phase):
    with pytest.raises(ValueError, match="requires original_device_mode"):
        SchedulerState.from_json(state_document(phase=phase.value))
