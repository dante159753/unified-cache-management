from pathlib import Path
from types import SimpleNamespace

import pytest
from a2_scheduler.host import (
    SIGKILL,
    SIGTERM,
    CommandResult,
    CommandRunner,
    NpuManager,
    NpuOperationError,
    parse_info_vnpu,
    parse_proc_mem,
)

FIXTURES = Path(__file__).with_name("fixtures")


def load_fixture(name):
    return (FIXTURES / name).read_text()


def ok(stdout=""):
    return CommandResult(0, stdout, "")


def failed(stderr="command failed"):
    return CommandResult(1, "", stderr)


def inventory(ids):
    rows = "\n".join(
        (
            "| {0:<14} | 0              | -              | FREE           "
            "| vir05_1c_16g         |"
        ).format(vnpu_id)
        for vnpu_id in ids
    )
    return """\
+----------------+----------------+----------------+----------------+----------------------+
| Vnpu ID        | Vgroup ID      | Container ID   | Status         | Template Name        |
+----------------+----------------+----------------+----------------+----------------------+
{rows}
+----------------+----------------+----------------+----------------+----------------------+
""".format(
        rows=rows
    )


class ScriptedRunner:
    def __init__(self, results, initial_ids=()):
        self.results = list(results)
        self.inventory_ids = list(initial_ids)
        self.commands = []
        self.created_ids = []
        self.destroyed_ids = []

    def run(self, argv, timeout):
        argv = list(argv)
        self.commands.append((argv, timeout))
        if "info-vnpu" in argv:
            return ok(inventory(self.inventory_ids))
        result = self.results.pop(0)
        if "create-vnpu" in argv and result.returncode == 0:
            new_id = 212 + len(self.created_ids)
            self.created_ids.append(new_id)
            self.inventory_ids.append(new_id)
        if "destroy-vnpu" in argv:
            vnpu_id = int(argv[argv.index("-v") + 1])
            self.destroyed_ids.append(vnpu_id)
            if result.returncode == 0:
                self.inventory_ids.remove(vnpu_id)
        return result


class ProcRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.commands = []

    def run(self, argv, timeout):
        self.commands.append((list(argv), timeout))
        return ok(self.outputs.pop(0))


def test_parse_vnpu_inventory_discovers_actual_ids():
    output = load_fixture("info-vnpu-four.txt")
    assert [v.id for v in parse_info_vnpu(output)] == [212, 213, 214, 215]
    assert {v.template for v in parse_info_vnpu(output)} == {"vir05_1c_16g"}


def test_parse_empty_vnpu_inventory():
    assert parse_info_vnpu(load_fixture("info-vnpu-zero.txt")) == []


def test_parse_card_processes_ignores_other_cards_and_nonpositive_pids():
    assert parse_proc_mem(load_fixture("proc-mem-card7.txt"), card_id=7) == [
        4312,
        8765,
    ]


def test_parse_actual_no_process_message():
    assert parse_proc_mem(load_fixture("proc-mem-none.txt"), card_id=7) == []


def test_list_commands_use_card_and_chip_argv():
    runner = ProcRunner(
        [load_fixture("info-vnpu-four.txt"), load_fixture("proc-mem-none.txt")]
    )
    manager = NpuManager(runner, card_id=7, chip_id=0)

    manager.list_vnpus()
    manager.list_card_pids()

    assert runner.commands == [
        (["npu-smi", "info", "-t", "info-vnpu", "-i", "7", "-c", "0"], 30),
        (["npu-smi", "info", "-t", "proc-mem", "-i", "7", "-c", "0"], 30),
    ]


def test_partial_create_rolls_back_discovered_ids():
    runner = ScriptedRunner(
        [
            ok("Create vnpu success"),
            ok("Create vnpu success"),
            failed("create failed"),
            ok("destroy 212"),
            ok("destroy 213"),
        ]
    )
    manager = NpuManager(runner, card_id=7, chip_id=0)

    with pytest.raises(NpuOperationError, match="create failed"):
        manager.create_vnpus(4, "vir05_1c_16g")

    assert runner.destroyed_ids == [212, 213]


def test_create_discovers_each_auto_assigned_id_by_inventory_difference():
    runner = ScriptedRunner([ok(), ok(), ok(), ok()], initial_ids=(99,))
    manager = NpuManager(runner, card_id=7, chip_id=0)

    created = manager.create_vnpus(4, "vir05_1c_16g")

    assert [vnpu.id for vnpu in created] == [212, 213, 214, 215]
    create_commands = [argv for argv, _ in runner.commands if "create-vnpu" in argv]
    assert all("-v" not in argv for argv in create_commands)
    assert all(argv[-2:] == ["-f", "vir05_1c_16g"] for argv in create_commands)
    assert 99 not in runner.destroyed_ids


def test_ambiguous_create_inventory_rolls_back_only_ids_from_this_call():
    class AmbiguousRunner(ScriptedRunner):
        def run(self, argv, timeout):
            result = super().run(argv, timeout)
            if "create-vnpu" in argv and result.returncode == 0:
                self.inventory_ids.append(900)
            return result

    runner = AmbiguousRunner([ok(), ok(), ok()], initial_ids=(99,))
    manager = NpuManager(runner, card_id=7, chip_id=0)

    with pytest.raises(NpuOperationError, match="exactly one"):
        manager.create_vnpus(1, "vir05_1c_16g")

    assert runner.destroyed_ids == []


def test_destroy_uses_each_supplied_vnpu_id():
    runner = ScriptedRunner([ok(), ok()], initial_ids=(212, 215))
    manager = NpuManager(runner, card_id=7, chip_id=0)

    manager.destroy_vnpus([212, 215])

    assert runner.destroyed_ids == [212, 215]


def test_evict_terms_waits_kills_remaining_and_requires_clean_samples():
    runner = ProcRunner(
        [
            load_fixture("proc-mem-card7.txt"),
            "| NPU ID | Chip ID | Process id | Process name | Used Memory |\n"
            "| 7 | 0 | 8765 | python3 | 2048 |\n",
            load_fixture("proc-mem-none.txt"),
            load_fixture("proc-mem-none.txt"),
            load_fixture("proc-mem-none.txt"),
        ]
    )
    signals = []
    sleeps = []
    manager = NpuManager(
        runner,
        card_id=7,
        chip_id=0,
        eviction_grace_seconds=30,
        kill=lambda pid, sig: signals.append((pid, sig)),
        sleep=sleeps.append,
    )

    manager.evict_card_pids()

    assert signals == [
        (4312, SIGTERM),
        (8765, SIGTERM),
        (8765, SIGKILL),
    ]
    assert sleeps == [30, 1, 1, 1]


def test_evict_rejects_processes_that_survive_kill():
    row = (
        "| NPU ID | Chip ID | Process id | Process name | Used Memory |\n"
        "| 7 | 0 | 8765 | python3 | 2048 |\n"
    )
    runner = ProcRunner([row, row, row])
    manager = NpuManager(
        runner,
        card_id=7,
        chip_id=0,
        kill=lambda pid, sig: None,
        sleep=lambda seconds: None,
    )

    with pytest.raises(NpuOperationError, match="remain"):
        manager.evict_card_pids()


def test_restrict_and_restore_device_preserve_exact_stat_mode():
    chmod_calls = []
    manager = NpuManager(
        ProcRunner([]),
        card_id=7,
        chip_id=0,
        stat_fn=lambda path: SimpleNamespace(st_mode=0o20660),
        chmod_fn=lambda path, mode: chmod_calls.append((path, mode)),
    )

    original_mode = manager.restrict_device()
    manager.restore_device(original_mode)

    assert original_mode == 0o20660
    assert chmod_calls == [("/dev/davinci7", 0), ("/dev/davinci7", 0o20660)]


def test_command_runner_invokes_subprocess_with_argv_and_no_shell(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="out", stderr="err")

    monkeypatch.setattr("a2_scheduler.host.subprocess.run", fake_run)

    result = CommandRunner().run(["npu-smi", "info"], timeout=17)

    assert result == CommandResult(0, "out", "err")
    assert calls == [
        (
            ["npu-smi", "info"],
            {"capture_output": True, "text": True, "timeout": 17, "check": False},
        )
    ]
