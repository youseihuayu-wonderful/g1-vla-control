import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from g1_dds_direct_readonly_profiler import (
    CYCLONEDDS_SDIST_SHA256,
    CYCLONEDDS_VERSION,
    _record_from_sample,
    profile_direct_reader,
    summarize_profile,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "g1_dds_direct_readonly_profiler.py"


def _fake_lowstate(tick=1234, source_timestamp=1_000_000_000):
    motors = [
        SimpleNamespace(
            mode=1,
            q=index / 10.0,
            dq=index / 100.0,
            ddq=index / 1000.0,
            tau_est=index / 20.0,
            temperature=[30 + index, 31 + index],
            vol=48.0,
            motorstate=0,
        )
        for index in range(35)
    ]
    message = SimpleNamespace(
        version=[1, 2],
        mode_pr=0,
        mode_machine=5,
        tick=tick,
        imu_state=SimpleNamespace(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            gyroscope=[0.1, 0.2, 0.3],
            accelerometer=[0.0, 0.0, 9.81],
            rpy=[0.01, 0.02, 0.03],
            temperature=41,
        ),
        motor_state=motors,
    )
    message.sample_info = SimpleNamespace(
        source_timestamp=source_timestamp,
        publication_handle=7,
        instance_handle=0,
        sample_state=2,
        view_state=4,
        instance_state=16,
        valid_data=True,
        sample_rank=0,
        generation_rank=0,
        absolute_generation_rank=0,
    )
    return message


def _record(index, tick, source_ns, app_ns, qhash, publication=7):
    return {
        "sample_index": index,
        "tick": tick,
        "source_timestamp_ns": source_ns,
        "application_unix_ns": source_ns + 2_000_000,
        "application_monotonic_ns": app_ns,
        "publication_handle": publication,
        "q29_sha256": qhash,
    }


class G1DDSDirectReadonlyProfilerTests(unittest.TestCase):
    def test_sample_record_preserves_sample_info_and_full29_hashes(self):
        sample = _fake_lowstate()
        record = _record_from_sample(
            sample,
            sample_index=0,
            batch_index=0,
            batch_offset=0,
            batch_size=1,
            application_unix_ns=1_002_000_000,
            application_monotonic_ns=50,
        )
        self.assertEqual(record["tick"], 1234)
        self.assertEqual(len(record["q29"]), 29)
        self.assertEqual(len(record["dq29"]), 29)
        self.assertEqual(record["source_timestamp_ns"], 1_000_000_000)
        self.assertEqual(record["publication_handle"], 7)
        self.assertTrue(record["valid_data"])
        self.assertEqual(len(record["q29_sha256"]), 64)
        self.assertEqual(len(record["payload_sha256"]), 64)

    def test_summary_separates_tick_repeat_from_unique_dds_identity(self):
        records = [
            _record(0, 10, 1_000_000_000, 1_000_000_000, "a"),
            _record(1, 10, 1_004_000_000, 1_004_000_000, "b"),
            _record(2, 11, 1_008_000_000, 1_064_000_000, "c"),
            _record(3, 13, 1_012_000_000, 1_068_000_000, "d"),
        ]
        batches = [
            {"batch_size": 1, "wait_duration_ms": 3.0, "take_duration_ms": 0.2},
            {"batch_size": 2, "wait_duration_ms": 60.0, "take_duration_ms": 0.4},
            {"batch_size": 1, "wait_duration_ms": 3.0, "take_duration_ms": 0.2},
        ]
        summary = summarize_profile(records, batches, {
            "sample_lost_total_count": 0,
            "sample_rejected_total_count": 0,
        })
        self.assertEqual(summary["tick"]["duplicate_transition_count"], 1)
        self.assertEqual(summary["tick"]["duplicate_tick_same_q29_count"], 0)
        self.assertEqual(summary["tick"]["duplicate_tick_changed_q29_count"], 1)
        self.assertFalse(summary["tick"]["tick_is_unique_sample_identity"])
        self.assertTrue(summary["sample_identity"]["identity_pair_is_unique"])
        self.assertEqual(
            summary["sample_identity"][
                "publication_handle_source_timestamp_duplicate_count"
            ],
            0,
        )
        self.assertEqual(
            summary["application_gap_ms"]["over_provisional_50ms_count"], 1
        )
        self.assertEqual(
            summary["source_timestamp"]["positive_gap_ms"][
                "over_provisional_50ms_count"
            ],
            0,
        )
        self.assertEqual(summary["reader_batches"]["multi_sample_batch_count"], 1)
        self.assertEqual(summary["reader_batches"]["maximum_batch_size"], 2)

    def test_summary_detects_duplicate_dds_identity(self):
        records = [
            _record(0, 1, 1_000, 10, "a"),
            _record(1, 2, 1_000, 20, "b"),
        ]
        summary = summarize_profile(records, [], {})
        self.assertFalse(summary["sample_identity"]["identity_pair_is_unique"])
        self.assertEqual(
            summary["sample_identity"][
                "publication_handle_source_timestamp_duplicate_count"
            ],
            1,
        )
        self.assertEqual(
            summary["source_timestamp"]["duplicate_transition_count"], 1
        )

    def test_direct_profile_runs_with_receive_only_backend(self):
        class FakeCondition:
            def __init__(self, reader, *_args):
                self.reader = reader

        class FakeWaitSet:
            def __init__(self, _participant):
                self.condition = None

            def attach(self, condition):
                self.condition = condition

            def detach(self, condition):
                self.condition = None

            def wait(self, _timeout):
                return int(bool(self.condition.reader.samples))

        class FakeReader:
            def __init__(self, _participant, _topic, listener=None):
                self.listener = listener
                self.samples = [
                    _fake_lowstate(
                        tick=100 + index // 2,
                        source_timestamp=1_000_000_000 + index * 4_000_000,
                    )
                    for index in range(20)
                ]

            def take(self, maximum, condition=None):
                count = min(maximum, 3, len(self.samples))
                result, self.samples = self.samples[:count], self.samples[count:]
                return result

        class FakeListener:
            def __init__(self, **callbacks):
                self.callbacks = callbacks

        core = ModuleType("cyclonedds.core")
        core.InstanceState = SimpleNamespace(Alive=16)
        core.SampleState = SimpleNamespace(NotRead=2)
        core.ViewState = SimpleNamespace(Any=4)
        core.Listener = FakeListener
        core.ReadCondition = FakeCondition
        core.WaitSet = FakeWaitSet
        domain = ModuleType("cyclonedds.domain")
        domain.Domain = lambda *_args: object()
        domain.DomainParticipant = lambda *_args: object()
        sub = ModuleType("cyclonedds.sub")
        sub.DataReader = FakeReader
        topic = ModuleType("cyclonedds.topic")
        topic.Topic = lambda *_args: object()
        util = ModuleType("cyclonedds.util")
        util.duration = lambda **kwargs: int(kwargs.get("milliseconds", 0) * 1e6)
        channel_config = ModuleType("unitree_sdk2py.core.channel_config")
        channel_config.ChannelConfigHasInterface = "interface=$__IF_NAME__$"
        dds = ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        dds.LowState_ = object
        modules = {
            "cyclonedds": ModuleType("cyclonedds"),
            "cyclonedds.core": core,
            "cyclonedds.domain": domain,
            "cyclonedds.sub": sub,
            "cyclonedds.topic": topic,
            "cyclonedds.util": util,
            "unitree_sdk2py": ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.core": ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel_config": channel_config,
            "unitree_sdk2py.idl": ModuleType("unitree_sdk2py.idl"),
            "unitree_sdk2py.idl.unitree_hg": ModuleType("unitree_sdk2py.idl.unitree_hg"),
            "unitree_sdk2py.idl.unitree_hg.msg": ModuleType("unitree_sdk2py.idl.unitree_hg.msg"),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": dds,
        }
        with mock.patch.dict("sys.modules", modules):
            report = profile_direct_reader(
                "lo",
                sample_count=20,
                timeout_s=1.0,
                maximum_batch_size=8,
                condition_label="offline_mock",
            )
        self.assertTrue(report["success"])
        self.assertEqual(report["observed"]["captured_samples"], 20)
        summary = report["observed"]["summary"]
        self.assertEqual(summary["reader_batches"]["maximum_batch_size"], 3)
        self.assertEqual(summary["source_timestamp"]["unique_count"], 20)
        self.assertTrue(summary["sample_identity"]["identity_pair_is_unique"])
        self.assertTrue(report["safety"]["direct_reader_only"])
        self.assertTrue(report["safety"]["sdk_callback_queue_bypassed"])
        self.assertFalse(report["safety"]["send_channel_created"])
        self.assertFalse(report["decision"]["real_policy_shadow_allowed"])

    def test_validation_fails_before_lazy_dds_import(self):
        for kwargs in (
            {"sample_count": 19, "timeout_s": 10.0},
            {"sample_count": 20, "timeout_s": 0.5},
            {"sample_count": 20, "timeout_s": 10.0, "maximum_batch_size": 0},
            {"sample_count": 20, "timeout_s": 10.0, "condition_label": "bad label"},
        ):
            with self.assertRaises(ValueError):
                profile_direct_reader("enp3s0", **kwargs)

    def test_pinned_cyclonedds_source_attestation(self):
        self.assertEqual(CYCLONEDDS_VERSION, "0.10.2")
        self.assertEqual(
            CYCLONEDDS_SDIST_SHA256,
            "f834962eabbdcdf4e9cd75cf87222f3c5ef22d9cb9e7ed651d9a8710fe984a30",
        )

    def test_offline_readiness_and_paired_plan_keep_hardware_locked(self):
        readiness = json.loads((
            ROOT / "results" / "g1_dds_direct_profiler_offline_readiness_20260828.json"
        ).read_text())
        plan = json.loads((
            ROOT / "results" / "g1_dds_direct_profiler_paired_plan_20260828.json"
        ).read_text())
        self.assertEqual(
            readiness["pinned_sources"]["profiler_sha256"],
            hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        )
        self.assertTrue(
            readiness["source_audit"]["direct_waitset_reader_bypasses_sdk_listener_and_bqueue"]
        )
        self.assertFalse(
            readiness["source_audit"]["adapter_callback_overflow_observes_sdk_bqueue_drop"]
        )
        self.assertFalse(
            readiness["source_audit"]["sample_info_explicit_sequence_number_available"]
        )
        self.assertFalse(readiness["decision"]["hardware_profiler_run_authorized"])
        self.assertFalse(readiness["decision"]["real_policy_shadow_allowed"])
        self.assertEqual(plan["status"], "OFFLINE_PLAN_ONLY_HARDWARE_NOT_AUTHORIZED")
        self.assertEqual(plan["phases"][2]["total_runs"], 20)
        self.assertEqual(plan["phases"][2]["total_requested_samples"], 20000)
        self.assertFalse(plan["qualification_boundary"]["diagnostic_plan_alone_can_pass_h3"])
        self.assertFalse(plan["qualification_boundary"]["robot_motion_allowed"])
        self.assertFalse(plan["safety"]["robot_command_publisher_created"])
        self.assertFalse(plan["hardware_execution_performed"])

    def test_module_has_only_read_side_dds_and_no_process_execution(self):
        tree = ast.parse(MODULE.read_text())
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("subprocess", imports)
        self.assertNotIn("os", imports)
        self.assertIn("DataReader", identifiers)
        forbidden = {
            "ChannelPublisher", "ChannelSubscriber", "DataWriter", "LowCmd_",
            "MotionSwitcherClient", "SportClient", "Write", "Popen", "system",
        }
        self.assertFalse(forbidden & identifiers)
        self.assertFalse(forbidden & attributes)


if __name__ == "__main__":
    unittest.main()
