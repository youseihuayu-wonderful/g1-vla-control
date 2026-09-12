import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from whole_body_speed_extension import (
    WholeBodyContext,
    WholeBodySpeedConfig,
    decide_whole_body_speed,
    nominal_phase_scale,
)


class WholeBodySpeedExtensionTests(unittest.TestCase):
    def safe_context(self, **overrides):
        values = {
            "body_phase": "UPRIGHT",
            "arm_task_phase": "free_space",
            "rl_controller_active": True,
            "support_confirmed": True,
            "estop_ready": True,
            "h_gates_resolved": 6,
            "h_gates_total": 6,
            "torque_margin": 0.8,
            "pelvis_stability": 0.95,
            "body_height_m": 0.75,
            "body_pitch_rad": 0.0,
        }
        values.update(overrides)
        return WholeBodyContext(**values)

    def test_upright_free_space_can_accelerate_only_when_all_gates_are_clear(self):
        decision = decide_whole_body_speed(self.safe_context())

        self.assertFalse(decision.hold)
        self.assertTrue(decision.execution_authorized)
        self.assertGreater(decision.target_scale, 1.0)

    def test_unresolved_hardware_gates_fail_closed(self):
        decision = decide_whole_body_speed(self.safe_context(h_gates_resolved=1))

        self.assertTrue(decision.hold)
        self.assertFalse(decision.execution_authorized)
        self.assertEqual(decision.target_scale, 0.0)
        self.assertIn("h_gates_unresolved", decision.reasons)

    def test_squat_reach_phases_are_conservative_and_payload_caps_standing(self):
        config = WholeBodySpeedConfig()

        lower = nominal_phase_scale("LOWER", "approach", config=config)
        reach = nominal_phase_scale("REACH", "grasp", config=config)
        rise_payload = nominal_phase_scale(
            "RISE", "lift", payload_estimated_kg=0.2, config=config
        )

        self.assertLess(lower, 1.0)
        self.assertLess(reach, lower)
        self.assertLessEqual(rise_payload, config.payload_scale_cap)

    def test_balance_critical_contact_holds(self):
        decision = decide_whole_body_speed(self.safe_context(
            body_phase="REACH",
            arm_task_phase="grasp",
            body_height_m=0.63,
            body_pitch_rad=0.2,
            contact=True,
        ))

        self.assertTrue(decision.hold)
        self.assertIn("contact_during_balance_critical_phase", decision.reasons)


if __name__ == "__main__":
    unittest.main()
