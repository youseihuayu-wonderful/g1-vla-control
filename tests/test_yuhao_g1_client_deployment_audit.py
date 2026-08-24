import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "yuhao_g1_client_deployment_audit_20260824.json"
ATTESTATION = ROOT / "results" / "lgg100_author32_semantic_source_attestation_20260824.json"
CONTRACT = ROOT / "g1_policy_contract.yaml"
DOC = ROOT / "YUHAO_G1_CLIENT_DEPLOYMENT_AUDIT_CN.md"
Q0 = ROOT / "results" / "lgg100_slurm_q0_output_only_20260824.json"


class YuhaoG1ClientDeploymentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads(AUDIT.read_text())
        cls.attestation = json.loads(ATTESTATION.read_text())
        cls.contract = json.loads(CONTRACT.read_text())
        cls.doc = DOC.read_text()
        cls.q0 = json.loads(Q0.read_text())

    def test_pins_training_and_deployment_sources(self):
        self.assertEqual(
            self.audit["audited_commit"],
            "1422e8d6ef674aa047cfb2878bc7dae54b118fbe",
        )
        self.assertEqual(self.contract["contract_version"], "1.3.0")
        policy = self.contract["production_policy"]
        self.assertEqual(policy["deployment_commit"], self.audit["audited_commit"])
        self.assertTrue(policy["official_action_consumer_postprocessing_verified"])

    def test_checkpoint_is_trained_and_consumer_normalization_is_explicit(self):
        model = self.audit["model_and_action_contract"]
        self.assertTrue(model["trained_eef_openpi_policy_explicitly_supported"])
        self.assertFalse(model["raw_predicted_quaternion_guaranteed_unit_norm"])
        self.assertTrue(model["consumer_normalizes_quaternion_before_ik"])
        self.assertTrue(self.audit["decision"]["checkpoint_is_trained"])
        self.assertFalse(self.audit["decision"]["retraining_required"])
        self.assertTrue(self.attestation["g1_contract_verified"])
        self.assertFalse(self.attestation["g1_sim_eligible"])

    def test_q0_reinterpretation_preserves_raw_metric_without_blocking(self):
        probe = self.q0["output_only_probe"]
        self.assertEqual(probe["raw_quaternion_exact_unit_passes"], 0)
        self.assertEqual(probe["official_consumer_postprocess_passes"], 30)
        self.assertFalse(self.q0["decision"]["quaternion_gap_blocks_adaptive_off"])
        self.assertFalse(
            self.q0["interpretation_revision"]["raw_artifact_bytes_modified"]
        )

    def test_live_publishers_are_explicitly_prohibited(self):
        risk = self.audit["publisher_risk"]
        self.assertTrue(risk["contains_channel_publisher"])
        self.assertTrue(risk["contains_lowcmd"])
        self.assertFalse(risk["safe_to_run_in_current_damping_readonly_session"])
        self.assertFalse(self.audit["decision"]["g1_execution_enabled"])
        for entrypoint in ("main_eef.py", "main.py", "replay.py"):
            self.assertIn(entrypoint, self.doc)

    def test_deployment_ambiguity_and_safety_gaps_remain_visible(self):
        loop = self.audit["deployment_loop"]
        self.assertEqual(loop["control_hz_code_default"], 15.0)
        self.assertEqual(loop["control_hz_help_text_claim"], 30.0)
        self.assertFalse(loop["control_hz_resolved"])
        ik = self.audit["ik_reference"]
        self.assertFalse(ik["warning_is_fail_closed"])
        self.assertFalse(ik["orientation_acceptance_threshold_present"])
        self.assertFalse(ik["swept_path_collision_preflight_present"])


if __name__ == "__main__":
    unittest.main()
