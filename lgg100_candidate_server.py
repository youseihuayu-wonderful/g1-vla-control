#!/usr/bin/env python3
"""Strict, quarantined OpenPI restore server for LGG100.

The author directly confirmed the core ``pi05_g1_eef`` settings used by this
checkpoint: action horizon 32 and ``discrete_state_input=False``. The complete
historical TrainConfig and exact OpenPI commit remain unpublished, so strict
parameter restoration and all fail-closed execution gates remain mandatory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from pathlib import Path
import socket
from typing import Any

import numpy as np

from g1_policy_contract import contract_metadata, validate_observation

HF_REPO = "LGG100/stack-cube-eef-24k"
HF_REVISION = "cced7a7ff7b454fdcac555457a1a2a3dc262ac77"
ASSET_ID = "stack-cube-eef"
OPENPI_AUDITED_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_FINTUNE_AUDITED_COMMIT = "29030046fd6a2810201db67b9804f243e0af3218"
AUTHOR_MODEL_CONFIG_NAME = "pi05_g1_eef"
AUTHOR_ACTION_HORIZON = 32
EXPERIMENTAL_ACTION_HORIZON = 48
AUTHOR_DISCRETE_STATE_INPUT = False
DEFAULT_PROMPT = (
    "Stack the blocks by color: put the red block in the center, then stack "
    "the blue block on the red block, then stack the yellow block on the blue block."
)
OBSERVATION_KEYS = (
    "observation/cam_left_high",
    "observation/cam_left_wrist",
    "observation/cam_right_wrist",
)


def _parse_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key} must be HxWx3, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{key} contains NaN/Inf")
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class G1CandidateInputs:
    """Candidate inference mapping reconstructed from public metadata."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        validate_observation(data)
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (16,) or not np.all(np.isfinite(state)):
            raise ValueError(f"observation/state must be finite [16], got {state.shape}")
        images = {
            "base_0_rgb": _parse_image(data[OBSERVATION_KEYS[0]], OBSERVATION_KEYS[0]),
            "left_wrist_0_rgb": _parse_image(data[OBSERVATION_KEYS[1]], OBSERVATION_KEYS[1]),
            "right_wrist_0_rgb": _parse_image(data[OBSERVATION_KEYS[2]], OBSERVATION_KEYS[2]),
        }
        output: dict[str, Any] = {
            "state": state,
            "image": images,
            "image_mask": {name: np.True_ for name in images},
        }
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class G1CandidateOutputs:
    """Publish only the 16 robot dimensions from the 32-wide OpenPI head."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise ValueError(f"Expected rank-2 actions with >=16 dims, got {actions.shape}")
        actions = actions[:, :16]
        if not np.all(np.isfinite(actions)):
            raise ValueError("Model returned NaN/Inf actions")
        return {"actions": actions}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_policy(checkpoint_dir: Path, action_horizon: int, default_prompt: str):
    """Load the real checkpoint with no extra-parameter intersection."""
    import jax
    import jax.numpy as jnp

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.policies import policy as policy_lib
    from openpi.training import checkpoints
    from openpi.training import config as config_lib

    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"LGG100 restore requires CUDA for this gate; JAX backend is {jax.default_backend()!r}"
        )
    required = (
        checkpoint_dir / "_CHECKPOINT_METADATA",
        checkpoint_dir / "params" / "_METADATA",
        checkpoint_dir / "assets" / ASSET_ID / "norm_stats.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint snapshot; missing {missing}")
    if action_horizon not in (AUTHOR_ACTION_HORIZON, EXPERIMENTAL_ACTION_HORIZON):
        raise ValueError(
            "action_horizon must be the author-confirmed 32 or explicitly experimental 48"
        )

    # Core config name, horizon=32, and continuous state input were confirmed
    # directly by the author. Architecture details are additionally supported
    # by the checkpoint parameter tree and the public pi05_g1_eef config.
    model_config = pi0_config.Pi0Config(
        dtype="bfloat16",
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=action_horizon,
        max_token_len=200,
        pi05=True,
        discrete_state_input=AUTHOR_DISCRETE_STATE_INPUT,
    )
    train_config = config_lib.TrainConfig(
        name="lgg100_stack_cube_eef_author_core_restore",
        model=model_config,
        data=config_lib.SimpleDataConfig(
            assets=config_lib.AssetsConfig(asset_id=ASSET_ID),
            data_transforms=lambda _: transforms.Group(
                inputs=[G1CandidateInputs()], outputs=[G1CandidateOutputs()]
            ),
            base_config=config_lib.DataConfig(prompt_from_task=True),
        ),
    )

    # Do not use create_trained_policy here: BaseModel.load defaults to
    # remove_extra_params=True.  This gate explicitly rejects missing *or extra*
    # leaves so a nearby architecture cannot silently discard checkpoint weights.
    params = model_lib.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = model_config.load(params, remove_extra_params=False)
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats = checkpoints.load_norm_stats(checkpoint_dir / "assets", ASSET_ID)
    policy = policy_lib.Policy(
        model,
        transforms=[
            *data_config.repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *data_config.repack_transforms.outputs,
        ],
        metadata={
            **contract_metadata(verified=True),
            "evidence_mode": "lgg100_pinned_author_contract_strict_restore",
            "neural_checkpoint_loaded": True,
            "strict_parameter_tree_restore": True,
            "author_core_config_directly_confirmed": True,
            "complete_author_train_config_available": False,
            "author_confirmation_provenance": "direct_communication_reported_by_project_owner",
            "author_model_config_name": AUTHOR_MODEL_CONFIG_NAME,
            "g1_action_compatible": True,
            "official_consumer_quaternion_postprocessing_required": True,
            "author_deployment_commit": "1422e8d6ef674aa047cfb2878bc7dae54b118fbe",
            "safe_for_g1_hardware": False,
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "openpi_audited_commit": OPENPI_AUDITED_COMMIT,
            "openpi_fintune_audited_commit": OPENPI_FINTUNE_AUDITED_COMMIT,
            "restored_model_config": dataclasses.asdict(model_config),
            "published_action_dim": 16,
            "internal_action_dim": 32,
            "action_horizon_author_confirmed": action_horizon == AUTHOR_ACTION_HORIZON,
            "experimental_action_horizon": action_horizon == EXPERIMENTAL_ACTION_HORIZON,
            "discrete_state_input_author_confirmed": True,
            "checkpoint_metadata_sha256": _sha256(checkpoint_dir / "_CHECKPOINT_METADATA"),
            "norm_stats_sha256": _sha256(checkpoint_dir / "assets" / ASSET_ID / "norm_stats.json"),
        },
    )
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=AUTHOR_ACTION_HORIZON)
    parser.add_argument("--default-prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--allow-candidate-restore",
        action="store_true",
        help="required acknowledgement that the complete historical TrainConfig remains unavailable",
    )
    args = parser.parse_args()
    if not args.allow_candidate_restore:
        raise SystemExit(
            "Refusing to restore silently. Re-run with --allow-candidate-restore after reading "
            "LGG100_REAL_VLA.md. This still never enables G1 hardware execution."
        )
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    policy = build_policy(checkpoint_dir, args.action_horizon, args.default_prompt)

    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    print(
        f"Strict LGG100 candidate restore succeeded on {socket.gethostname()}; "
        f"serving output-only policy on {args.host}:{args.port}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
