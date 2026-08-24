# Unitree G1 SDK2 read-only integration

## Pinned official source

The project pins the official Python SDK as a submodule:

```text
repository: https://github.com/unitreerobotics/unitree_sdk2_python.git
commit:     65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
path:       third_party/unitree_sdk2_python
license:    BSD-3-Clause
```

The C++ SDK was reviewed at commit
`69c04316d81ac244c314e730106e24787b3e61b8`; it is not required by the
subscriber-only Python path.

## G1 facts supported by the pinned SDK

- G1 uses `unitree_hg`, not the `unitree_go` IDL used by several other robots.
- The official state topic used by G1 examples is `rt/lowstate`.
- The state type is `unitree_hg.msg.dds_.LowState_`.
- The official 29-DOF G1 indices place the left arm at 15–21 and the right arm
  at 22–28. This exactly matches the frozen 14-joint contract order.
- `LowState_` contains version, mode, tick, IMU, 35 motor-state slots, wireless
  remote bytes, reserve fields and CRC.

Names are preserved directly from the official IDL. Units and IMU quaternion
ordering must be verified against the actual robot/firmware before they are
used to construct the policy state.

## Examples that must not be run

The official G1 examples are demonstrations, not read-only diagnostics:

- `example/g1/low_level/g1_low_level_example.py` creates `rt/lowcmd`, releases
  the active motion service and commands ankles/wrists at a 2 ms period.
- `example/g1/high_level/g1_arm7_sdk_dds_example.py` creates `rt/arm_sdk`,
  enables the arm service and commands a multi-stage arm trajectory.
- Loco, arm-action and motion-switcher clients can change robot state.

None of those examples is part of this project's hardware preflight.

## Project adapter

`g1_unitree_lowstate.py` is a separate subscriber-only implementation. It:

- lazily imports only `ChannelFactoryInitialize`, `ChannelSubscriber`, and the
  G1 `LowState_` type;
- fixes the topic to `rt/lowstate`;
- requires an explicit validated network-interface name;
- extracts the 3 waist and 14 arm joints in their frozen FK/contract order;
- validates finite motor and IMU fields;
- records callback timing, tick range and safety evidence;
- closes the reader on success, timeout or exception.

It contains no command message type, send channel, motion switcher, sport
client, mode transition or write call. An AST regression test enforces those
boundaries.

## Staged use

Do not install or run the adapter on the robot merely because the source is
available.

1. **Inventory approval:** identify the existing SDK, Python, CycloneDDS and
   robot-facing interface using read-only shell commands.
2. **Installation review:** if the pinned SDK is absent, create an isolated
   user virtualenv; do not use `sudo`, replace system packages or restart a
   service.
3. **Subscriber approval:** capture a small fixed sample count while the robot
   remains in Damping. Stream evidence back to the operator host when possible.
4. **Semantic review:** verify units, quaternion ordering, active 29-DOF
   variant, joint indices, tick behavior and freshness.
5. **Zero-motion Shadow:** run current-pose FK/IK and safety checks without any
   command path.

Each stage requires a separate result and does not grant hardware execution.

## Yuhao deployment repository is reference-only

The pinned audit of <https://github.com/leihao100/g1-client> at commit
`1422e8d6ef674aa047cfb2878bc7dae54b118fbe` recovers the trained EEF policy
consumer, Pinocchio IK and receding-horizon scheduling reference. It is not a
read-only client:

- `g1_client/arm_controller.py` creates a `ChannelPublisher` for `rt/arm_sdk`
  and writes `LowCmd_`;
- it sets the arm-sdk handover slot, locks body joints, moves to a ready pose
  and streams arm targets;
- `g1_client/gripper_controller.py` publishes Dex1 commands;
- its documented precondition is operator-selected `ai` mode, not Damping.

Therefore `openpi/main_eef.py`, `openpi/main.py` and `openpi/replay.py` must not
be copied to or run on the connected robot in the current read-only phase. Only
static source audit and mock-sink/offline extraction are permitted.
