# Connection Runbook

This file records connection topology and recovery commands only. It must never
contain passwords, private-key material, access tokens, or robot command
credentials.

## Herdr L40S pane

- Preferred pane: `w1:p3` (pane IDs can change after a new Herdr session)
- Label: `l40-ssh`
- Reconnect command: `shihua-vla-connect`
- Direct SSH alias: `ssh shihua-vla`
- Expected prompt: `(base) user1@nnmc65:~/workspace/shihua$`

### Route

```text
Mac
  -> shihua-vla-jump (test@10.188.57.209:22)
  -> shihua-vla      (user1@10.145.87.65:22; hostname nnmc65)
```

Both aliases use `~/.ssh/id_ed25519_shihua_vla`, `ControlMaster=auto`,
`ControlPersist=12h`, 15-second keepalives, and fail-closed host-key checking.
The private key itself is not part of this repository.

### Health check

```bash
shihua-vla-connect --check
```

Equivalent individual checks:

```bash
ssh -O check shihua-vla-jump
ssh -O check shihua-vla
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

For a non-interactive remote command, override the configured login command:

```bash
ssh -o RemoteCommand=none -o RequestTTY=no -T shihua-vla 'hostname'
```

## LGG100 policy tunnel

The target alias establishes this loopback-only forward:

```text
Mac 127.0.0.1:8000 -> nnmc65 127.0.0.1:8000
```

A local SSH listener proves only that the tunnel exists. Before running a
policy client, separately verify that the quarantined server is listening on
remote port 8000 and that its strict metadata/checkpoint gates pass. Never bind
the policy server to `0.0.0.0`.

## Remote project paths

```text
workspace:  /home/user1/workspace/shihua
repository: /home/user1/workspace/shihua/projects/g1-vla-control
venv:       /home/user1/workspace/shihua/.venvs/g1-sim
results:    /home/user1/workspace/shihua/results
logs:       /home/user1/workspace/shihua/logs
```

Remote long jobs must use `setsid`, a PID file, an independent log, and a
status JSON. They must not depend on the lifetime of an interactive Herdr/SSH
pane. Never terminate or alter another workload to free a GPU.

## Mandatory next real-G1 connection interlock

Before any new LowState, camera, DDS, policy, or Shadow work, read
`G1_NEXT_CONNECTION_INTERLOCK_CN.md`. The previous camera-only process received
a stop request, but termination could not be verified after network loss.

The first connected operation must verify the saved PID identity, stop only a
matching `g1_camera_server_readonly.py` process if it remains alive, and prove
that ports `55555/55556/55557/60000` are closed. No later step is eligible until
a result records `cleanup_verified=true`.

Completion note (2026-08-28): this interlock passed. The robot had rebooted, the
saved PID was stale and not alive, no matching process or listener remained,
and `results/g1_next_connection_cleanup_20260828.json` records
`cleanup_verified=true`. This clears only the reconnect cleanup blocker; H2/H3,
Policy Shadow, and motion remain locked.

## Robot connection is separate

The authenticated read-only robot route is currently:

```text
Mac -> yixiao@192.168.1.13 -> unitree@192.168.123.164 (unitree-g1-nx)
```

It belongs in the `robot-readonly` pane and must not be confused with the L40S
pane. An SSH connection or policy tunnel never grants robot motion authority;
`g1_execution_enabled` remains false until all registered hardware gates pass.
