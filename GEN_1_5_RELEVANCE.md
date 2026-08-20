# GEN-1.5 relevance to the G1 faster-grasp project

## Source and evidence boundary

This review refers to **Generalist AI's GEN-1.5**, announced on 2026-08-19:

- Official post: <https://generalistai.com/blog/gen-1.5>
- Title: *GEN-1.5: Embodied Foundation Models are One-Shot Learners*
- Retrieved source SHA-256:
  `a999de7b30c0afb29ec230e700d10f3d6877be378648dbcbd78c8e26319d1452`

The release is currently a company research post, not a reproducible model
release. The post does not provide model weights, code, an inference API,
architecture dimensions, action schema, supported robot list, deployment
requirements, latency, license, or a Unitree G1 integration. Reported results
should therefore be treated as first-party claims until independently
reproduced.

## What is actually claimed

GEN-1.5 is described as a large multimodal robot foundation model that:

- consumes video with a 30-second memory, plus other sensors, language and
  proprioception;
- produces 100 Hz action trajectories;
- uses 3–12 seconds of one demonstration as an in-context “physical prompt”;
- can compose two independently recorded prompts into a longer behavior;
- can use a simulation rollout as a prompt for a real-world task despite no
  simulation data in pretraining;
- sometimes transfers a human hand demonstration to robot hands;
- adapts with 1–10 gradient steps on 1–5 minutes of data;
- demonstrates improvisation, error recovery, ambidexterity and novel tool use.

Across 10 company-selected, simple, short-horizon tasks, the post reports:

- 59% ±10% average success from one 3–12 second in-context demonstration;
- 83% ±9% after 10 gradient steps on five minutes/~50 demonstrations;
- 66.5% on one held-out task after one gradient step on one minute of data.

The authors explicitly say that one-shot skills are more brittle than
fine-tuned policies, tasks are simple/short-horizon, and success rates remain
modest. The “GPT-3 moment” is an analogy to emergent in-context learning after
large-scale pretraining; it is not proof of general physical intelligence.

## What this teaches our project

### 1. Treat demonstrations as runtime context, not only training data

Our current LGG100 path receives the current observation and instruction. A
GEN-1.5-style system additionally retains a synchronized sensorimotor example
inside a long context window. The immediate actionable lesson is to improve
our data recorder so every demonstration contains aligned video, LowState,
EEF, Dex1, action, contact/phase and timestamps, and can be replayed as a
3–12-second context segment.

This does not mean adding an arbitrary demonstration to LGG100. LGG100 was not
trained for physical prompting; doing so would be an unsupported semantic
change. The prompt buffer should first be an offline dataset/replay artifact.

### 2. Simulation rollouts may become a prompt library

GEN-1.5 claims that a simulation rollout can prompt a real robot without
training on that task. This is directly relevant to our MuJoCo work: safe
far/approach/grasp/lift segments could become reusable physical prompts for a
future model that officially supports the interface.

However, our current simulation is not yet a valid prompt source: the
Adaptive-OFF task has zero completed cycles, IK is blocked, and physical
calibration is missing. Only successful, contract-matched, collision-free
rollouts should enter a prompt library.

### 3. Separate policy context rate from actuator/control rate

GEN-1.5 says it produces 100 Hz action trajectories. This should not be read as
100 Hz end-to-end neural inference. It suggests a useful architecture:

- a slower multimodal policy produces a trajectory/context-conditioned plan;
- a local 100 Hz (or officially validated Unitree rate) controller tracks a
  safety-filtered prefix using fresh feedback;
- stale policy output never bypasses the local watchdog.

Our current 30 Hz policy contract and 119.94 ms commit age should therefore be
split from the future hardware command loop rather than forcing remote VLA
inference to be the motor servo.

### 4. Prompt composition is useful but increases path risk

Composing short prompts could support longer tasks such as approach → grasp →
lift → place without collecting every combination. It also means the model may
invent transitions, regrasp or switch hands. Every generated transition must
still pass sequential IK, swept collision, contact allowlists and freshness
checks. Emergent behavior raises the value of our safety governor; it does not
make it obsolete.

### 5. Optimize adaptation data around phases and failures

For the faster-grasp objective, future demonstrations should explicitly mark:

- free-space travel;
- approach/alignment;
- pre-contact and grasp closure;
- stable lift/place;
- slip, collision, failed grasp and recovery;
- timestamps and achieved completion time.

This supports both physical prompting and few-step adaptation, while allowing
Adaptive-OFF/ON comparisons to remain phase-matched and safety-auditable.

### 6. Generality claims need a harder evaluation protocol

A model that improvises can find faster strategies, but it can also create
novel contacts that were absent from training. We should add evaluation axes
beyond success rate:

- first-attempt success and completion time;
- forbidden contact and peak force;
- IK/collision rejection rate;
- recovery success after controlled perturbations;
- prompt sensitivity and repeated-trial variance;
- latency and stale-action behavior;
- held-out objects, poses, lighting and camera perturbations.

## Compatibility with the current system

No drop-in compatibility can be established. The official post does not
specify GEN-1.5's action dimension, joint/EEF semantics, coordinate frames,
camera layout, normalization, action horizon, robot controller, checkpoint
format, hardware requirements or API. It cannot currently replace LGG100 or be
connected to the Unitree SDK path.

Even if access becomes available, it would enter as a quarantined policy
candidate behind the same contract and safety boundaries:

1. obtain official API/model/schema/license documentation;
2. perform output-only schema and latency audit;
3. create a versioned adapter without reinterpreting unknown actions;
4. evaluate one-shot prompts offline on saved MuJoCo/real demonstrations;
5. require the same IK, collision, watchdog and task-success gates;
6. run MuJoCo Adaptive-OFF before any hardware consideration.

## Decision

GEN-1.5 is strategically relevant because it shifts the research question
from “How many task-specific episodes are needed?” to “Can a safe robot learn
from one short sensorimotor example?” The current project decision is to retain
this as a research record only. No GEN-1.5-specific demonstration/context
infrastructure is planned, and it is not part of the execution roadmap.

It does **not** change the immediate blocker order: complete IK regression,
obtain a stable Adaptive-OFF simulation baseline, reach the 100 ms commit Gate,
validate real LowState/cameras/calibration, and complete zero-motion Shadow/HIL.
No GEN-1.5 claim grants real-robot motion authority.
