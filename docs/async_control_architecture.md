# Async Control Architecture

How `finetune_async.py` keeps the rig's control loop at strict 100 Hz
(or any configured rate) while SAC's gradient updates run in parallel.
This is a runtime-systems doc — it has no opinions about RL semantics
(see [`rl_transitions.md`](rl_transitions.md) for those) or how to
*choose* a control rate (see [`control_rate_selection.md`](control_rate_selection.md)
for that). It only explains how we *enforce* whatever rate was chosen.

## The bug this exists to prevent

During Phase 4 fine-tuning we discovered that SB3's `SAC.learn()` runs
`collect_rollouts` (which calls `env.step()`) and `train()` (gradient
updates) sequentially in **one thread**. With `--gradient-steps 4`,
gradient updates take ~20 ms per env step. The env's pacing logic
(`time.sleep(next_tick - monotonic())`) silently dropped the configured
100 Hz to ~35 Hz actual. The policy *learned 35 Hz dynamics*. Deployment
at 100 Hz produced motor chatter and step-skipping — same weights,
totally different observed behavior.

`finetune_async.py` is the architectural fix: a custom training loop
(replacing `model.learn()`) that owns two cooperating threads.

## Two threads

```
┌──────────────────────────────────────────┐    ┌──────────────────────────────────┐
│ Control thread (background, strict 100Hz)│    │ Learner thread (main, best effort)│
│                                          │    │                                  │
│ every 10 ms:                             │    │ loop:                            │
│   action ← snapshot.predict(obs)         │    │   transitions ← queue.drain()    │
│   env.apply_action(action)               │    │   for t: replay_buffer.add(t)    │
│   sleep_busywait_until_next_tick()       │ →→ │   if past warmup:                │
│   next_obs ← env.observe_and_step(...)   │    │     model.train(K, batch_size)   │
│   queue.put(transition)                  │    │     snapshot.refresh_from(actor) │
│                                          │    │                                  │
└──────────────────────────────────────────┘    └──────────────────────────────────┘
            │                                                 │
            ↓                                                 ↓
        TransitionQueue (deque + lock) — producer/consumer between threads
                                                              │
                                                              ↓
                                              model.replay_buffer (SB3 ReplayBuffer)
```

The control thread **never blocks on the learner**. The learner **never
blocks on the rig**. They communicate through one shared queue and one
shared replay buffer, both lock-protected.

## Three locks

1. **`LowLevelClient._lock`** (per-serial-transaction). Prevents
   interleaved bytes when the control thread is mid-`get_state` and
   the main thread fires `disengage_motor` for safety.

2. **`replay_buffer_lock`** (orchestrator-owned). Held during
   `replay_buffer.add()` *and* during the entire `model.train(K, batch)`
   call. SAC's `train()` calls `sample()` K times internally; we want a
   frozen view across those calls. In practice contention is zero —
   both contenders are on the main thread; the lock exists for
   future-proofing and audit trail.

3. **`PolicySnapshot._lock`**. Guards `self._net.load_state_dict(...)`
   (called by the learner after `train()`) versus `self._net(obs)`
   (called by the control thread). **Required**: PyTorch's
   `optimizer.step()` mutates parameter `.data` in place, and a forward
   pass concurrent with that write would yield a corrupted action. The
   snapshot is a deepcopy of `model.actor`; the learner's working actor
   is untouched by the control thread.

## Why a snapshot, not the live actor?

If the control thread called `model.actor(obs)` directly, every
inference would race with whichever step of `optimizer.step()` happened
to be running on the learner. Reading partially-updated parameters
gives a corrupted action — fine in a benchmark, dangerous when it
drives a motor. The snapshot adds a one-`state_dict`-copy cost per
train cycle (microseconds) and gives consistent reads.

## Pacing — the busy-wait detail

macOS `time.sleep` typically overshoots by 1-2 ms (no `SCHED_FIFO`).
A naive sleep-the-full-remainder pattern yields a mean rate of ~83 Hz
when you ask for 100 Hz. The fix is two-stage:

```python
sleep_for = next_tick - time.monotonic()
if sleep_for > 0.001:
    time.sleep(sleep_for - 0.001)
while time.monotonic() < next_tick:
    pass    # busy-wait the last ≤ 1 ms
```

Adds <1% CPU at 100 Hz, pulls jitter from ~2 ms down to <0.1 ms.
Standalone V1/V2 tests measure mean dt = 10.000 ms, std < 1 ms.

## Watchdog: the 3-strike timing rule

Inside the tick loop, if the control thread overruns its deadline
by more than `timing_violation_threshold_ms` (default 5 ms) for **3
consecutive** ticks, `AsyncControlLoop` disengages the motor and
raises `TimingViolation`. Three-strike tolerance because macOS
schedulers occasionally hiccup once and recover; three-in-a-row is
structural (probably the learner thread is holding the GIL too long).

This is how future users learn about the bug *quickly* if they
misconfigure things — no more silent rate drops.

When raising, the loop also calls `queue.discard_recent(max_strikes - 1)`
so transitions queued during the violating ticks (with stretched dt)
don't pollute the replay buffer.

## Signal handling

Python only delivers signals to the main thread. The orchestrator
installs `SIGINT`/`SIGTERM` handlers that:

1. Set the shared `stop_flag` (which the control thread polls at the
   top of every tick → exits cleanly).
2. Call `env.disengage_safely()` (which uses `LowLevelClient._lock`
   to safely write the disengage byte even if the control thread is
   mid-read).

The synchronous `RealRotaryInvertedPendulumEnv.__init__` no longer
auto-registers signal handlers — that was load-bearing for the old
sync flow but conflicts with multi-threaded ownership. Direct callers
who want the old behavior can wire `signal.signal(SIGINT, env._on_signal)`
themselves.

## Replay buffer persistence

`finetune_async.py` saves `runs/<run-name>/replay_buffer.pkl` (~6 MB
per 30 k transitions, pickle) at session end and at every checkpoint.
The `--resume-buffer <path>` flag lets a future session load it via
`model.load_replay_buffer`. This means real-robot transitions
**accumulate across sessions** instead of being discarded each time —
critical because real-robot data is ~1000× more expensive to collect
than sim data.

Important: do **not** load buffers from synchronous fine-tunes
(legacy `finetune_real.py`). Those transitions describe `(s, a, s')`
at variable 28-30 ms intervals (see "the rate-mismatch bug" above).
Mixing them with the new strict-rate transitions teaches the critic
contradictory mappings. Same caveat for buffers from a different
control rate — always resume against the same rate the buffer was
collected at.

## Verification protocol

Implemented and run in this order — each step proves a property in
isolation before composing:

- **V1**: control loop alone with no-op learner → mean dt = 10.000 ms,
  std < 1 ms. Proves the control thread alone holds 100 Hz on macOS.
- **V2**: control loop with synthetic CPU-hog learner → identical
  timing. Proves the learner doesn't perturb the control thread via
  the GIL.
- **V3** (rig required): real `model.train(K=4)` with a pre-loaded
  buffer → identical timing. Proves PyTorch + numpy + optimizer step
  doesn't starve the control thread.
- **V4** (rig required): short end-to-end fine-tune (10 episodes × 6 s,
  K=4) — orchestrator-reported mean tick rate within ±2 Hz of target,
  zero violations, redeployed policy doesn't chatter at the same rate.
- **V5** (rig required): full re-baseline — fresh sim curriculum at
  the chosen design rate, then a clean fine-tune. Final policy must
  show sustained balance windows ≥ 5 s in deployment runs.

Per-episode telemetry is logged to `runs/<run-name>/timing.csv`:
`mean_tick_dt_ms`, `std_tick_dt_ms`, `max_overrun_ms`, `n_violations`,
`learner_train_calls`, `learner_total_train_s`. This is the artifact
that proves we don't regress in future PRs.

## Where it lives in code

| File | What it does |
|---|---|
| `async_control.py` | `TransitionQueue`, `PolicySnapshot`, `AsyncControlLoop`. Pure runtime primitives — no SB3 references except for accepting an `nn.Module` to snapshot. |
| `finetune_async.py` | The orchestrator: loads SAC, manages threads + locks, handles signals, persists buffer + policy. |
| `lowlevel_client.py` | `_lock` added so concurrent `get_state` / `disengage_motor` calls don't interleave bytes. |
| `real_env.py` | `apply_action` and `observe_and_step` split out of the synchronous `step` so the async loop can interleave them with externally-paced sleeps. |
