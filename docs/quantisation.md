# On-device weight quantisation

Notes on quantising the distilled student MLP for the Arduino Nano. The
broader rationale (when to bother, end-to-end recipe, code-change scope)
is in [`policy_improvement_ideas.md`](policy_improvement_ideas.md) under
"On-device performance"; this file captures the **format choice** and
the techniques that actually got int8 to work.

The Arduino Nano runs an **AVR** ATmega328P. AVR is the family of 8-bit
RISC microcontrollers originally designed by Alf-Egil Bogen and Vegard
Wollan (hence "AVR" — *Alf and Vegard's RISC*) and currently sold by
Microchip Technology after the Atmel acquisition. AVR has no FPU, so
the choice of weight format is dominated by which integer multiplications
the silicon can do natively.

## int8 vs int16 on AVR

Speed and fidelity numbers:

| Format | Bytes/weight | Per-MAC cost | Quantisation granularity | Forward pass (1216 MAC) |
|---|---|---|---|---|
| float32 | 4 | ~190 cycles (MUL + ADD, software) | ~7 decimal digits | ~14.5 ms |
| **int8** | **1** | **~5 cycles** (1× MUL + acc) | **max(W) / 127** | **~0.4 ms (~36× faster)** |
| int16 | 2 | ~15 cycles (4× MUL + accumulate) | max(W) / 32767 (~256× finer than int8) | ~1.1 ms (~13× faster) |

AVR has hardware `MUL` (8×8→16) and `MULS` (signed 8×8) but **no 16×16
multiplier** — int16 multiplication is composed from four 8×8 MULs plus
carries. So int16 is 3× slower per MAC than int8 even though the number
of bytes only doubles.

## Decision

**Use int8.** The hardware path is direct (a single `MULS` instruction
per weight×activation product, two cycles), and the resulting 36×
speedup over float32 is the whole reason to do quantisation in the first
place. Quantisation-aware training (QAT) plus the techniques below
closes the fidelity gap to negligible — this is the same recipe the
embedded-NN ecosystem (TFLite Micro, ONNX Runtime Mobile, MobileNet
papers) standardises on.

**int16 is the fallback** if int8 ever can't be made to hit the
closed-loop balance behaviour of the float student. int16 with even
post-training quantisation is essentially fidelity-equivalent to float
(256× finer granularity than int8), so it's the safety net — but at
3× the per-MAC cost, it shouldn't be the default.

## What it took to make int8 actually work on this rig

A naive int8 + QAT pipeline (per-tensor scaling everywhere, weights and
activations both in int8 range, no special handling of biases or input
heterogeneity) gets stuck at **~0.70 mean upright** on this rig — the
policy *catches* the pendulum reliably but can't *hold* it, drifting
out of the calm attractor within a few seconds. Closing that gap to
0.951 required layering several techniques on top of vanilla QAT. None
are exotic; they're all canonical "make int8 work" tricks.

### 1. Quantisation-aware training (QAT)

Train the model with simulated int8 rounding inserted into the forward
pass during training. The optimiser then learns weights that are
*robust to the rounding errors int8 introduces*, instead of being
surprised by them at deploy time.

The simulation is "fake-quant + straight-through estimator": forward
pass rounds activations and weights to the int8 grid, backward pass
pretends the rounding was identity so gradients flow through normally.

### 2. Per-channel input quantisation

The five obs dimensions have wildly different ranges:

| Dim | Range | LSB at shared scale | LSB with per-channel |
|---|---|---|---|
| `motor_pos` | ±2.2 rad | 0.23 (12 distinct values) | 0.014 (164 values) |
| `sin(theta)` | ±1.0 | 0.23 (8 distinct values) | 0.008 (256 values) |
| `cos(theta)` | ±1.0 | 0.23 (8 distinct values) | 0.008 (256 values) |
| `motor_vel` | ±15 rad/s | 0.23 | 0.029 |
| `pen_vel` | ±30 rad/s | 0.23 | 0.229 |

A *shared* obs scale, set by the largest-range dim (`pen_vel` reaching
±30 rad/s during swing-up), crushes `sin(theta)` and `cos(theta)` to
4 distinct int8 values right at the equilibrium where they need maximum
precision. Giving each obs dim its **own** scale recovers ~30× finer
LSB on the small-range dims without losing range on the big-range dims.
This was the single biggest unlock — it's the difference between
"can't hold balance" and "balances".

### 3. Per-row weight quantisation

For each Linear layer, give each output neuron's row of weights its
*own* scale rather than sharing a single scale across the whole weight
matrix. Different output neurons have different weight magnitudes, and
forcing them to share one scale wastes precision on the rows whose
maximum is far below the global max. Standard TFLite-Micro pattern.

The rescale to the next layer's int8 input becomes per-output-channel
too — instead of one fixed-point multiplier `M_q15`, the Arduino has
an array of them, looked up per output neuron during the matmul.

### 4. Bias quantisation to int32, not int8

Biases live in the int32 *accumulator* on the deployed path, where they
get added to `sum_j (W_int8[i,j] × x_int8[j])` before rescale. The
"step size" the bias rounds to is `s_w × s_x` (the accumulator unit),
which is much smaller than 1 — typical bias values are integer
multiples of `s_w × s_x` in the *thousands*.

The first attempt used the same int8 fake-quant for biases as for
weights, which **clamped them to ±127 × s_w × s_x**. For typical layer
3 with `s_w × s_x ≈ 6e-4`, a bias of 0.1 → quantised to 170 → clamped
to 127 → loses 25% of its value. Catastrophic. The fix is a separate
"int32 fake-quant" that rounds to the grid but doesn't clamp. (We use
±2³⁰ as a safety bound; biases are O(10²) at worst.)

### 5. Skip pre-tanh quantisation

The output layer's activation goes through `tanh`. The first attempt
quantised the pre-tanh value to int8 too (matching what a hypothetical
LUT-based deploy would do). But the deploy path doesn't use a LUT — it
dequantises the int32 accumulator straight to float and calls libm
`tanhf`. So QAT shouldn't quantise pre-tanh either; doing so trains
the weights to compensate for a precision loss the deploy never sees,
which makes things worse.

`tanh` is called once per inference on the AVR (~200 cycles) so a
float-based final layer is essentially free.

### 6. Layer-1 absorbing

Per-channel input quantisation (trick 2) gives each obs dim its own
scale `s_obs[j]`. Per-row weight quantisation (trick 3) gives each
output row its own scale `s_w[i]`. Combined, the matmul `y = Wx + b`
expands to:

```
y[i] = sum_j (s_w[i] × W_int[i,j]) × (s_obs[j] × x_int[j])
     = sum_j  s_w[i] × s_obs[j] × W_int[i,j] × x_int[j]
```

Each term has a *different* scale (`s_w[i] × s_obs[j]`), so we can't
factor a common multiplier and apply it once after the sum. That would
break the simple "int matmul + per-row rescale" pattern that makes int8
fast on AVR.

The trick: at *export time* (not at runtime), pre-multiply the weights
by the per-channel input scales:

```
W_eff[i,j] = W[i,j] × s_obs[j]                ← columns stretched by their input scale
W_eff_int[i,j] = round(W_eff[i,j] / s_w_eff[i])   ← then per-row quantise
```

Then the Arduino runs ordinary int8 matmul + per-row rescale:

```
accum[i] = sum_j  W_eff_int[i,j] × x_int[j]
y[i]     = accum[i] × s_w_eff[i]
```

Same answer, but the per-channel input scales have been "absorbed" into
the weight matrix. The Arduino never sees `s_obs` directly at runtime;
that information is folded into `W_eff_int` at compile time. Free at
runtime, just a bit of extra work in `export_weights_quantised.py`.

## Final result

End-to-end on the rig with all six tricks layered together:

| Variant | val_mse | Mean upright (tethered) |
|---|---|---|
| Float H=16 (production reference) | 0.040 | 0.946 |
| Naive int8 H=16 (per-tensor everything) | 0.080 | 0.706 |
| Naive int8 H=32 (more capacity, same noise) | 0.088 | 0.696 |
| **Per-channel int8 H=16** (with all tricks) | **0.045** | **0.934** |
| **Per-channel int8 H=32** (with all tricks) | **0.040** | **0.951** |

**Production:** int8 H=32 with QAT + tricks 1–6, scoring 0.951 mean
upright tethered — within trial-to-trial noise of the float baseline,
at ~0.4 ms inference on the AVR (vs ~15 ms float). 1.5 KB of weights
(4× smaller than float). On-device int8 is the standalone-deployment
path going forward. The float build (`#define POLICY_QUANTISED`
unset) is preserved as the comparison reference.

One observable difference at deploy: the int8 student tends to land at
the "active correction" attractor (small but persistent motor
oscillations during balance) while the float student more often lands
at the "calm" attractor (motor essentially stationary at a fixed
offset). Both balance equivalently well; the choice is set by SAC
training noise — see [`control_rate_selection.md`](control_rate_selection.md)
for the attractor split's underlying dynamics.

## Why bother — what int8 actually unlocks

Fair question once we've seen them side-by-side: float H=16 already
balances at 0.95 with calmer motor behaviour, so what does the 36×
faster int8 path buy on *this* rig?

For the immediate task (balance + swing-up at 35 Hz on the markovian
obs we have), almost nothing — float H=16 has plenty of latency
margin and the calm attractor it tends to land in is qualitatively
nicer than the int8 student's active correction. The two practical
reasons quantisation matters here anyway:

### 1. Bigger inputs and richer architectures aren't gated by latency

The float H=16 student takes ~5 ms per inference, well inside the
28.6 ms control-tick budget. Most expansions we might want push
that number close to or past the budget:

| Architecture change | Float inference | Int8 inference |
|---|---|---|
| Frame stacking (N=4 → 20-dim obs, MLP H=32) | ~7 ms | ~0.3 ms |
| GRU H=16 (recurrent layer) | ~25 ms | ~1 ms |
| MLP H=64 (richer feedforward) | ~50 ms | ~1.5 ms |

**Honest caveat about the upside**: we can't claim any of those
expansions *would* improve this rig. The current float student
already scores 0.95+ tethered, which is plausibly bumping against
the hardware noise floor (12-bit AS5600 ≈ 0.09° resolution,
AccelStepper microstep quantisation ≈ 0.225°, bearing transients).
A GRU *could* learn to self-calibrate encoder drift, but the rig
doesn't visibly suffer from it. Frame stacking *could* smooth
velocity estimates, but the obs already includes filtered velocity.
Whether either gives a measurable performance gain is hopeful, not
proven.

So the honest framing: int8 is **infrastructure that doesn't
preclude future experiments**, not infrastructure that guarantees
gains today. The day we want to try frame stacking (~one day of
work) or a GRU (multi-day framework migration since SB3 doesn't
ship recurrent SAC — would need CleanRL or Tianshou), the inference
budget will already be there. Without int8, both experiments are
gated by inference time even on this microscopic policy.

### 2. Educational / portfolio value

A working int8-on-AVR pipeline is the canonical mobile-ML deployment
recipe in miniature — a project credential and a natural climax for
the "how far down does this rabbit hole go" arc.

## Why anyone quantises at all (broader context)

For *this* rig, the int8 speedup is the rare case where the benefit
shows up directly in compute time, because the ATmega328P has no FPU
— every float multiply is software-emulated, so int8's hardware
multiply path is roughly 36× faster. On modern hardware (a phone, a
server GPU, even a Raspberry Pi), float math is fast and the speedup
looks much more subtle. The technique is *everywhere* in production
ML anyway, because three other forces start to dominate:

- **Memory bandwidth, not compute, is the bottleneck on accelerators.**
  An A100 can do ~10 TFLOPS of fp32 but only fetches ~2 TB/s from
  HBM. A 70 B-parameter LLM at fp32 (280 GB) takes ~140 ms *just to
  load the weights through memory once*. At int4 (35 GB) the same
  fetch is ~17 ms. The arithmetic doesn't get cheaper, but feeding
  the silicon does — and that's almost always the binding constraint.
- **Energy and battery life.** An int8 multiply uses roughly 4× less
  energy than an fp32 one (silicon area + switching activity). For
  server fleets serving billions of inferences a day, that's millions
  of pounds of electricity. For battery-powered devices it's the
  difference between an hour and a day of usable inference.
- **Specialised hardware paths.** NVIDIA Tensor Cores, Apple's Neural
  Engine, Google TPUs all have *much* faster int8 / int4 / fp8
  pipelines than fp32 — typically 4–8× the throughput. The whole
  reason on-device LLMs (phones running Llama-3, MacBooks running
  Mistral) exist is that quantisation maps the model into the
  silicon's fast paths. Without int4, a 70 B model wouldn't fit on
  a 64 GB MacBook in the first place.

So on a tiny MCU the win is **compute time** (no FPU); on a phone
it's **power and memory size**; in the cloud it's **$$/token and
latency**. The technique is the same; what binds varies with the
hardware:

| Where it runs | What's actually being saved | Why people quantise |
|---|---|---|
| **Arduino Nano (us)** | Software-float multiply cycles | Compute is the binding constraint |
| **Phone running Llama-3** | RAM (4× smaller weights) + Neural Engine throughput | A 70 B fp32 model = 280 GB. Doesn't fit. Period. |
| **Cloud serving GPT-class** | Memory bandwidth (HBM fetches), $$/token | int4/int8 → 4–8× more tokens per GPU-hour |
| **Edge device (drone, doorbell, watch)** | Battery life | int8 multiply ≈ 4× less energy than fp32 |

Our rig is the rare case where the benefit shows up most directly —
but the same recipe (QAT, per-channel scales, weight absorption) is
what runs Llama-3 on a phone, just at a much larger scale.
