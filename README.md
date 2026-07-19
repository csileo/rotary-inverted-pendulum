# Rotary Inverted Pendulum

[![Watch the build video](assets/youtube-thumbnail-pendulum-build.jpg)](https://www.youtube.com/watch?v=rKChjuuR7K8)

A DIY rotary inverted pendulum you can print, solder, and train at home — for about **£20** in parts. It's an open, hackable take on the rigs you'd usually buy from a lab-equipment vendor (Quanser's [QUBE Servo 2](https://www.quanser.com/products/qube-servo-2) lists at around £4,500). The pendulum balances itself with a reinforcement-learning policy trained in simulation, fine-tuned on the real hardware, and quantised to run on an Arduino Nano.

## What's in this repo

| Directory                                                           | Contents                                                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [`meshes/`](meshes), [`urdf/`](urdf)                                | 3D-printable STLs and the URDF model (single source of truth for pendulum geometry)              |
| [`diagrams/`](diagrams)                                             | Wiring diagrams and component photos                                                             |
| [`RotaryInvertedPendulum-arduino/`](RotaryInvertedPendulum-arduino) | Firmware — low-level server, hand-tuned PID, on-device RL controller                             |
| [`RotaryInvertedPendulum-python/`](RotaryInvertedPendulum-python)   | Sim env, SAC training, system identification, real-hardware bridge, distillation and int8 export |
| [`RotaryInvertedPendulum-julia/`](RotaryInvertedPendulum-julia)     | MPC/LQR exploration and MeshCat visualisation                                                    |
| [`docs/`](docs)                                                     | Build runbook, BOM, electronics design, RL stack documentation                                   |

## Where to start

- **Build one** — [`docs/BOM.md`](docs/BOM.md), [`docs/electronics_design.md`](docs/electronics_design.md), [`docs/end_to_end_runbook.md`](docs/end_to_end_runbook.md)
- **Train a policy** — [`RotaryInvertedPendulum-python/README.md`](RotaryInvertedPendulum-python/README.md) walks through the simulation and training pipeline
- **Understand the RL stack** — [`docs/rl_transitions.md`](docs/rl_transitions.md), [`docs/domain_randomization.md`](docs/domain_randomization.md), [`docs/transport_delay.md`](docs/transport_delay.md), [`docs/quantisation.md`](docs/quantisation.md), [`docs/sysid_runbook.md`](docs/sysid_runbook.md)

## Prefer to buy rather than build?

DIY kits run [$100–$200 on AliExpress](https://www.aliexpress.com/w/wholesale-rotary-inverted-pendulum.html); the Quanser QUBE Servo 2 mentioned above is around £4,500.

## Related work

- [Desktop Inverted Pendulum, build-its-inprogress](https://build-its-inprogress.blogspot.com/2016/08/desktop-inverted-pendulum-part-2-control.html) ([full series](https://build-its-inprogress.blogspot.com/search/label/Pendulum))
- [Furuta pendulum, dagor.dev](https://www.dagor.dev/blog/furuta-pendulum)
- [The Rotary Control Lab — Quanser brochure (PDF)](https://tecsolutions.us/sites/default/files/quanser/The%20Rotary%20Control%20Lab%20Brochure_4.pdf)
- [Survey paper, *Trans. Inst. Meas. Control*](https://journals.sagepub.com/doi/full/10.1177/00202940211035406)
- Video builds: [[1]](https://www.youtube.com/watch?v=2koXcs0IhOc), [[2]](https://www.youtube.com/watch?v=bY4t6yfBA24), [[3]](https://www.youtube.com/watch?v=VVQ-PGfJMuA)

## Acknowledgments

I would like to thank the following people for their contributions to this project:
- [Joe](https://github.com/spookycouch) for suggesting I try reinforcement learning with Stable Baselines 3, which kicked off the learned-control parts of this project.
- [Mykha](https://github.com/Mika412) for early discussions about this project over a beer in the park.
- [André](https://github.com/Esser50K), [Rafael](https://github.com/rkourdis), and [Vlad](https://github.com/VladimirIvan) for technical discussions, feedback, and support.
- [Vivek](https://github.com/svrkrishnavivek) for his invaluable help and feedback on the electronics of the system.
- [心诺 (Xinnuo)](https://github.com/XinnuoXu) for her company and support while working on this project.

Finally, I would like to thank the open-source community in general for providing the tools and resources that have also helped make this project possible.
