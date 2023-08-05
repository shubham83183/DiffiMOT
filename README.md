# DiffiMOT:End-to-End Differential Multi-Object Tracking using Transformers

This repository provides an improvement over the official implementation of the [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702) paper by [Tim Meinhardt](https://dvl.in.tum.de/team/meinhardt/), [Alexander Kirillov](https://alexander-kirillov.github.io/), [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/) and [Christoph Feichtenhofer](https://feichtenhofer.github.io/). The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw).

<!-- **As the paper is still under submission this repository will continuously be updated and might at times not reflect the current state of the [arXiv paper](https://arxiv.org/abs/2012.01866).** -->

## Installation

For installation follow the same installation procedure given by Trackformer repo. Refer to  [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train DiffiMot

The training procedure is also similar to Trackformer. Refer to [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Evaluate DiffiMot

### MOT17

#### Private detections

```
python src/track.py with reid

```
## For changing lambda value 
Refer the file src/trackformer/models/matcher.py




