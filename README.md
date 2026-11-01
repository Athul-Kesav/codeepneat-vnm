# V:N:M Sparsity for NEAT-Evolved CNNs

This project evolves CNNs with [CoDeepNEAT](https://arxiv.org/abs/1703.00548) and then compresses every convolution with [VENOM](https://arxiv.org/abs/2401.01764)'s **V:N:M** sparse format (N=2, M ∈ {4, 8, 16, 32}).

Pruned models are fine-tuned and then run on Sparse Tensor Cores with VENOM's **Spatha** kernel. We use three datasets: CIFAR-10, CIFAR-100 and ImageNet-100.

**Full report:** [`report/report.pdf`](report/report.pdf) (25 pages; LaTeX source in `report/report.tex`).

**Team.** The CIFAR-10 experiments were done by Anirudh (originally in [Anirudhmoosad/codeepneat-venom](https://github.com/Anirudhmoosad/codeepneat-venom)): the Colab pilot, the 80-generation strong search, and the first Python prune/pack path with its 8-epoch recover study. Athul did CIFAR-100 and ImageNet-100, the Spatha integration and timing, and the second-order pruning. This repository merges both; the CIFAR-10 winner was then re-run through the CIFAR-100 pipeline. Anirudh's original CIFAR-10 report is kept as [`report/CoDeepNEAT_VNM.pdf`](report/CoDeepNEAT_VNM.pdf). Its accuracies were computed with CoDeepNEAT's buggy evaluator and are about 2 points high; `report/report.pdf` gives the corrected values.

**Slides:** [`presentation/VNM_NEAT_CNNs.pptx`](presentation/VNM_NEAT_CNNs.pptx) (19 slides, with speaker notes).

## Headline results

Best evolved model: `bp493` from the residual+GAP search on CIFAR-100. Dense, it has 8.4M parameters, reaches 76.50% and takes 33.7 MB.

| Setting | Test acc. | Conv compression | Stored size | Whole-network speedup (batch 128) |
|---|---|---|---|---|
| dense | 76.50 | 1× | 33.7 MB | – |
| 2:4, magnitude, V=8 | 76.26 | 3.1× | 10.8 MB | – |
| 2:8, gradual OBS, V=8 | 74.40 | 6.3× | 5.4 MB | – |
| 2:8, magnitude, V=32 (Spatha-runnable) | 74.01 | 6.4× | 5.4 MB | 1.08× |
| 2:16, gradual OBS, V=8 | 70.50 | 12.6× | 2.8 MB | – |
| 2:32, gradual OBS, V=8 | 63.72 | 25.1× | 1.4 MB | – |
| 2:32, gradual OBS, V=32 (Spatha-runnable) | 61.50 | 25.5× | ≈1.4 MB | 1.19× |

Speedup is only reported for V ≥ 32, because Spatha has no kernels for V = 8 or 16.

CIFAR-10 Colab pilot: `bp608` (12 generations on a T4) scores 81.98% dense (83.16% as originally reported). Its convolutions are only 4.8% of its 47.4 MB, so V:N:M barely shrinks it (zero-shot V=8: 2:4 69.55%, 2:8 30.77%, 2:16 11.47%).

CIFAR-10: `bp1485`, winner of an 80-generation search (`ada_cifar10_strong`), 11.1M parameters, 44.4 MB. Its dense checkpoint scores 85.61%; fine-tuned for 30 epochs without pruning (the fair control) it reaches 86.45%.

| Setting (30-epoch KD fine-tune, magnitude unless noted) | Test acc. | Conv compression | Stored size | Whole-network speedup (batch 128) |
|---|---|---|---|---|
| 2:4, V=32 | 86.35 | 3.3× | 13.5 MB | 0.92× |
| 2:8, V=8 | 86.52 | 6.3× | 7.0 MB | – |
| 2:8, V=32 | 86.36 | 6.6× | 6.7 MB | 1.11× |
| 2:16, V=8 | 85.48 | 12.6× | 3.5 MB | – |
| 2:16, V=32 | 85.02 | 13.2× | 3.4 MB | 1.25× |
| 2:32, V=32 | 80.52 | 26.3× | 1.7 MB | 1.30× |
| 2:32, V=8, gradual OBS | 81.55 | 25.3× | 1.8 MB | – |

Speedups are at V=32, where the accuracy was measured; at V=128 the same patterns reach 1.24× (2:8), 1.33× (2:16) and 1.37× (2:32), the highest in the study.

- **V:N:M works on NEAT-evolved CNNs.**
  - 2:4 is near-lossless on all 15 well-trained models: median drop 0.3 pt, maximum 1.05 pt.
  - On ImageNet-100, 2:4 is also near-lossless, and accuracy degrades gracefully up to 2:32 (38.7–41.6%).
- **The search space decides whether compression pays off.**
  - CoDeepNEAT's flatten+linear head keeps up to 77% of the bytes dense, so whole-model compression is only about 1.3×.
  - We added residual-sum aggregation and a GAP head to the search. With that, over 99% of the bytes are in prunable convolutions, which gives 24× whole-model compression and 4 pt more dense accuracy.
- **Irregular evolved channel widths clash with large V.** Row padding drops 2:8 compression from 6.3× to 4.1–5.7× at V=128. On ResNets, a larger V raises compression instead.
- **Gradual second-order pruning (VENOM-style) is decisive at high sparsity.**
  - At 2:32 it beats magnitude pruning by 9.2 pt on average.
  - One-shot OBS has a much better zero-shot accuracy, but the gain disappears after fine-tuning.
- **Speed is limited.**
  - Spatha speeds up the conv GEMMs by 1.4–2.6× at 2:8, and up to about 8× at 2:32.
  - A full forward pass is only 1.0–1.3× faster at large batch and is not faster at batch 1. Convolutions are under half the runtime, and im2col costs as much as the sparse GEMM.
- **The recovery recipe decides what looks feasible.** The earlier CIFAR-10 study used an 8-epoch Adam recover without augmentation or distillation and found 2:16 unusable (26%). Our 30-epoch distilled fine-tune brings it to 85.5%.
- **CoDeepNEAT's evaluator overstates accuracy** by n/(n−1) for n test batches: +1.3% relative at batch 128 on CIFAR, +2.6% at batch 256. It is fixed in this copy, and the report corrects every affected number.
- **Search robustness.** A blueprint whose network would not fit in GPU memory during the search now scores 0 instead of crashing the generation (merged from the CIFAR-10 search code).
- **Two silent Spatha bugs**, both returning wrong results without an error:
  - At M=32, the reduction dimension K must be padded to a multiple of 256.
  - A single call is limited to 65,535 column tiles.
- **ResNet baseline (brief).** Size-matched CIFAR ResNets are 1.6–1.8 pt more accurate when dense and lose about the same under V:N:M. At 2:32, gradual OBS on the evolved models (63.7%) beats every magnitude-pruned ResNet (best 54.7%).

## Repository layout

```text
final/
├── README.md, requirements.txt, requirements-venom.txt, .gitignore, .gitattributes
├── CoDeepNEAT/                 sash-a CoDeepNEAT (PyTorch), with our changes (see below)
│   ├── src/, configuration/    search code; configs in configuration/configs/{venom_cifar10,venom_cifar100,
│   │                           venom_cifar100_res,imagenet100,ada_cifar10_strong}.json
│   ├── data/                   DataManager.py (code) + Datasets/ (CIFAR-10/100 included; ImageNet-100 via setup_data.sh)
│   └── runs/<run>/             one folder per search (CIFAR-10: colab_baseline = Colab pilot, ada_cifar10_strong = 80-generation search,
│                               venom_cifar10 = small search)
│       ├── config.json, generations/*.pickle      architectures (needed to rebuild any blueprint)
│       ├── fully_trained_models/                  CoDeepNEAT full-train weights (CIFAR-10 searches)
│       ├── dense_fm3/                             bp1485 at feature multiplier 3 (ada_cifar10_strong)
│       ├── retrained_v2/, retrained/              dense weights used for pruning (+ results.jsonl)
│       ├── resnet_baselines/                      ResNet-18 / 20w58 / 56w32 weights (venom_cifar100)
│       └── vnm*/results.jsonl, spatha_bench.jsonl raw per-run pruning and Spatha results
├── vnm/                        our code
│   ├── vnm_pack.py             conv<->GEMM layout, V:N:M prune / pack / unpack, byte counting
│   ├── recipe.py               data pipeline (GPU aug, CutMix, EMA), training loops, KD fine-tune
│   ├── train_candidates.py     dense retraining of evolved blueprints
│   ├── train_resnet.py         CIFAR ResNet baselines
│   ├── prune_eval.py           first V:N:M grid (CoDeepNEAT weights, short fine-tune)
│   ├── prune_grid2.py          V×M grid, magnitude or one-shot OBS, 30-ep KD fine-tune
│   ├── prune_resnet_grid.py    the same grid for the ResNets
│   ├── prune_obs.py            V:N:M-aware one-shot OBS (SparseGPT-style)
│   ├── prune_gradual.py        gradual OBS (VENOM / oBERT-style schedule)
│   ├── spatha_bench.py         per-layer Spatha vs dense GEMM timing        (VENOM env)
│   ├── spatha_vscan.py         Spatha speed vs V on synthetic conv shapes    (VENOM env)
│   ├── e2e_bench.py            SpathaConv2d + whole-network timing          (VENOM env)
│   ├── e2e_check.py            Spatha-vs-cuDNN agreement check               (VENOM env)
│   ├── venom_compat.py         stubs unused heavy imports so CoDeepNEAT models build in the VENOM env
│   ├── plot_architectures.py   CoDeepNEAT's own Graphviz plots of the winners → figures/architectures/
│   └── build_cache.py          ImageNet-100 → uint8 numpy cache
├── scripts/                    setup_{main_env,venom,data}.sh, run_all.sh, run_*.sh per stage, env.sh, venom_env.sh,
│                               patches/spatha_v_custom.patch, export_imagenet100.py
├── external/                   created by the setup scripts, not shipped: env_main/, venom_build/
├── results/                    tables and raw results
│   ├── summary/all_experiments.md     every experiment in one table
│   ├── summary/resnet_vs_evolved.md
│   ├── pruning_criteria/              magnitude vs one-shot OBS vs gradual OBS
│   ├── grids/<grid>/                  per-grid summaries and CSVs
│   ├── spatha_gemm/vscan.jsonl
│   ├── end_to_end/                    e2e.csv (288 runs), e2e_summary.md, raw jsonl
│   └── search/                        per-generation search accuracy (+ ada_cifar10_strong_progress.csv)
├── figures/                    300-dpi PNGs of every plot and diagram (for slides)
│   └── architectures/          CoDeepNEAT's own graphs of the 15 evolved winners
├── presentation/               VNM_NEAT_CNNs.pptx (19-slide deck)
└── report/                     report.tex, report.pdf, figs/ (PDF figures), diagrams/ (TikZ sources)
```

## Quick start

Everything runs from this folder; nothing outside it is referenced. Large, machine-specific things (Python envs, the Spatha builds, the ImageNet-100 cache, caches and temp files) are created under `external/` by the setup scripts and are not part of the submission.

```bash
scripts/setup_main_env.sh     # external/env_main: Python 3.13, PyTorch 2.6 (CUDA 12.4), requirements.txt, Graphviz
scripts/setup_venom.sh        # external/venom_build: VENOM env (PyTorch 1.13.1, CUDA 11.7) + Spatha for V = 32/64/128/256
scripts/setup_data.sh all     # CIFAR-10/100 (shipped; re-downloaded if missing) + ImageNet-100 (about 8 GB download)
scripts/run_all.sh            # the whole project; or pick stages: run_all.sh search train prune speed figures
```

Requirements: conda, git, an NVIDIA GPU of the Ampere generation or newer (Sparse Tensor Cores), a driver that supports CUDA 12.4, and 3 GPUs for the scripts as written. Every script sources `scripts/env.sh`, which sets `PROJECT_ROOT`, points `VENOM_BUILD` at `external/venom_build` (override by exporting `VENOM_BUILD`), and puts `external/env_main/bin` first on `PATH`.

## Environments

Two environments are needed, because Spatha only builds against an old toolchain.

1. **Main** (search, training, pruning, plots), made by `scripts/setup_main_env.sh`: Python 3.13, PyTorch 2.6 + CUDA 12.4, `requirements.txt`, and Graphviz (`dot`, for the architecture graphs).
2. **VENOM** (all Spatha timing), made by `scripts/setup_venom.sh` in `$VENOM_BUILD`:
   - a conda env with Python 3.10, PyTorch 1.13.1, CUDA toolkit 11.7.1, cmake, pybind11;
   - `UDC-GAC/venom` at commit `8ddaf38`;
   - Spatha built for sm_86 (it runs on sm_89 Ada): the default build (V=128) is installed into the env; V=64 uses upstream's `setup_v64.py`; V=32 and V=256 use our one-hunk patch `scripts/patches/spatha_v_custom.patch` (a `V_CUSTOM` thread-block tile).
   - The extra builds live in `$VENOM_BUILD/spatha_v{32,64,256}_lib`; scripts pick one with `SPATHA_LIB`. Import `torch` before `spatha`.

## Data

- **CIFAR-10 and CIFAR-100** are included in `CoDeepNEAT/data/Datasets/` (356 MB). If they are missing, `scripts/setup_data.sh cifar` downloads them.
- **ImageNet-100** (CMC 100-class split, [clane9/imagenet-100](https://huggingface.co/datasets/clane9/imagenet-100), about 8 GB) is not included. `scripts/setup_data.sh imagenet` downloads it, exports it to an ImageFolder (used by the CoDeepNEAT search), and builds the 144-px uint8 cache in `$VENOM_BUILD/cache/in100` (used by training, pruning and timing).
- `CoDeepNEAT/data/DataManager.py` is code, not data, and must stay.

## Reproducing

`scripts/run_all.sh` runs these in order. All commands are run from this folder and write logs to `logs/`.

| Stage | Step | Command |
|---|---|---|
| search | Evolve + CoDeepNEAT full train | `scripts/run_search.sh <config>`: `venom_cifar10`, `venom_cifar100`, `imagenet100`, `venom_cifar100_res`, `ada_cifar10_strong` (the Colab pilot `colab_baseline` was run on one T4: `cd CoDeepNEAT && python src/main/evo.py -c colab_baseline -g 1 && python src/main/ft.py -c colab_baseline -g 1`) |
| train | Dense retrain, recipe v1 (CIFAR-100, ImageNet-100) | `scripts/run_retrain.sh <gpu>` for gpu 0, 1, 2 |
| train | Dense retrain, recipe v2 (300 ep, CutMix, EMA) | `scripts/run_retrain_v2.sh orig` and `res` |
| train | ResNet baselines | `scripts/run_resnets.sh` |
| prune | First grids (CoDeepNEAT weights, short fine-tune) | `scripts/run_grid.sh <run> <gpu> <epochs>` |
| prune | V×M grids, magnitude, 30-ep KD fine-tune, Spatha GEMM timing | `scripts/run_grid2.sh` (ImageNet-100), `run_grid3.sh`, `run_gridres.sh`, `run_grid_resnet.sh` |
| prune | One-shot OBS / gradual OBS | `scripts/run_obs.sh`, `scripts/run_gradual.sh` |
| prune | CIFAR-10 strong-search winner bp1485 | `scripts/run_cifar10_ada.sh` |
| prune | CIFAR-10 Colab pilot bp608, zero-shot | `cd CoDeepNEAT && python ../vnm/prune_eval.py --run colab_baseline --config colab_baseline --model runs/colab_baseline/fully_trained_models/bp-608_fm-1-best-1.model --v 8 --ms 4,8,16 --ft-epochs 0 --out vnm_zeroshot --layers-for-v ""` |
| speed | Spatha V-scan | `scripts/run_vscan.sh <gpu> <V>` |
| speed | End-to-end inference | `scripts/run_e2e.sh <gpu> <V>` (set `EXTRA="--models ... --batches 32,1"` for ImageNet-100) |
| speed | ImageNet-100 correctness (32 images) | `scripts/run_in100_check.sh <gpu> <V>` |
| figures | Architecture graphs | `cd vnm && python plot_architectures.py` |
| figures | Report | `cd report && pdflatex report.tex && pdflatex report.tex` |

The trained weights of every model in the report are included, so any stage can be run without the stages before it. The diagram sources are in `report/diagrams/*.tex` (standalone TikZ, compiled to `report/figs/diag_*.pdf`).

## How V:N:M is applied to a convolution

- **Weight layout.** `W2d = w.permute(0,2,3,1).reshape(C_out, kh*kw*C_in)`, so column groups are input channels at one kernel tap.
- **Pruning.** In each V×M block, keep 4 columns. Inside those columns, keep 2 values per row (2:4).
- **What stays dense.** BN, biases and the classifier stay dense in fp32.
- **Stored format.** fp16 values, a 2-bit index per value, and 1 byte per kept-column position.
- **Fine-tuning.** 30 epochs with KD from the dense model. The mask is re-applied after every optimizer step.
- **Inference (`SpathaConv2d`).** im2col by one strided copy per kernel tap, then `spatha.spmm`; the output is used directly as NHWC.
- **Correctness.** Checked against cuDNN on every configuration: top-1 agreement ≥ 99.2%.

## Evolved architectures

`figures/architectures/` holds the plots made by CoDeepNEAT's built-in visualisers (Graphviz), for the top 3 models of every search:

- `<run>_bp<id>_blueprint_genome.png` (`BlueprintGenome.visualize`): the blueprint graph. Each node names its module species, and the module chosen for each species is drawn as a subgraph.
- `<run>_bp<id>_phenotype.png` (`Network.visualize`): the assembled network, layer by layer. Conv nodes are yellow, aggregation nodes violet, other nodes green. The classifier head (GAP or flatten, then linear) is added after the graph, so it is not drawn.

The best model is `venom_cifar100_res_bp493_*`.

## Changes to CoDeepNEAT (vs. sash-a upstream)

- **New options:**
  - `global_avg_pool`: GAP head;
  - `residual_sum`: sum aggregation, identity for the widest input and 1×1 projection otherwise;
  - `final_relu`;
  - `image_size`.
- **Data loaders:** CIFAR-100 and ImageNet-100.
- **`.view` → `.reshape`:** needed for channels-last tensors.
- **Oversized classifiers:** such blueprints are dropped instead of crashing the generation.
- **Out-of-memory during the search** (from the CIFAR-10 search code): networks are built on the CPU and moved to the GPU only for training; a CUDA out-of-memory error scores the blueprint 0 instead of killing the generation.
- **Evaluator:** `test_nn` counts every test sample once (the upstream version overstated accuracy by n/(n−1)).
- **Compatibility:** `imgaug` patched for current NumPy.

## Limitations

- Single seed, and model / V selection was done on the test set.
- Gradual OBS got 35–45 fine-tune epochs, against 30 for the other criteria.
- OBS was not run on the ResNets.
- Timings were taken on shared GPUs, in eager PyTorch 1.13.
- Settings with V = 8 or 16 cannot be timed.

The report discusses each point.

## References

- Miikkulainen et al., *Evolving Deep Neural Networks*, 2019 (arXiv:1703.00548). Implementation: https://github.com/sash-a/CoDeepNEAT
- Castro et al., *VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores*, SC'23. Code: https://github.com/UDC-GAC/venom
- Frantar & Alistarh, *SparseGPT*, ICML 2023. Kurtic et al., *The Optimal BERT Surgeon*, EMNLP 2022.
