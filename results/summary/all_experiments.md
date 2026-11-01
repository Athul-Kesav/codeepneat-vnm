# All V:N:M experiments

Cells: best fine-tuned test acc % over V (V used, conv compression vs fp32).

> **Evaluator correction.** Rows marked "CoDeepNEAT weights" / "search weights" (CIFAR-10 small search, early CIFAR-100 and ImageNet-100 rows) were scored by CoDeepNEAT's `test_nn`, which divides by n−1 batches instead of n. Their true values are lower by a factor (n−1)/n: ×0.9873 on CIFAR (10k test images, batch 128) and ×0.975 on ImageNet-100 (5k images, batch 128). Every other row was scored by our own evaluator and is exact. The report uses the corrected values. The evaluator in `CoDeepNEAT/` is now fixed.

| Dataset | Models | Pruning | Model | Params | Dense | 2:4 | 2:8 | 2:16 | 2:32 |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | evolved (orig) | magnitude | bp671 | 0.57M | 80.75 | 80.77 (V8, 3.1x) | 74.09 (V8, 6.1x) | 54.61 (V16, 12.2x) | 39.66 (V8, 23.7x) |
| CIFAR-10 | evolved (orig) | magnitude | bp495 | 1.91M | 81.63 | 81.31 (V32, 2.9x) | 80.04 (V8, 6.1x) | 72.60 (V8, 12.2x) | 35.06 (V8, 24.2x) |
| CIFAR-10 | evolved (orig) | magnitude | bp546 | 2.84M | 79.26 | 78.53 (V16, 3.1x) | 72.80 (V32, 5.9x) | 48.33 (V32, 11.8x) | 31.88 (V8, 24.4x) |
| CIFAR-100 | evolved (orig), search weights | magnitude | bp606 | 1.22M | 49.83 | 49.21 (V8, 3.1x) | 42.13 (V16, 6.4x) | 15.40 (V64, 12.9x) | 6.75 (V32, 25.3x) |
| CIFAR-100 | evolved (orig), search weights | magnitude | bp735 | 38.19M | 42.39 | 40.50 (V64, 2.8x) | 35.27 (V16, 6.5x) | 26.16 (V16, 13.0x) | 19.62 (V16, 25.9x) |
| CIFAR-100 | evolved (orig), search weights | magnitude | bp315 | 4.30M | 44.84 | 44.25 (V128, 2.5x) | 41.00 (V8, 6.0x) | 27.29 (V8, 11.9x) | 11.25 (V32, 25.5x) |
| CIFAR-100 | evolved (orig), 200ep retrain | magnitude | bp427 | 3.27M | 67.50 | 67.42 (V128, 2.8x) | 65.33 (V8, 6.3x) | 60.00 (V8, 12.6x) | 43.53 (V8, 25.0x) |
| CIFAR-100 | evolved (orig), 200ep retrain | magnitude | bp430 | 3.62M | 67.44 | 67.06 (V32, 3.2x) | 64.63 (V16, 6.4x) | 59.62 (V8, 12.6x) | 47.36 (V8, 25.0x) |
| CIFAR-100 | evolved (orig), 200ep retrain | magnitude | bp499 | 5.49M | 67.06 | 67.00 (V16, 3.2x) | 64.90 (V32, 6.5x) | 61.13 (V16, 12.7x) | 53.24 (V8, 24.8x) |
| CIFAR-100 | evolved (orig), 300ep retrain | magnitude | bp430 | 3.62M | 72.39 | 71.34 (V32, 3.2x) | 68.40 (V8, 6.3x) | 62.40 (V8, 12.6x) | 49.79 (V8, 25.0x) |
| CIFAR-100 | evolved (orig), 300ep retrain | magnitude | bp427 | 3.27M | 72.17 | 71.50 (V32, 3.2x) | 68.61 (V8, 6.3x) | 62.35 (V8, 12.6x) | 50.50 (V8, 25.0x) |
| CIFAR-100 | evolved (orig), 300ep retrain | magnitude | bp499 | 5.49M | 69.57 | 69.29 (V16, 3.2x) | 66.84 (V8, 6.3x) | 63.42 (V8, 12.5x) | 56.53 (V8, 24.8x) |
| CIFAR-100 | ResNet from scratch, 300ep | magnitude | r20w58 | 3.58M | 78.26 | 78.13 (V64, 3.2x) | 75.43 (V8, 6.3x) | 70.79 (V8, 12.6x) | 50.63 (V8, 24.9x) |
| CIFAR-100 | ResNet from scratch, 300ep | magnitude | r56w32 | 3.42M | 78.08 | 77.20 (V16, 3.4x) | 74.40 (V8, 6.4x) | 70.19 (V8, 12.8x) | 54.08 (V8, 25.6x) |
| CIFAR-100 | ResNet from scratch, 300ep | magnitude | r18 | 11.22M | 79.83 | 79.49 (V32, 3.5x) | 77.62 (V8, 6.4x) | 73.06 (V8, 12.8x) | 54.72 (V8, 25.6x) |
| CIFAR-100 | evolved residual+GAP, 300ep | magnitude | bp493 | 8.42M | 76.50 | 76.26 (V8, 3.1x) | 74.01 (V32, 6.4x) | 70.06 (V8, 12.6x) | 57.23 (V8, 25.1x) |
| CIFAR-100 | evolved residual+GAP, 300ep | magnitude | bp482 | 2.51M | 72.69 | 72.44 (V32, 3.1x) | 69.60 (V8, 6.3x) | 62.14 (V8, 12.5x) | 42.37 (V8, 25.0x) |
| CIFAR-100 | evolved residual+GAP, 300ep | magnitude | bp741 | 4.33M | 72.47 | 72.20 (V8, 3.1x) | 70.15 (V8, 6.3x) | 66.60 (V8, 12.6x) | 57.05 (V8, 25.1x) |
| CIFAR-100 | evolved residual+GAP, 300ep | OBS one-shot | bp493 | 8.42M | 76.50 | - | 73.92 (V8, 6.3x) | 69.23 (V8, 12.6x) | 57.98 (V8, 25.1x) |
| CIFAR-100 | evolved residual+GAP, 300ep | OBS one-shot | bp482 | 2.51M | 72.69 | - | 69.94 (V8, 6.3x) | 62.75 (V8, 12.5x) | 47.23 (V8, 25.0x) |
| CIFAR-100 | evolved residual+GAP, 300ep | OBS one-shot | bp741 | 4.33M | 72.47 | - | 70.45 (V8, 6.3x) | 66.43 (V8, 12.6x) | 57.11 (V8, 25.1x) |
| CIFAR-100 | evolved residual+GAP, 300ep | OBS gradual | bp493 | 8.42M | 76.50 | - | 74.40 (V8, 6.3x) | 70.50 (V8, 12.6x) | 63.72 (V8, 25.1x) |
| CIFAR-100 | evolved residual+GAP, 300ep | OBS gradual | bp482 | 2.51M | 72.69 | - | 70.24 (V8, 6.3x) | 64.33 (V8, 12.5x) | 55.37 (V8, 25.0x) |
| CIFAR-100 | evolved residual+GAP, 300ep | OBS gradual | bp741 | 4.33M | 72.47 | - | 70.92 (V8, 6.3x) | 68.09 (V8, 12.6x) | 61.78 (V8, 25.1x) |
| ImageNet-100 | evolved, search weights | magnitude | bp230 | 0.40M | 29.63 | 29.51 (V128, 1.5x) | 15.77 (V8, 6.3x) | 6.09 (V16, 11.6x) | 3.43 (V32, 23.6x) |
| ImageNet-100 | evolved, search weights | magnitude | bp22 | 1.39M | 1.04 | 10.70 (V32, 2.7x) | 10.70 (V32, 5.3x) | 6.07 (V64, 8.9x) | 3.39 (V128, 13.0x) |
| ImageNet-100 | evolved, search weights | magnitude | bp254 | 0.40M | 28.45 | 27.24 (V8, 3.2x) | 16.77 (V8, 6.3x) | 5.89 (V8, 12.6x) | 2.62 (V8, 25.0x) |
| ImageNet-100 | evolved, retrained | magnitude | bp207 | 2.77M | 53.68 | 54.30 (V32, 2.9x) | 53.28 (V8, 6.2x) | 48.00 (V8, 12.4x) | 38.70 (V8, 24.7x) |
| ImageNet-100 | evolved, retrained | magnitude | bp182 | 3.05M | 52.36 | 51.84 (V8, 3.0x) | 50.24 (V8, 6.0x) | 47.48 (V8, 11.9x) | 41.60 (V8, 23.7x) |
| ImageNet-100 | evolved, retrained | magnitude | bp230 | 0.47M | 52.28 | 51.96 (V8, 3.2x) | 49.26 (V16, 5.8x) | 45.38 (V8, 12.6x) | 40.12 (V8, 25.0x) |
| CIFAR-10 | evolved, Colab pilot (12 gen, T4) | magnitude, zero-shot (no fine-tune) | bp608 | 11.85M | 81.98 | 69.55 (V8, 3.2x) | 30.77 (V8, 6.4x) | 11.47 (V8, 12.7x) | - |
| CIFAR-10 | evolved, strong search (80 gen), fm 3 | magnitude, 8-ep Adam (original recipe) | bp1485 | 11.11M | 85.59 | 86.01 (V8, 3.2x) | 79.78 (V8, 6.3x) | 26.36 (V8, 12.6x) | - |
| CIFAR-10 | evolved, strong search (80 gen), fm 3 | magnitude, 30-ep KD | bp1485 | 11.11M | 85.61 (control 86.45) | 86.35 (V32, 3.3x) | 86.52 (V8, 6.3x) | 85.48 (V8, 12.6x) | 80.52 (V32, 26.3x) |
