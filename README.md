# Deep Image Prior – étude des biais d'optimisation

Ce dépôt fournit une implémentation **pure PyTorch** du *Deep Image Prior* (DIP) et un banc d'essais
permettant d'étudier l'impact sur la qualité de restauration de :

1) le choix du réseau (architecture et profondeur),
2) l'algorithme d'optimisation,
3) le pas de descente (learning rate),
4) l'échantillonnage / l'ensemblage des sorties DIP (multi-seeds),
5) des métriques simples de "lissage" pour relier ces observations à l'implicite biais de stabilité des minima.

> Référence DIP : Ulyanov, Vedaldi, Lempitsky — *Deep Image Prior*, CVPR 2018.
> Référence biais implicite/stabilité : Mulayoff, Michaeli, Soudry — *The Implicit Bias of Minima Stability: A View from Function Space*, NeurIPS 2021.

## Installation

```bash
# Créez un environnement, installez torch (cpu ou cuda) puis :
pip install -r requirements.txt
```

## Quickstart (une image)

Dén oisage supervisé (connu x_ref) :
```bash
python train.py --task denoise --image path/to/img.png --gt path/to/img_clean.png   --net unet --depth 5 --features 64 --optimizer adam --lr 0.01 --iterations 4000   --save_dir runs/denoise_demo
```

Sur-échantillonnage (x4) :
```bash
python train.py --task superres --image path/to/lr.png --scale 4   --net unet --depth 5 --optimizer adam --lr 0.01 --iterations 2000   --save_dir runs/sr_x4_demo
```

Inpainting (masque binaire) :
```bash
python train.py --task inpaint --image path/to/corrupted.png --mask path/to/mask.png   --net encdec --depth 6 --optimizer sgd --momentum 0.9 --lr 0.05 --iterations 3000   --save_dir runs/inpaint_demo
```

## Scanner de configurations (grille)

```bash
python experiment_sweep.py --task denoise --image path/to/img.png   --grid configs/grid_search.json --save_dir runs/sweep_denoise
```

## Ensemblage / échantillonnage

```bash
python sample_ensemble.py --task denoise --image path/to/noisy.png   --net unet --depth 5 --optimizer adam --lr 0.01 --iterations 1800   --seeds 8 --aggregate mean --save_dir runs/denoise_ens
```

## Mesures de lissage vs. learning rate

```bash
python analyze_lr_bias.py --task denoise --image path/to/noisy.png   --net unet --depth 5 --optimizer sgd --lrs 0.0005 0.001 0.005 0.01 0.05   --iterations 2000 --save_dir runs/lr_bias
```

Cette commande trace et sauvegarde pour chaque LR : PSNR, loss, TV(L1),
énergie haute fréquence (FFT) — utiles pour discuter du biais vers des solutions plus lisses
lorsque le pas augmente.

## Structure

- `dip/models/` : UNet (avec/ sans skip), Encoder-Decoder, ResUNet minimal, ConvNeXt-lite.
- `dip/tasks/` : formulations DIP pour *denoise*, *superres*, *inpaint*.
- `dip/operators/` : downsamplers différentiables, gestion du masque.
- `dip/utils/` : PSNR/SSIM simple, TV, I/O, early-stopping, seed.
- `train.py` : exécute une expérience unique.
- `experiment_sweep.py` : grille d'architectures/optimiseurs/LR.
- `sample_ensemble.py` : ensemblage multi-seeds.
- `analyze_lr_bias.py` : métriques de lissage vs LR (TV, FFT).

## Remarques

- DIP est sensible à l'**early stopping** : surveillez PSNR/SSIM (si GT) ou utilisez la
  meilleure itération sur la *courbe de perte* (arrêt avant sur-apprentissage du bruit).
- Les opérateurs de downsampling sont **différentiables** (interpolate bilinear) pour SR.
- Le code supporte plusieurs ***optimizers*** (SGD, SGD+momentum, Adam, RMSprop, LBFGS).

## Citation

- Ulyanov et al., *Deep Image Prior*, CVPR 2018.
- Mulayoff et al., *The Implicit Bias of Minima Stability*, NeurIPS 2021.

