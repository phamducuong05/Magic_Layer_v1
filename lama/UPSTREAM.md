# Vendored LaMa source

- Repository: https://github.com/advimman/lama.git
- Commit: `786f5936b27fb3dacd2b1ad799e4de968ea697e7`
- License: Apache-2.0; see `LICENSE`.

Local runtime compatibility change:

- `saicinpainting/utils.py` imports `pytorch_lightning.seed_everything`
  lazily because generator-only inference does not require PyTorch Lightning.
