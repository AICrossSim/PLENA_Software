# Installation Guide

## 1. Create the virtual environment

```bash
uv venv --python 3.13
source .venv/bin/activate
```

## 2. Install `mase`

```bash
git clone git@github.com:DeepWok/mase.git
cd mase
git checkout <branch-or-commit>
uv pip install -e .
cd ..
```

## 3. Install `fast-hadamard-transform`

```bash
git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
uv pip install -e ./fast-hadamard-transform --no-build-isolation
```

## 4. Install evaluation backends

```bash
uv pip install lm-eval
uv pip install evalplus
```
