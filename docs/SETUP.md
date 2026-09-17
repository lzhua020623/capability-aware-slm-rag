# Windows setup

PyTorch is **not** listed in `requirements.txt`. Install it separately for your hardware, then install the remaining packages.

## 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Install PyTorch

NVIDIA CUDA 13.0 users should run this first:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

CPU-only or other CUDA versions: follow the official PyTorch install command for your machine at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

## 3. Install project dependencies

```powershell
pip install -r requirements.txt
```

## 4. Check GPU

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`True` means PyTorch can see the GPU. `False` means it is using CPU (or CUDA was not installed correctly).
