# Installation

This page describes a reproducible setup for ParaStereoSync on Windows, macOS, and Linux.

## Prerequisites

- Python 3.9+ (3.10 recommended)
- Anaconda or Miniconda (recommended)
- `pip`
- Optional: `git`

## Conda Setup (Recommended)

```bash
conda create -n stereosync python=3.10 -y
conda activate stereosync
cd /path/to/ParaStereoSync
pip install -r requirements.txt
```

## Verify Core Imports

```bash
python -c "import numpy, scipy, pandas, cv2, matplotlib, skimage, tqdm; print('OK')"
```

## External Tool Dependency

`VidTrim.py` uses FFmpeg for lossless trimming.

Install FFmpeg and verify:

```bash
ffmpeg -version
```

If FFmpeg is not on `PATH`, trimming functions in `VidTrim.py` will fail.

## Platform Notes

1. Windows
- `tkinter` is typically included in standard Python distributions.
- GUI file dialogs and OpenCV windows are fully supported.

2. Linux
- Install GUI dependencies if missing (Tk/OpenCV highgui backend).

3. macOS
- Ensure Python build includes Tk support for GUI tools.

## Reproducible Environment Locking (Recommended)

After successful setup, archive environment details:

```bash
pip freeze > requirements-lock.txt
```

For Conda users:

```bash
conda env export --no-builds > environment.yml
```
