# FAQ

Q: Where are calibration state files stored?
A: Intermediate calibration state is saved as `tkCalib_init_*.npy` in the repository root. These store grid, image and guess parameters used by the GUI.

Q: How do I reset calibration state?
A: Remove or move `tkCalib_init_*.npy` files and re-run `tkCalib2.py`.

Q: The GUI freezes when I run tracking on Windows.
A: Ensure multiprocessing is guarded with `if __name__ == '__main__':` in any script you call directly. Also try reducing worker count.

Q: Tracking templates drift or fail.
A: Increase template size, increase pyramid levels, or reduce maximum warp per frame. Verify video quality and lighting stability.

Q: How do I reproduce results for publication?
A: Save exact camera parameter files, template CSVs, tracking CSVs and the `tkCalib_init_*.npy` files; include software version and `requirements.txt`. Consider using Docker for environment capture.
