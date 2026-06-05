# Visualization Tutorial

This tutorial explains how to visualize 3D profiles and exported tracking data.

1. Visualize with the GUI

```bash
python KineVis3D.py
```

- Use `Load file` to open a `3d_points.csv` produced by `ParaStereoSync.py`.
- Select points or groups and play the frame slider to inspect displacement over time.

2. Command-line inspection
- Use `pandas` and `matplotlib` to quickly plot selected point trajectories. Example:

```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('3d_points.csv')
plt.plot(df['frame'], df['Z_point_1'])
plt.xlabel('frame')
plt.ylabel('Z displacement (units)')
plt.show()
```

3. Exporting figures
- Use Matplotlib `savefig()` in your scripts to produce publication-quality figures. Consider using `matplotlib.rcParams` to set font sizes and figure resolution (e.g., `dpi=300`).
