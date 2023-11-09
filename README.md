# Red Ball Tracking and LiDAR Data Analysis

This README provides guidelines for running the red ball tracking and LiDAR data analysis code.

## Libraries

The following libraries are required. Please install them if not already present:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

## Configuration
Before running the code, update the paths to the video and Excel files according to your local setup.

## Execution Instructions

### Problem 1: Red Ball Movement Analysis

To analyze the red ball movement:

1. Execute the provided code in the terminal. Runtime may vary based on system performance.
2. Expect five sequential outputs:
   - A video displaying the red ball's movement, with its center marked in green.
   - A graph illustrating the trajectory of the ball's center.
   - A plot of the fitted parabola to the ball's trajectory.
   - The equation describing the curve.
   - The x-coordinate of the ball's landing position.

### Problem 2: LiDAR Data Analysis

#### Part 2.1

1. Run the code in the terminal.
2. Outputs will include:
   - The covariance matrix.
   - The magnitude and direction of the surface normal.

#### Part 2.2

1. Execute the RANSAC code in the terminal. Note that due to iterations, execution may take some time.
2. The following outputs will be provided sequentially:
   - Least Square for pc1, including error.
   - Total least square for pc1, including error.
   - RANSAC for pc1, including error.
   - Least Square for pc2, including error.
   - Total least square for pc2, including error.
   - RANSAC for pc2, including error.

## Results

The results are visualized in the images below:

![Initial Plot](/Results/Q1/initialplot.png)
![Fitted Plot](/Results/Q1/fitted.png)


<!-- <img src ="Results\Q1\initialplot.png" width=400/>
<img src ="Results\Q1\fitted.png" width=400/> -->
