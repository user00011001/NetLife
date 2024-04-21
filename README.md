# NetLife

This Python script demonstrates a simulation of cells controlled by individual neural networks. The cells interact with their neighbors and evolve over time based on the predictions made by their neural networks.

## Features

- Each cell has its own neural network to predict its next state.
- Cells can breed with neighbors if their states exceed a threshold.
- Tkinter-based GUI for visualization and control.

## Dependencies

- Python 3.x
- PyTorch
- Tkinter

## How to Run

1. Install dependencies.
2. Run `netlife.py`.
3. Use the GUI controls to adjust parameters and observe the cells.

## Configuration

Adjust the following constants to modify the simulation:

- `WIDTH`, `HEIGHT`: Simulation window dimensions.
- `CELL_SIZE`: Size of each cell.
- `BREED_THRESHOLD`: Threshold for cell breeding.
- `AGE_LIMIT`: Maximum age of a cell before reset.
- `DELAY_SCALE_RANGE`: Range of simulation delay values.
