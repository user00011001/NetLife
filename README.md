# Netlife

Netlife is a cellular automaton simulation implemented using Python and Tkinter. It models the behavior of cells in a grid and their evolution over time based on simple rules. Each cell in the grid has a neural network that predicts its next state based on the states of its neighboring cells.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- Tkinter library
- PyTorch library

## Getting Started

1. Clone the repository or download the project files to your local machine.
2. Ensure that you have the necessary dependencies installed.
3. Open a terminal or command prompt and navigate to the project directory.

## Running the Simulation

To start the Netlife simulation, follow these steps:

1. Run the following command in the terminal:

   ```
   python netlife.py
   ```

2. A new window will open displaying the grid of cells.

3. Use the "Delay(ms)" slider to adjust the delay between each iteration of the simulation. Higher values result in slower updates, while lower values increase the speed.

4. Click the "Reset" button to reset the grid and start a new simulation.

## Rules and Behavior

- Each cell in the grid is represented by a colored square.
- Cells can be in one of two states: active or inactive.
- The initial state of each cell is randomly assigned.
- Cells update their states based on the states of their neighboring cells.
- The neural network of each cell predicts its next state based on the current neighborhood conditions.
- The neighborhood of a cell includes its eight adjacent cells.
- The target state for each cell depends on the number of active neighbors:
  - If a cell has fewer than two or more than three active neighbors, it targets an inactive state (underpopulation or overpopulation).
  - If a cell has two or three active neighbors, it targets an active state (ideal conditions).
- The neural network is trained using an Adam optimizer and the mean squared error loss function.
- Cells receive a time-dependent reward for being in an active state, encouraging them to stay active.

## Modifying the Simulation

You can modify the simulation by making changes to the code. Here are a few suggestions:

- Adjust the size of the grid by modifying the `WIDTH`, `HEIGHT`, and `CELL_SIZE` constants.
- Change the behavior of the cells by modifying the rules in the `Cell` class.
- Customize the appearance of the cells by modifying the color assignments in the `Cell` and `Game` classes.
- Experiment with different neural network architectures by modifying the `NeuralNet` class.
- Add additional functionality or controls to the GUI by extending the `Game` class.
