# Netlife

Netlife is a cellular automaton simulation implemented using Python, Tkinter, and PyTorch. It models the behavior of cells in a grid and their evolution over time based on simple rules. Each cell in the grid has a neural network that predicts its next state based on the states of its neighboring cells.

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
   python3 netlife.py
   ```

2. A new window will open displaying the grid of cells.
3. Use the "Delay(ms)" slider to adjust the delay between each iteration of the simulation. Higher values result in slower updates, while lower values increase the speed.
4. Click the "Reset" button to reset the grid and start a new simulation.

## Rules and Behavior

- Each cell in the grid is represented by a colored square.
- Cells can be in one of two states: active or inactive. The initial state of each cell is randomly assigned.
- Cells update their states based on the states of their neighboring cells.
- The neural network of each cell predicts its next state based on the current neighborhood conditions.
- The neighborhood of a cell includes its eight adjacent cells.
- If a cell's state is greater than 0.5, it is considered active and will be represented by a colored square. If the cell's state is less or equal to 0.5, it will be considered inactive and will be represented by a white square.
- The cell's state is predicted based on the mean state of its neighboring cells.
- The neural network is trained using an Adam optimizer and the mean squared error loss function.
- Each cell has an "age" attribute, which increases every time the cell updates its state. If a cell's age exceeds a certain limit (e.g., 100), the cell resets itself, changing its color and state randomly.
- Cells with a state above a certain threshold (e.g., 0.9) can breed, creating a new cell at a random position in the grid.

## Modifying the Simulation

You can modify the simulation by making changes to the code. Here are a few suggestions:

- Adjust the size of the grid by modifying the `WIDTH`, `HEIGHT`, and `CELL_SIZE` constants.
- Change the behavior of the cells by modifying the rules in the `Cell` class. You can alter the conditions for breeding, resetting, or predicting the next state.
- Customize the appearance of the cells by modifying the color assignments in the `Cell` class.
- Experiment with different neural network architectures by modifying the `NeuralNet` class.
- Add additional functionality or controls to the GUI by extending the `Game` class.
