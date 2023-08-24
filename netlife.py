import torch
import random
import tkinter as tk
from tkinter import ttk
import torch.nn as nn
import torch.optim as optim

WIDTH, HEIGHT, CELL_SIZE = 400, 400, 10
GRID_WIDTH, GRID_HEIGHT = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
BREED_THRESHOLD = 0.75
AGE_LIMIT = 100
DELAY_SCALE_RANGE = (1, 1000)

def random_color():
    """Generate a random hex color."""
    return f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}"

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(9, 18)
        self.fc2 = nn.Linear(18, 9)
        self.fc3 = nn.Linear(9, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Cell:
    """Represents a cell with a neural network and a state."""
    
    def __init__(self):
        self.neural_net = NeuralNet()
        self.optimizer = optim.Adam(self.neural_net.parameters())
        self.state = torch.tensor([random.randint(0, 1)], dtype=torch.float32)
        self.color = random_color()
        self.age = 0

    def predict_next_state(self, neighborhood_states):
        """Predict and adjust the cell's state based on its neighborhood."""
        self.optimizer.zero_grad()
        target_state = torch.tensor([sum(neighborhood_states) / len(neighborhood_states)], dtype=torch.float32)
        predicted_state = self.neural_net(torch.tensor(neighborhood_states, dtype=torch.float32))
        loss = (predicted_state - target_state) ** 2
        loss.backward()
        self.optimizer.step()
        self.state = predicted_state.detach().clone()
        self.age += 1
        if self.age > AGE_LIMIT:
            self.reset()

    def breed(self, other):
        """Create a new cell by breeding with another."""
        child = Cell()
        for child_param, parent_param1, parent_param2 in zip(child.neural_net.parameters(), self.neural_net.parameters(), other.neural_net.parameters()):
            child_param.data.copy_((parent_param1.data + parent_param2.data) / 2)
        return child

    def reset(self):
        """Reset the cell to a default state."""
        self.state = torch.tensor([0], dtype=torch.float32)
        self.age = 0

class Grid:
    """Manages a grid of cells."""
    
    def __init__(self):
        self.grid = [[Cell() for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    def update(self):
        """Update the states of the cells in the grid."""
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                neighborhood_states = [cell.state.item() for _, _, cell in self.get_neighborhood(row, col)]
                self.grid[row][col].predict_next_state(neighborhood_states)
                if self.grid[row][col].state.item() > BREED_THRESHOLD:
                    for _, _, neighbor in self.get_neighborhood(row, col):
                        if neighbor.state.item() > BREED_THRESHOLD:
                            new_cell = self.grid[row][col].breed(neighbor)
                            self.grid[random.randint(0, GRID_HEIGHT-1)][random.randint(0, GRID_WIDTH-1)] = new_cell

    def get_neighborhood(self, i, j):
        """Retrieve the cells surrounding a given cell in the grid."""
        return [
            (x%GRID_HEIGHT, y%GRID_WIDTH, self.grid[x%GRID_HEIGHT][y%GRID_WIDTH])
            for x in range(i-1, i+2)
            for y in range(j-1, j+2)
        ]

class Game:
    """GUI for visualizing and controlling the grid of cells."""
    
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=WIDTH, height=HEIGHT)
        self.canvas.pack()
        self.grid = Grid()
        self.delay = tk.StringVar(master, value=100)
        tk.Scale(master, from_=DELAY_SCALE_RANGE[0], to=DELAY_SCALE_RANGE[1], orient=tk.HORIZONTAL, label='Delay(ms)', variable=self.delay).pack()
        tk.Button(master, text="Reset", command=self.reset).pack()

    def reset(self):
        """Reset the game grid."""
        self.grid = Grid()

    def draw(self):
        """Draw the current state of the grid."""
        self.canvas.delete("all")
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                cell = self.grid.grid[row][col]
                color = cell.color if cell.state.item() > 0.5 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
        self.master.update()

    def game_loop(self):
        """Continuously update and draw the game."""
        self.draw()
        self.grid.update()
        self.master.after(int(self.delay.get()), self.game_loop)

if __name__ == "__main__":
    root = tk.Tk()
    game = Game(root)
    game.game_loop()
    root.mainloop()
