import random
import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

WIDTH, HEIGHT, CELL_SIZE = 400, 400, 10
GRID_WIDTH, GRID_HEIGHT = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(9, 9)
        self.fc2 = nn.Linear(9, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
class Cell:
    def __init__(self):
        self.neural_net = NeuralNet()
        self.optimizer = optim.Adam(self.neural_net.parameters())
        self.state = torch.tensor([random.randint(0, 1)], dtype=torch.float32)
        self.color = f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}"
        self.time_alive = 0  # To store the time for which the cell is alive

    def predict_next_state(self, neighborhood):
        self.optimizer.zero_grad()
        
        # Defining target state based on neighborhood conditions
        active_neighbors = sum(neighborhood)
        if active_neighbors < 2 or active_neighbors > 3:
            target_state = torch.tensor([0.], dtype=torch.float32)  # Underpopulation or Overpopulation
        else:
            target_state = torch.tensor([1.], dtype=torch.float32)  # Ideal conditions

        predicted_state = self.neural_net(torch.tensor(neighborhood, dtype=torch.float32))

        # Adding a time-dependent reward to the loss function
        reward = torch.tensor([self.time_alive], dtype=torch.float32)
        loss = (predicted_state - target_state) ** 2 - reward
        loss.backward()
        self.optimizer.step()

        # Update state and time alive
        self.state = predicted_state.detach().clone()
        if self.state.item() > 0.5:
            self.time_alive += 1
        else:
            self.time_alive = 0  # Reset time alive if the cell is not in an active state


class Grid:
    def __init__(self):
        self.grid = [[Cell() for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    def update(self):
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                neighborhood = self.get_neighborhood(row, col)
                self.grid[row][col].predict_next_state(neighborhood)

    def get_neighborhood(self, i, j):
        neighborhood = []
        for x in range(i-1, i+2):
            for y in range(j-1, j+2):
                neighborhood.append(self.grid[x%GRID_HEIGHT][y%GRID_WIDTH].state.item())
        return neighborhood

class Game:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=WIDTH, height=HEIGHT)
        self.canvas.pack()
        self.grid = Grid()
        self.delay = tk.StringVar(master)
        self.delay.set(100)
        tk.Scale(master, from_=1, to=1000, orient=tk.HORIZONTAL, label='Delay(ms)', variable=self.delay).pack()
        tk.Button(master, text="Reset", command=self.reset).pack()

    def reset(self):
        self.grid = Grid()

    def draw(self):
        self.canvas.delete("all")
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                cell = self.grid.grid[row][col]
                if cell.state.item() > 0.5:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=cell.color)
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
        self.master.update()

    def game_loop(self):
        self.draw()
        self.grid.update()
        self.master.after(int(self.delay.get()), self.game_loop)


if __name__ == "__main__":
    root = tk.Tk()
    game = Game(root)
    game.game_loop()
    root.mainloop()
