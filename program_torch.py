"""
SETUP:

conda install cudatoolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

"""

from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

GUI_WINDOW = None
GUI_CANVAS = None
GUI_CHART = None

TOTAL_POINTS = 0

POINTS_MAP = {
    "simple": 10,
    "complex": 15,
    "medium": 25,
    "large": 50
}
def on_window_close():
    import sys
    sys.exit()

# Simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define a more complex neural network
class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 512),  # Increase input size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output
        )

    def forward(self, x):
        return self.layers(x)

class ComplexMediumNN(nn.Module):
    def __init__(self):
        super(ComplexMediumNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 1024),  # Increase input size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output
        )

    def forward(self, x):
        return self.layers(x)

class ComplexLargeNN(nn.Module):
    def __init__(self):
        super(ComplexLargeNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),  # Increase input size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output
        )

    def forward(self, x):
        return self.layers(x)

# Training function with progress visualization
def train_model(use_device, models, epochs=100):
    global GUI_WINDOW, GUI_CANVAS, GUI_CHART, TOTAL_POINTS, POINTS_MAP

    start_time = time()

    loss_values = []

    # Initialize model, optimizer, and data
    if use_device == 'cpu':
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running benchmark on: {} " . format(device))

    model = None
    for use_model in models:
        if use_model == "simple":
            model = SimpleNN().to(device)
        elif use_model == "complex":
            model = ComplexNN().to(device)
        elif use_model == "medium":
            model = ComplexMediumNN().to(device)
        elif use_model == "large":
            model = ComplexLargeNN().to(device)

        print("Using model: ", use_model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Sample data
        if use_model == "simple":
            X_train = torch.rand((1000, 2)).to(device)
            y_train = torch.sum(X_train, dim=1, keepdim=True).to(device)
        elif use_model == "complex":
            X_train = torch.rand((1000, 100)).to(device)  # 100 features per sample
            y_train = torch.sum(X_train ** 2, dim=1, keepdim=True).to(device)
        elif use_model == "medium":
            # Update input data to match network's expected input size of 100 features
            X_train = torch.rand((10000, 100)).to(device)
            y_train = torch.sum(X_train ** 2, dim=1, keepdim=True).to(device)
        elif use_model == "large":
            # Update input data to match network's expected input size of 100 features
            X_train = torch.rand((200000, 512)).to(device)
            y_train = torch.sum(X_train ** 2, dim=1, keepdim=True).to(device)

        inter_time = time()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Update progress visualization
            loss_values.append(loss.item())
            GUI_CHART.clear()
            GUI_CHART.plot(loss_values, label="Training Loss")
            GUI_CHART.set_title("Training Progress [{} model]" . format(use_model))
            GUI_CHART.set_xlabel("Epoch")
            GUI_CHART.set_ylabel("Loss")
            GUI_CHART.legend()
            GUI_CANVAS.draw()

            # Update the GUI to prevent freezing
            GUI_WINDOW.update()

        end_inter_time = time()
        inter_diff = end_inter_time-inter_time
        inter_per_sec = epochs / inter_diff
        TOTAL_POINTS += POINTS_MAP[use_model]*inter_per_sec

        loss_values = []

    end_time = time()

    GUI_CHART.clear()
    GUI_CHART.plot(loss_values, label="Training Loss")
    GUI_CHART.set_title("Points: {}  Time elapsed: {} seconds" . format(TOTAL_POINTS, round(end_time - start_time)))
    GUI_CHART.set_xlabel("Epoch")
    GUI_CHART.set_ylabel("Loss")
    GUI_CHART.legend()
    GUI_CANVAS.draw()
    GUI_WINDOW.update()

    print("Training completed.")
def run(use_device, models, train_times):
    global GUI_WINDOW, GUI_CANVAS, GUI_CHART

    # Create the main window
    GUI_WINDOW = tk.Tk()
    GUI_WINDOW.title("Training Progress Visualization")

    # Create a Matplotlib figure and embed it in the Tkinter window
    fig, GUI_CHART = plt.subplots()
    GUI_CANVAS = FigureCanvasTkAgg(fig, master=GUI_WINDOW)
    canvas_widget = GUI_CANVAS.get_tk_widget()
    canvas_widget.pack()


    # Add a button to start training
    train_button = tk.Button(GUI_WINDOW, text="Start Training", command=lambda: train_model(use_device, models, epochs=train_times))
    train_button.pack()

    GUI_WINDOW.protocol("WM_DELETE_WINDOW", on_window_close)

    # Run the GUI
    GUI_WINDOW.mainloop()


if __name__ == "__main__":
    train_times = 100
    use_device = 'gpu'  # 'gpu' | 'cpu'   (if 'gpu' it will use GPU only if possible, otherwise it will use CPU)
    available_models = [
        "simple",
        "complex",
        "medium",
        #"large"
    ]


    cuda_available = torch.cuda.is_available()
    print("Cuda available: {} " . format(cuda_available))  # Should return True if GPU is detected
    if cuda_available:
        print(torch.cuda.current_device())  # Should return the device index
        print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Returns GPU name

    run(use_device, available_models, train_times)
