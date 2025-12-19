
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the target function
def target_function(c, chi=1.0, N1=5, N2=5):
    # Ensure c is within the valid range (0, 1) to avoid log(0)
    c = np.clip(c, 1e-8, 1 - 1e-8)
    return (1 - c) * chi - c * chi + 1/N1 - 1/N2 - np.log(1 - c)/N2 + np.log(c)/N1

# --- Network Architectures ---
class FEDerivative_Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
    def forward(self, c): return self.mlp(c)

class FEDerivative_LeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 50), nn.LeakyReLU(),
            nn.Linear(50, 50), nn.LeakyReLU(),
            nn.Linear(50, 1)
        )
    def forward(self, c): return self.mlp(c)

class FEDerivative_GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 50), nn.GELU(),
            nn.Linear(50, 50), nn.GELU(),
            nn.Linear(50, 1)
        )
    def forward(self, c): return self.mlp(c)

class FEDerivative_SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 1)
        )
    def forward(self, c): return self.mlp(c)

# Training function
def train_network(model, data, targets, epochs=10000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        if epoch % 2000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return loss.item()

# Generate data
c_values = np.linspace(0.01, 0.99, 200)
y_values = target_function(c_values)

c_tensor = torch.from_numpy(c_values).view(-1, 1).float()
y_tensor = torch.from_numpy(y_values).view(-1, 1).float()

# --- Main Execution ---
models = {
    "Tanh": FEDerivative_Tanh(),
    "Leaky ReLU": FEDerivative_LeakyReLU(),
    "GELU": FEDerivative_GELU(),
    "SiLU": FEDerivative_SiLU()
}
predictions = {}
final_losses = {}

for name, model in models.items():
    print(f"\n--- Training {name} model ---")
    final_loss = train_network(model, c_tensor, y_tensor)
    final_losses[name] = final_loss
    with torch.no_grad():
        predictions[name] = model(c_tensor).numpy()

# --- Plotting ---
plt.figure(figsize=(12, 8))
plt.plot(c_values, y_values, label="Target Function", linewidth=3, color='black')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linestyles = ['--', '-.', ':', '-']

for i, (name, pred) in enumerate(predictions.items()):
    final_loss = final_losses[name]
    plt.plot(c_values, pred, label=f"{name} Model (Final Loss: {final_loss:.2e})", linestyle=linestyles[i], color=colors[i], linewidth=2)

plt.title("Activation Function Comparison")
plt.xlabel("c")
plt.ylabel("f(c)")
plt.legend()
plt.grid(True)
plt.savefig("fit_comparison_2x50.png")
print("\nPlot saved to fit_comparison.png")
