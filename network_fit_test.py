import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Define the target function
def target_function(c, chi=1.0, N1=5, N2=5):
    # Ensure c is within the valid range (0, 1) to avoid log(0)
    c = np.clip(c, 1e-8, 1 - 1e-8)
    return (1 - c) * chi - c * chi + 1/N1 - 1/N2 - np.log(1 - c)/N2 + np.log(c)/N1

depth = 5030
layers = 2

# --- Network Architecture ---
class FEDerivative(nn.Module):
    def __init__(self, layers, depth, activation_fn):
        super().__init__()
        
        module_list = [nn.Linear(1, 50), activation_fn()]
        for _ in range(layers - 1):
            module_list.extend([nn.Linear(50, 30), activation_fn()])
        module_list.append(nn.Linear(30, 1))

        self.mlp = nn.Sequential(*module_list)

    def forward(self, c):
        return self.mlp(c)

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
    "Tanh": FEDerivative(layers, depth, nn.Tanh),
    "Leaky ReLU": FEDerivative(layers, depth, nn.LeakyReLU)
}
predictions = {}
final_losses = {}

# Calculate and print the number of hyperparameters once
# Assuming both models have the same architecture, we can just use one
a_model = list(models.values())[0]
num_params = sum(p.numel() for p in a_model.parameters())
print(f"Network architecture: {layers} hidden layers, {depth} neurons each.")
print(f"Number of trainable parameters: {num_params}")

for name, model in models.items():
    print(f"\n--- Training {name} model ---")
    final_loss = train_network(model, c_tensor, y_tensor)
    final_losses[name] = final_loss
    with torch.no_grad():
        predictions[name] = model(c_tensor).numpy()

# --- Plotting ---
plt.figure(figsize=(12, 8))
plt.plot(c_values, y_values, label="Target Function", linewidth=3, color='black')

colors = ['#1f77b4', '#ff7f0e']
linestyles = ['--', '-.']

for i, (name, pred) in enumerate(predictions.items()):
    final_loss = final_losses[name]
    plt.plot(c_values, pred, label=f"{name} Model (Final Loss: {final_loss:.2e})", linestyle=linestyles[i], color=colors[i], linewidth=2)

# --- CSV Logging ---
csv_file_path = 'network_fit_results.csv'
file_exists = os.path.isfile(csv_file_path)

# Check if the architecture already exists in the CSV
architecture_exists = False
if file_exists:
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        try:
            next(reader)  # Skip header
            for row in reader:
                if int(row[0]) == depth and int(row[1]) == layers:
                    architecture_exists = True
                    print("Architecture already in CSV, skipping write.")
                    break
        except StopIteration:
            # This can happen if the file exists but is empty
            file_exists = False


if not architecture_exists:
    # Write header if file doesn't exist or is empty
    if not file_exists:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['depth', 'layers', 'tanh_loss', 'leakyrelu_loss', 'number_of_hyperparameters'])

    # Append the new data
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([depth, layers, f"{final_losses['Tanh']:.4e}", f"{final_losses['Leaky ReLU']:.4e}", num_params])
        print(f"Appended new data for architecture {depth}x{layers} to {csv_file_path}")

plt.title(f"Activation Function Comparison ({layers}x{depth})")
plt.xlabel("c")
plt.ylabel("f(c)")
plt.legend()
plt.grid(True)
plt.savefig("fit_comparison_"+str(layers)+"x"+str(depth)+".png")
plt.show()

print("\nPlot saved to fit_comparison.png")
