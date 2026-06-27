import os
import matplotlib.pyplot as plt
import numpy as np

class RoleMetricsLogger:
    def __init__(self, log_dir="logs", plot_dir="plots"):
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        self.metrics = {}

    def log_metric(self, name, value, step):
        """Logs a metric value to memory and a CSV file."""
        if name not in self.metrics:
            self.metrics[name] = {"steps": [], "values": []}
        self.metrics[name]["steps"].append(step)
        self.metrics[name]["values"].append(value)
        
        file_path = os.path.join(self.log_dir, f"{name}.csv")
        # Write header if new file
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("step,value\n")
                
        with open(file_path, "a") as f:
            f.write(f"{step},{value}\n")

    def plot_metrics(self):
        """Generates plots for all logged metrics and saves them to the plot directory."""
        for name, data in self.metrics.items():
            if not data["steps"]:
                continue
            plt.figure()
            plt.plot(data["steps"], data["values"], label=name)
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.title(f"{name} Over Time")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.plot_dir, f"{name}.png"))
            plt.close()
