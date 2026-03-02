import flwr as fl
from flwr.server import ServerConfig
from torch.utils.tensorboard import SummaryWriter
import json
import os

# Create runs directory if it doesn't exist
os.makedirs("./runs", exist_ok=True)
global_history = {"round": [], "accuracy": [], "loss": [], "f1": []}
# Initialize TensorBoard Writer
writer = SummaryWriter(log_dir="./runs/seizure_tracking")

class DashboardStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        # Catch the evaluation metrics from all 10 edge devices
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        
        if aggregated_metrics is not None and "accuracy" in aggregated_metrics:
            acc = aggregated_metrics["accuracy"]
            f1 = aggregated_metrics["f1_score"]
            # Stream live to Dashboard!
            writer.add_scalar("Global/Accuracy", acc, rnd)
            writer.add_scalar("Global/F1_Score", f1, rnd)
            writer.add_scalar("Global/Loss", aggregated_loss, rnd)
            print(f"[Round {rnd}] Dashboard Updated - Acc: {acc:.4f} | F1: {f1:.4f}")
            global_history["round"].append(rnd)
            global_history["accuracy"].append(acc)
            global_history["loss"].append(aggregated_loss)
            global_history["f1"].append(f1)
            with open("./runs/global_history.json", "w") as f:
                json.dump(global_history, f)
            
        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    print("Starting Central Server & TensorBoard Logger...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=15),
        strategy=DashboardStrategy(
            min_fit_clients=10,      # Wait for exactly 10 edge devices
            min_available_clients=10,
            min_evaluate_clients=10, # Force all 10 to report their metrics
            evaluate_metrics_aggregation_fn=lambda metrics: {
                "accuracy": sum([num * m["accuracy"] for num, m in metrics]) / sum([num for num, _ in metrics]),
                "f1_score": sum([num * m["f1"] for num, m in metrics]) / sum([num for num, _ in metrics])
            }
        )
    )
    writer.close()
