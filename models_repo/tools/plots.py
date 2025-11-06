import pandas as pd
import json
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def save_results(args, file_name, metrics_test, elapsed, preds_last, trues_last, metrics_val):
    base_folder = f"results/{file_name}/"
    pd.DataFrame(preds_last).to_csv(base_folder+args.save_name+"_preds.csv", index=False)
    trues_path = base_folder+"trues.csv"
    if not any("trues" in os.path.basename(f) for f in glob.glob(base_folder+"*")):
        pd.DataFrame(trues_last).to_csv(trues_path, index=False)

    # déballage des métriques test
    mae_test, mse_test, rmse_test, mape_test, mspe_test, rse_test, corr_test, kge_test = metrics_test
    # déballage des métriques val
    mae_val, mse_val, rmse_val, mape_val, mspe_val, rse_val, corr_val, kge_val = metrics_val

    metrics_dict = {
        "test": {
            "MAE": float(mae_test), "MSE": float(mse_test), "RMSE": float(rmse_test),
            "MAPE": float(mape_test), "MSPE": float(mspe_test), "RSE": float(rse_test),
            "CORR": float(corr_test), "KGE": float(kge_test), "elapsed": float(elapsed)
        },
        "val": {
            "MAE": float(mae_val), "MSE": float(mse_val), "RMSE": float(rmse_val),
            "MAPE": float(mape_val), "MSPE": float(mspe_val), "RSE": float(rse_val),
            "CORR": float(corr_val), "KGE": float(kge_val), "elapsed": float(elapsed)
        }
    }

    metrics_file = base_folder+"metrics.json"
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                content = f.read().strip()
                all_metrics = json.loads(content) if content else {}
        except json.JSONDecodeError:
            all_metrics = {}
    else:
        all_metrics = {}

    all_metrics[args.save_name] = metrics_dict
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)



def plot_selection_distribution(args,selection_counter):
    sorted_items = sorted(selection_counter.items(), key=lambda x: x[1], reverse=True)
    ids, counts = zip(*sorted(selection_counter.items()))
    plt.figure(figsize=(10, 5))
    plt.scatter(ids, counts, s=10, c="blue", alpha=0.7)
    plt.xlabel("Data points")
    plt.ylabel("Selection count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"figs/Buffer_distribution_{args.save_name}.png", dpi=300, format='png')
    plt.show()


def plot_module_on_series(trues_last, global_focus, args, file_name, module):
    series = trues_last[:, 0] if trues_last.ndim > 1 else trues_last

    focus = (global_focus - global_focus.min()) / (global_focus.max() - global_focus.min() + 1e-8)
    focus = np.sqrt(focus)
    
    # --- Alignement des tailles ---
    min_len = min(len(series), len(focus))
    series = series[:min_len]
    focus = focus[:min_len]
    
    # --- Top-k absolute selection ---
    to = args.attn_topk / 100
    k = int(to * len(focus))
    topk_idx = np.argsort(focus)[-k:]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series, color="blue", alpha=0.7, label="Time series")

    # Highlight top-k points
    ax.scatter(topk_idx, series[topk_idx], color="red" if module=="attn" else "green", s=15, label=f"Top {args.attn_topk:.0f}% attention")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()

    plt.tight_layout()
    save_path = f"figs/{args.save_name}_{module}_series.png"
    plt.savefig(save_path, dpi=300)
    plt.show()