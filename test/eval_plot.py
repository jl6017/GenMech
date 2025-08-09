import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
sns.set_theme(style="whitegrid")

PATH_DICT = {
    # baseline models
    "vit_tiny_t2": "results/2025-07-30_02-13-26",
    "vit_tiny_t2st2": "results/2025-07-30_02-18-39",
    "vit_base_t2": "results/2025-07-31_12-00-57",
    "vit_base_t2st2": "results/2025-08-01_15-20-15",
    "cnn_t2": "results/2025-07-31_13-51-08",
    "cnn_t2st2": "results/2025-07-31_13-53-22",

    # ablation studies
    "vit_tiny_bw_t2": "results/2025-07-30_19-35-32",
    "vit_tiny_bw_t2st2": "results/2025-07-30_19-40-49",

    "vit_tiny_ab1_t2st2": "results/2025-08-01_23-30-08",
    "vit_tiny_ab2_t2st2": "results/2025-08-02_04-01-09",
    "vit_tiny_ab3_t2st2": "results/2025-08-02_04-01-43",

    "vit_tiny_ab1_t2": "results/2025-08-03_02-26-36",
    "vit_tiny_ab2_t2": "results/2025-08-03_02-26-58",
    "vit_tiny_ab3_t2": "results/2025-08-03_02-27-23",

    # beta experiments
    "vit_base_b01_t2": "results/2025-08-04_03-41-11",
    "vit_base_b1_t2": "results/2025-08-05_04-03-54",
    "vit_base_b10_t2": "results/2025-08-06_01-55-31",
    "cnn_b1_t2": "results/2025-08-06_23-45-18",

    # complex datasets
    "vit_tiny_complex_t4": "results/2025-08-06_15-37-38",
    "vit_base_complex_t4": "results/2025-08-07_01-02-40",
}

json_list = []

for i, (key, path) in enumerate(PATH_DICT.items()):
    *model_parts, dataset = key.split("_")
    model = "_".join(model_parts)
    json_list.append({
        "id": i,
        "model": model,
        "dataset": dataset,
        "path": path
    })

with open("experiments.json", "w") as f:
    json.dump(json_list, f, indent=2)


def learning_curve(path_dict: dict):
    for key, result in path_dict.items():
        train_log_file = np.loadtxt(glob(f"{result}/*_training_log_*.txt")[0], skiprows=1)
        validation_log_file = np.loadtxt(glob(f"{result}/*_validation_log_*.txt")[0], skiprows=1)
        print(train_log_file.shape)
        print(validation_log_file.shape)

        df_train = pd.DataFrame(train_log_file, columns=["Epoch", "Avg_Loss", "Recon_c", "Recon_l", "Pred_link", "Pred_curve", "Recon_c_psnr", "Recon_l_psnr", "Pred_link_psnr", "Pred_curve_psnr", "LR"])
        df_val = pd.DataFrame(validation_log_file, columns=["Epoch", "Avg_Loss", "Recon_c", "Recon_l", "Pred_link", "Pred_curve", "Recon_c_psnr", "Recon_l_psnr", "Pred_link_psnr", "Pred_curve_psnr", "LR"])

        plt.figure(figsize=(10, 6))
        # to psnr

        sns.lineplot(data=df_train, x="Epoch", y="Pred_link_psnr", label="Predicted Link PSNR")
        sns.lineplot(data=df_train, x="Epoch", y="Pred_curve_psnr", label="Predicted Curve PSNR")
        sns.lineplot(data=df_val, x="Epoch", y="Pred_link_psnr", label="Validation Predicted Link PSNR", linestyle='--')
        sns.lineplot(data=df_val, x="Epoch", y="Pred_curve_psnr", label="Validation Predicted Curve PSNR", linestyle='--')

        plt.title("Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f"{result}/{key}_learning_curve.png")


if __name__ == "__main__":
    learning_curve(PATH_DICT)

