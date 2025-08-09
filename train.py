import os
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from tqdm import tqdm   
from model.VAE_VIT import PairwiseViTVAE
from model.VAE_CNN import PairwiseCNNVAE
from model.loss import pairvae_loss
from model.dataset import get_dataloader
from model.validation import validate 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import time
from glob import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
complex_dataset = True
if complex_dataset:
    # get complex path list, every subfolder in complex/
    DATAPATH = "./../dataset/complex_t4/"
    complex_folders = sorted(glob(os.path.join(DATAPATH, "*")))
    DATASET = [os.path.basename(folder) for folder in complex_folders if os.path.isdir(folder)]
else:
    DATAPATH = "./../dataset/"
    DATASET = ["tri_2"]  # , "stri_2"


def train(model, optimizer, scheduler, epochs, device, save_interval, beta, ls_weight, gamma):

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    if complex_dataset:
        log_train = os.path.join(save_dir, f"base_b10_training_log_complex.txt")
        log_val = os.path.join(save_dir, f"base_b10_validation_log_complex.txt")
    else:
        log_train = os.path.join(save_dir, f"base_b10_training_log_{DATASET}.txt")
        log_val = os.path.join(save_dir, f"base_b10_validation_log_{DATASET}.txt")

    # header = f"{'Epoch':<6}{'TrainLoss':<15}{'Recon':<15}{'KL':<15}" \
    #         f"{'LatSim':<15}{'Pred_link':<15}{'Pred_curve':<15}{'LR':<10}\n"
    with open(log_train, "w") as log:
        log.write(
            f"{'Epoch':<6}{'Avg_Loss':<15}" \
            f"{'Recon_c':<15}{'Recon_l':<15}{'Pred_link':<15}{'Pred_curve':<15}" \
            f"{'Recon_c_psnr':<15}{'Recon_l_psnr':<15}{'Pred_link_psnr':<15}{'Pred_curve_psnr':<15}" \
            f"{'LR':<10}\n"
        )

    with open(log_val, "w") as log:
        log.write(
            f"{'Epoch':<6}{'Avg_Loss':<15}" \
            f"{'Recon_c':<15}{'Recon_l':<15}{'Pred_link':<15}{'Pred_curve':<15}" \
            f"{'Recon_c_psnr':<15}{'Recon_l_psnr':<15}{'Pred_link_psnr':<15}{'Pred_curve_psnr':<15}" \
            f"{'LR':<10}\n"
        )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        overall_loss = 0.0
        overall_recon_c = 0.0
        overall_recon_l = 0.0
        # overall_kl = 0.0
        # overall_lat = 0.0
        overall_pred_link = 0.0
        overall_pred_curve = 0.0

        # psnr
        overall_recon_c_psnr = 0.0
        overall_recon_l_psnr = 0.0
        overall_pred_link_psnr = 0.0
        overall_pred_curve_psnr = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for curve_img, link_img, name in progress_bar:
            curve_img = curve_img.to(device)
            link_img =  link_img.to(device)
            optimizer.zero_grad()

            out = model(curve_img, link_img)    

            z_curve_det = out["z_curve"]
            pred_link = model.link_decoder(z_curve_det )

            z_link_det = out["z_link"]
            pred_curve = model.curve_decoder(z_link_det )

            loss, loss_log = pairvae_loss(out, curve_img, link_img, pred_link, pred_curve,
                                                beta=beta, lam=ls_weight, gamma=gamma)
            loss.backward()
            optimizer.step()

            overall_loss += loss.item()
            overall_recon_c += loss_log["recon_c"].item()
            overall_recon_l += loss_log["recon_l"].item()
            overall_pred_link += loss_log["pred_link_loss"].item()
            overall_pred_curve += loss_log["pred_curve_loss"].item()
            # psnr
            overall_recon_c_psnr += loss_log["recon_c_psnr"].item()
            overall_recon_l_psnr += loss_log["recon_l_psnr"].item()
            overall_pred_link_psnr += loss_log["pred_link_psnr"].item()
            overall_pred_curve_psnr += loss_log["pred_curve_psnr"].item()

            progress_bar.set_postfix(loss=loss.item())


        # loss mse
        n_batches = len(train_loader)
        avg_train_loss  = overall_loss  / n_batches
        avg_train_recon_c = overall_recon_c / n_batches
        avg_train_recon_l = overall_recon_l / n_batches
        # avg_train_kl    = overall_kl    / n_batches
        # avg_train_lat    = overall_lat    / n_batches
        avg_train_pred_link = overall_pred_link / n_batches
        avg_train_pred_curve = overall_pred_curve / n_batches
        # psnr
        avg_train_recon_c_psnr = overall_recon_c_psnr / n_batches
        avg_train_recon_l_psnr = overall_recon_l_psnr / n_batches
        avg_train_pred_link_psnr = overall_pred_link_psnr / n_batches
        avg_train_pred_curve_psnr = overall_pred_curve_psnr / n_batches


        avg_val_loss, mse_loss, psnr_loss = validate(model, val_loader, epoch, save_dir, save_interval=save_interval)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}, lr: {current_lr:.6f}")

        row_train = f"{epoch:<8}{avg_train_loss:<15.8f}" \
            f"{avg_train_recon_c:<15.8f}{avg_train_recon_l:<15.8f}" \
            f"{avg_train_pred_link:<15.8f}{avg_train_pred_curve:<15.8f}" \
            f"{avg_train_recon_c_psnr:<15.8f}{avg_train_recon_l_psnr:<15.8f}" \
            f"{avg_train_pred_link_psnr:<15.8f}{avg_train_pred_curve_psnr:<15.8f}" \
            f"{current_lr:<10.6f}\n"
        with open(log_train, "a") as log:
            log.write(row_train)

        row_val = f"{epoch:<8}{avg_val_loss:<15.8f}" \
            f"{mse_loss[0]:<15.8f}{mse_loss[1]:<15.8f}{mse_loss[2]:<15.8f}{mse_loss[3]:<15.8f}" \
            f"{psnr_loss[0]:<15.8f}{psnr_loss[1]:<15.8f}{psnr_loss[2]:<15.8f}{psnr_loss[3]:<15.8f}" \
            f"{current_lr:<10.6f}\n"
        with open(log_val, "a") as log:
            log.write(row_val)

        # save the best validation model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model at epoch {epoch} with loss {best_val_loss:.8f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(save_dir, f"best_model.pth"))

    return model


if __name__ == "__main__":
    start_time = time.time()

    train_loader, val_loader = get_dataloader(
        batch_size=BATCH_SIZE,
        datapath=DATAPATH,
        dataset=DATASET,
        num_workers=8,
    )

    # Initialize model, optimizer, and scheduler
    model = PairwiseViTVAE().to(device)  # run ViT version
    # model = PairwiseCNNVAE().to(device)  # run cnn version

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    model = train(model, optimizer, scheduler, epochs=100, device=device, save_interval=5, beta=0.1, ls_weight=10, gamma=10)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"used time  {elapsed_time:.2f} second")


