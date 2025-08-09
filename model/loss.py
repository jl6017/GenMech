import torch
import torch.nn.functional as F
import torch.nn as nn
mse = nn.MSELoss()

def mse2psnr(mse_batch, max_value=1.0):
    """Convert MSE to PSNR."""
    return 20 * torch.log10(max_value / torch.sqrt(mse_batch + 1e-8))

def pairvae_loss(out, curve_img, link_img, pred_link, pred_curve, beta=0.0001, lam=10, gamma=10):

    recon_c = mse(out["rec_curve"], curve_img)
    recon_l = mse(out["rec_link"],  link_img)
    recon   = recon_c + recon_l

    recon_c_psnr = mse2psnr(recon_c)
    recon_l_psnr = mse2psnr(recon_l)

    # KL 
    kl_c = (-0.5 * (1 + out["logvar_curve"] - out["mu_curve"].pow(2) - out["logvar_curve"].exp())).mean()
    kl_l = (-0.5 * (1 + out["logvar_link"]  - out["mu_link"].pow(2) - out["logvar_link"].exp())).mean()
    kl = kl_c + kl_l

    # latent similarity  
    lat_sim = torch.mean((out["z_curve"] - out["z_link"]) ** 2)
    # pred loss pred_link with gt link

    pred_link_loss = mse(pred_link, link_img)
    pred_curve_loss = mse(pred_curve, curve_img)

    pred_link_psnr = mse2psnr(pred_link_loss)
    pred_curve_psnr = mse2psnr(pred_curve_loss)

    loss = recon + beta * kl + lam * lat_sim + gamma * pred_link_loss + gamma * pred_curve_loss
    loss_log = {
        "loss": loss,
        "recon_c": recon_c.detach(),
        "recon_l": recon_l.detach(),
        "kl": kl.detach(),
        "lat_sim": lat_sim.detach(),
        "pred_link_loss": pred_link_loss.detach(),
        "pred_curve_loss": pred_curve_loss.detach(),
        "recon_c_psnr": recon_c_psnr.detach(),
        "recon_l_psnr": recon_l_psnr.detach(),
        "pred_link_psnr": pred_link_psnr.detach(),
        "pred_curve_psnr": pred_curve_psnr.detach(),
    }

    return loss, loss_log