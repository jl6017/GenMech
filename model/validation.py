from torchvision.utils import save_image
import os, torch
from .loss import pairvae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, val_loader, epoch, save_dir, save_interval,
             beta=0.0001, lam=0.5, gamma = 10, num_save=50):
    model.eval()
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
    saved = 0

    with torch.no_grad():
        for curve_img, link_img, names in val_loader:         
            curve_img = curve_img.to(device)
            link_img = link_img.to(device)
            out = model(curve_img, link_img)

            z_curve_det = out["z_curve"].detach()
            pred_link = model.link_decoder(z_curve_det )

            z_link_det = out["z_link"].detach()
            pred_curve= model.curve_decoder(z_link_det )

            loss, loss_log = pairvae_loss(out, curve_img, link_img, pred_link,pred_curve,
                                                beta=beta, lam=lam,gamma = gamma)
            
            # mse
            overall_loss  += loss.item()
            overall_recon_c += loss_log["recon_c"].item()
            overall_recon_l += loss_log["recon_l"].item()
            overall_pred_link += loss_log["pred_link_loss"].item()
            overall_pred_curve += loss_log["pred_curve_loss"].item()
            # psnr
            overall_recon_c_psnr += loss_log["recon_c_psnr"].item()
            overall_recon_l_psnr += loss_log["recon_l_psnr"].item()
            overall_pred_link_psnr += loss_log["pred_link_psnr"].item()
            overall_pred_curve_psnr += loss_log["pred_curve_psnr"].item()

            # ───── save each image individually ─────
            if epoch == 1 or (epoch % save_interval == 0 and saved < num_save):
                m = min(curve_img.size(0), num_save - saved)
                save_path = os.path.join(save_dir, f"epoch_{epoch}")
                os.makedirs(save_path, exist_ok=True)

                for i in range(m):
                    base = names[i]
                    print(f"Saving {base} at epoch {epoch}")
                    save_image(curve_img[i],
                               os.path.join(save_path, f"{base}_curve_gt.png"))
                    save_image(out["rec_curve"][i],
                               os.path.join(save_path, f"{base}_curve_output.png"))
                    save_image(pred_curve[i],
                               os.path.join(save_path, f"{base}_curve_pred.png"))
                    save_image(link_img[i],
                               os.path.join(save_path, f"{base}_link_gt.png"))
                    save_image(out["rec_link"][i],
                               os.path.join(save_path, f"{base}_link_output.png"))
                    save_image(pred_link[i],
                               os.path.join(save_path, f"{base}_link_pred.png"))
                saved += m

    n_batches = len(val_loader)

    avg_val_loss  = overall_loss  / n_batches
    avg_val_recon_c = overall_recon_c / n_batches
    avg_val_recon_l = overall_recon_l / n_batches
    avg_val_pred_link = overall_pred_link / n_batches
    avg_val_pred_curve = overall_pred_curve / n_batches
    # psnr
    avg_val_recon_c_psnr = overall_recon_c_psnr / n_batches
    avg_val_recon_l_psnr = overall_recon_l_psnr / n_batches
    avg_val_pred_link_psnr = overall_pred_link_psnr / n_batches
    avg_val_pred_curve_psnr = overall_pred_curve_psnr / n_batches

    mse_loss = [avg_val_recon_c, avg_val_recon_l, avg_val_pred_link, avg_val_pred_curve]
    psnr_loss = [avg_val_recon_c_psnr, avg_val_recon_l_psnr, avg_val_pred_link_psnr, avg_val_pred_curve_psnr]

    print(f"Total {avg_val_loss:.8f} | [Val] Pred_link {avg_val_pred_link :.8f} | Pred_curve {avg_val_pred_curve :.8f}")
    return avg_val_loss, mse_loss, psnr_loss