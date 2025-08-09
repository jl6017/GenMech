import numpy as np
import torch, os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from model.VAE_VIT import PairwiseViTVAE
from model.VAE_CNN import PairwiseCNNVAE
import json
from glob import glob
import torch.nn as nn
from tqdm import tqdm
mse = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1999)
torch.manual_seed(1999)
N_TEST = 100  # number of test samples to evaluate
N_GEN = 9

test_curve_T2 = sorted(glob("./../dataset/test_0807/tri_2/images/curve/*.png"))
test_mechanism_T2 = sorted(glob("./../dataset/test_0807/tri_2/images/mechanism/*.png"))
test_curve_T2ST2 = sorted(glob("./../dataset/test_0807/*/images/curve/*.png"))
test_mechanism_T2ST2 = sorted(glob("./../dataset/test_0807/*/images/mechanism/*.png"))
test_curve_T4 = sorted(glob("./../dataset/test_0807/complex_T4/*/images/curve/*.png"))
test_mechanism_T4 = sorted(glob("./../dataset/test_0807/complex_T4/*/images/mechanism/*.png"))
print(f"Test size (T2, ST2, T4): {len(test_curve_T2)}, {len(test_curve_T2ST2)}, {len(test_curve_T4)}")
indices_1 = np.random.choice(len(test_curve_T2), N_TEST, replace=False)
indices_2 = np.random.choice(len(test_curve_T2ST2), N_TEST, replace=False)  # to align the sample indices, only sample once
indices_3 = np.random.choice(len(test_curve_T4), N_TEST, replace=False)

train_curve_T2 = sorted(glob("./../dataset/tri_2/images/curve/*.png"))
train_mechanism_T2 = sorted(glob("./../dataset/tri_2/images/mechanism/*.png"))
train_curve_T2ST2 = sorted(glob("./../dataset/*/images/curve/*.png"))
train_mechanism_T2ST2 = sorted(glob("./../dataset/*/images/mechanism/*.png"))
train_curve_T4 = sorted(glob("./../dataset/complex_T4/*/images/curve/*.png"))
train_mechanism_T4 = sorted(glob("./../dataset/complex_T4/*/images/mechanism/*.png"))
print(f"Train size (T2, ST2, T4): {len(train_curve_T2)}, {len(train_curve_T2ST2)}, {len(train_curve_T4)}")  # to get the latent distribution

with open("experiments.json", "r") as f:
    experiments = json.load(f)

def mse2psnr(mse_batch, max_value=1.0):
    """Convert MSE to PSNR."""
    return 20 * torch.log10(max_value / torch.sqrt(mse_batch + 1e-8))

def bw_filter(image_tensor, threshold=0.001):
    """Convert RGB image to black and white 3 channel image, given a threshold."""
    # print(image_tensor.shape)
    bw_mask = (image_tensor > threshold).float()  # [B, C, H, W]
    return bw_mask

def eval_model(experiment, num_samples, num_latents, num_gen, test_model=True, latent_gen=True, bw=False):
    model_type = experiment["model"]
    dataset = experiment["dataset"]
    path = experiment["path"]
    eval_dir = f"evaluation/{experiment['id']}"
    os.makedirs(eval_dir, exist_ok=True)

    ## Select test samples
    if dataset == "t2":
        test_indices = indices_1
        curve_paths = test_curve_T2
        mechanism_paths = test_mechanism_T2
        train_curve_paths = train_curve_T2
        train_mechanism_paths = train_mechanism_T2
    elif dataset == "t2st2":
        test_indices = indices_2
        curve_paths = test_curve_T2ST2
        mechanism_paths = test_mechanism_T2ST2
        train_curve_paths = train_curve_T2ST2
        train_mechanism_paths = train_mechanism_T2ST2
    elif dataset == "t4":
        test_indices = indices_3
        curve_paths = test_curve_T4
        mechanism_paths = test_mechanism_T4
        train_curve_paths = train_curve_T4
        train_mechanism_paths = train_mechanism_T4
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    ## Load model
    if model_type.startswith("vit_tiny"):
        model = PairwiseViTVAE(
            num_heads = 6,
            model_name = "vit_tiny_patch16_224",
        ).to(device)
    elif model_type.startswith("vit_base"):
        model = PairwiseViTVAE(
            num_heads = 12,
            model_name = "vit_base_patch16_224",
        ).to(device)
    elif model_type.startswith("cnn"):
        model = PairwiseCNNVAE().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt_path = os.path.join(path, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ## c2m, m2c, c2m2c evaluation
    if test_model:
        c2m_psnr_list, m2c_psnr_list, c2m2c_psnr_list = [], [], []
        for index in tqdm(test_indices, desc="Evaluating"):
            c2m_psnr = c2m(model, index, curve_paths, mechanism_paths, eval_dir, bw)
            m2c_psnr = m2c(model, index, curve_paths, mechanism_paths, eval_dir, bw)
            c2m2c_psnr = c2m2c(model, index, curve_paths, mechanism_paths, eval_dir, bw)

            c2m_psnr_list.append(c2m_psnr)
            m2c_psnr_list.append(m2c_psnr)
            c2m2c_psnr_list.append(c2m2c_psnr)

        c2m_mean = np.mean(c2m_psnr_list)
        m2c_mean = np.mean(m2c_psnr_list)
        c2m2c_mean = np.mean(c2m2c_psnr_list)

        c2m_std = np.std(c2m_psnr_list)
        m2c_std = np.std(m2c_psnr_list)
        c2m2c_std = np.std(c2m2c_psnr_list)

        with open(f"evaluation/{experiment['id']}.txt", "w") as f:
            f.write(f"Model: {model_type}, Dataset: {dataset}\n")
            f.write(f"C2M PSNR: {c2m_mean:.4f}, {c2m_std:.4f}\n")
            f.write(f"M2C PSNR: {m2c_mean:.4f}, {m2c_std:.4f}\n")
            f.write(f"C2M2C PSNR: {c2m2c_mean:.4f}, {c2m2c_std:.4f}\n")

    ## Latent space visualization
    if latent_gen:
        latent_generation(model, train_curve_paths, train_mechanism_paths, eval_dir, num_latents=num_latents, num_gen=num_gen)

def c2m(model, index, curve_paths, mechanism_paths, output_path, bw):
    """
    curve to mechanism
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    curve_img = tf(Image.open(curve_paths[index]).convert("RGB")).unsqueeze(0).to(device)  # [1, 3, H, W]
    mechanism_img = tf(Image.open(mechanism_paths[index]).convert("RGB")).unsqueeze(0).to(device)  # [1, 3, H, W]

    if bw:
        curve_img = bw_filter(curve_img)
        mechanism_img = bw_filter(mechanism_img)

    with torch.no_grad():
        z_curve, mu, logvar = model.encode(
            model.curve_encoder,
            curve_img,
            model.curve_mu_mlp,
            model.curve_logvar_mlp
        )  # [1, 196, D]
        # print(z_curve.shape, mu.shape, logvar.shape)

        pred_mechanism = model.link_decoder(z_curve)  # [1, 3, H, W]

    concat_img = torch.cat([curve_img, pred_mechanism, mechanism_img], dim=0)  # [3, 3, H, W], input, pred, gt
    c2m_mse = mse(pred_mechanism, mechanism_img)
    c2m_psnr = mse2psnr(c2m_mse)

    save_image(concat_img, f"{output_path}/c2m_{index:06}.png", nrow=3)
    return c2m_psnr.item()


def m2c(model, index, curve_paths, mechanism_paths, output_path, bw):
    """
    mechanism to curve
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    mechanism_img = tf(Image.open(mechanism_paths[index]).convert("RGB")).unsqueeze(0).to(device)  # [1, 3, H, W]
    curve_img = tf(Image.open(curve_paths[index]).convert("RGB")).unsqueeze(0).to(device)  # [1, 3, H, W]

    if bw:
        mechanism_img = bw_filter(mechanism_img)
        curve_img = bw_filter(curve_img)

    with torch.no_grad():
        z_mechanism, mu, logvar = model.encode(
            model.link_encoder,
            mechanism_img,
            model.link_mu_mlp,
            model.link_logvar_mlp
        )  # [1, 196, D]

        pred_curve = model.curve_decoder(z_mechanism)  # [1, 3, H, W]

    concat_img = torch.cat([mechanism_img, pred_curve, curve_img], dim=0)  # [3, 3, H, W], input, pred, gt
    m2c_mse = mse(pred_curve, curve_img)
    m2c_psnr = mse2psnr(m2c_mse)

    save_image(concat_img, f"{output_path}/m2c_{index:06}.png", nrow=3)
    return m2c_psnr.item()


def c2m2c(model, index, curve_paths, mechanism_paths, output_path, bw):
    """
    curve to mechanism to curve, for testing the roll-out of predicted mechanism
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    curve_img = tf(Image.open(curve_paths[index]).convert("RGB")).unsqueeze(0).to(device)  # [1, 3, H, W]
    # mechanism_img = tf(Image.open(mechanism_paths[index]).convert("RGB")).unsqueeze(0).to(device)  # [1, 3, H, W]

    if bw:
        curve_img = bw_filter(curve_img)
        # mechanism_img = bw_filter(mechanism_img)

    with torch.no_grad():
        z_curve, mu, logvar = model.encode(
            model.curve_encoder,
            curve_img,
            model.curve_mu_mlp,
            model.curve_logvar_mlp
        )  # [1, 196, D]

        pred_mechanism = model.link_decoder(z_curve)  # [1, 3, H, W]

    with torch.no_grad():
        z_mechanism, mu, logvar = model.encode(
            model.link_encoder,
            pred_mechanism,
            model.link_mu_mlp,
            model.link_logvar_mlp
        )  # [1, 196, D]

        pred_curve = model.curve_decoder(z_mechanism)  # [1, 3, H, W]

    concat_img = torch.cat([curve_img, pred_mechanism, pred_curve, curve_img], dim=0)  # [4, 3, H, W], input, pred, pred, gt
    c2m2c_mse = mse(pred_curve, curve_img)
    c2m2c_psnr = mse2psnr(c2m2c_mse)

    save_image(concat_img, f"{output_path}/c2m2c_{index:06}.png", nrow=4)
    return c2m2c_psnr.item()

def latent_generation(model, train_curve_paths, train_mechanism_paths, output_path, num_latents, num_gen):
    """
    Generate random latent vectors and decode them to images.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    model.eval()
    random_image_input = torch.randn(1, 3, 224, 224).to(device)  # [1, 3, H, W], to get the latent shape
    with torch.no_grad():
        z, mu, logvar = model.encode(
            model.curve_encoder,
            random_image_input,
            model.curve_mu_mlp,
            model.curve_logvar_mlp
        )
    print(f"Latent shape: {z.shape}")  # [1, 196, D]
    latent_shape = list(z.shape)
    latent_shape[0] = num_gen  # [num_gen, ...]

    # get training set mean and std
    mu_c_list, mu_m_list = [], []
    with torch.no_grad():
        for cp, mp in tqdm(zip(train_curve_paths[:num_latents], train_mechanism_paths[:num_latents]), desc="Calculating Latent Mean/Std", total=num_latents):
            curve_img = Image.open(cp).convert("RGB")
            mechanism_img = Image.open(mp).convert("RGB")
            curve_tensor = tf(curve_img).unsqueeze(0).to(device)
            mechanism_tensor = tf(mechanism_img).unsqueeze(0).to(device)

            z_curve, mu_c, _ = model.encode(
                model.curve_encoder,
                curve_tensor,
                model.curve_mu_mlp,
                model.curve_logvar_mlp
            )
            z_mechanism, mu_m, _ = model.encode(
                model.link_encoder,
                mechanism_tensor,
                model.link_mu_mlp,
                model.link_logvar_mlp
            )
            mu_c_list.append(mu_c)
            mu_m_list.append(mu_m)

    mu_c_list = torch.stack(mu_c_list, dim=0)  # [num_latents, 196, D]
    mu_m_list = torch.stack(mu_m_list, dim=0)  # [num_latents, 196, D]

    mu_c = torch.mean(mu_c_list, dim=0)  # [196, D]
    mu_m = torch.mean(mu_m_list, dim=0)  # [196, D]
    std_c = torch.std(mu_c_list, dim=0)  # [196, D]
    std_m = torch.std(mu_m_list, dim=0)  # [196, D]
    print(f"Latent Mean (Curve): {mu_c.shape}, Std: {std_c.shape}")
    print(f"Latent Mean (Mechanism): {mu_m.shape}, Std: {std_m.shape}")

    random_eps = torch.randn(latent_shape).to(device)
    generated_curves = model.curve_decoder(mu_c + std_c * random_eps)  # [num_samples, 3, H, W]
    generated_mechanisms = model.link_decoder(mu_m + std_m * random_eps)  # [num_samples, 3, H, W]
    # generated_curves = model.curve_decoder(mu_c)  # [num_samples, 3, H, W]
    # generated_mechanisms = model.link_decoder(mu_m)  # [num_samples, 3, H, W]

    save_image(generated_curves, f"{output_path}/_latent_generation_curves.png", nrow=3)
    save_image(generated_mechanisms, f"{output_path}/_latent_generation_mechanisms.png", nrow=3)
    print(f"Generated {num_gen} random images from latent space.")


if __name__ == "__main__":
    for i, experiment in enumerate(experiments[:-2]):  # skip the last two experiments for now
        if i == 6 or i == 7:
            black_and_white = True
        else:
            black_and_white = False
        print(f"Evaluating experiment {i+1}/{len(experiments)}: {experiment['model']}, {experiment['dataset']}")
        eval_model(experiment, num_samples=N_TEST, num_latents=1000, num_gen=N_GEN, test_model=False, latent_gen=True, bw=black_and_white)
