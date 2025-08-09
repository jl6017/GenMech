import torch
import torch.nn as nn
from timm import create_model


class SimpleDecoderCNN(nn.Module):
    def __init__(self, in_dim=512, out_chans=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, kernel_size=4, stride=2, padding=1),  # 14 → 28
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 28 → 56
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 56 → 112
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 112 → 224
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 224 → 448 (then crop to 224)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, out_chans, kernel_size=3, padding=1),  # final 3-channel image
            nn.Tanh()
        )

    def forward(self, z):  # [B, 512, 14, 14]
        x = self.decoder(z)
        # # Crop from 448x448 to 224x224 if needed
        # if x.size(-1) > 224:
        #     x = x[:, :, :224, :224]
        return x


class PairwiseCNNVAE(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()

        # ───── Encoder ─────
        self.curve_encoder = create_model(
            "resnet18", pretrained=True, num_classes=0, in_chans=in_chans, global_pool=""
        )
        self.link_encoder = create_model(
            "resnet18", pretrained=True, num_classes=0, in_chans=in_chans, global_pool=""
        )
        self.latent_dim = 512  # default from resnet18
        self.spatial_hw = 14   # feature map size from resnet18 with 224x224 input

        # ───── MLPs for μ and logσ² ─────
        hidden_dim = self.latent_dim * 2

        def create_mlp():
            return nn.Sequential(
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.latent_dim)
            )

        self.curve_mu_mlp = create_mlp()
        self.curve_logvar_mlp = create_mlp()
        self.link_mu_mlp = create_mlp()
        self.link_logvar_mlp = create_mlp()

        # ───── Decoder ─────
        self.curve_decoder = SimpleDecoderCNN(in_dim=self.latent_dim)
        self.link_decoder = SimpleDecoderCNN(in_dim=self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, encoder, x, mu_mlp, logvar_mlp):
        feats = encoder.forward_features(x)  # [B, 512, 14, 14]
        B, C, H, W = feats.shape

        # flatten spatial structure into sequence like ViT
        tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 196, 512]

        mu = mu_mlp(tokens)         # [B, 196, 512]
        logvar = logvar_mlp(tokens) # [B, 196, 512]
        z = self.reparameterize(mu, logvar)  # [B, 196, 512]

        z_feat = z.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, 512, 14, 14]
        mu_feat = mu.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, 512, 14, 14], only for visualization
        logvar_feat = logvar.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, 512, 14, 14]
        return z_feat, mu_feat, logvar_feat

    def forward(self, curve_img, link_img):
        z_curve, mu_curve, log_curve = self.encode(
            self.curve_encoder, curve_img, self.curve_mu_mlp, self.curve_logvar_mlp
        )
        z_link, mu_link, log_link = self.encode(
            self.link_encoder, link_img, self.link_mu_mlp, self.link_logvar_mlp
        )

        rec_curve = self.curve_decoder(z_curve)
        rec_link = self.link_decoder(z_link)

        return {
            "rec_curve": rec_curve,
            "rec_link": rec_link,
            "mu_curve": mu_curve,
            "logvar_curve": log_curve,
            "mu_link": mu_link,
            "logvar_link": log_link,
            "z_curve": z_curve,
            "z_link": z_link,
        }
