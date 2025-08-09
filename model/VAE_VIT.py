import torch
import torch.nn as nn
from timm import create_model


from .VIT_decoder_timm import ViTDecoderUsingTimmBlock


class PairwiseViTVAE(nn.Module):


    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        depth: int = 4,
        # num_heads: int = 6,  # this only changes the decoder, not the encoder
        num_heads: int = 12,  # base
        model_name: str = "vit_base_patch16_224",
    ):
        super().__init__()

        # ────────────────────────────────── Encoders ──────────────────────────────────
        self.curve_encoder = create_model(
            model_name, pretrained=True, num_classes=0, in_chans=in_chans
        )
        self.link_encoder = create_model(
            model_name, pretrained=True, num_classes=0, in_chans=in_chans
        )

        self.embed_dim = self.curve_encoder.embed_dim  # ViT-base = 768, ViT‑small = 384, ViT‑tiny = 192

        # ───────────────────────────── Latent parameter MLPs ──────────────────────────

        hidden_dim = self.embed_dim * 2
        drop = 0.1   

        def create_mlp():
            return nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, self.embed_dim)
            )
        
        self.curve_mu_mlp = create_mlp()
        self.curve_logvar_mlp = create_mlp()
        self.link_mu_mlp = create_mlp()
        self.link_logvar_mlp = create_mlp()

        # self.curve_mu = nn.Linear(self.embed_dim, self.embed_dim)
        # self.curve_logvar = nn.Linear(self.embed_dim, self.embed_dim)
        # self.link_mu = nn.Linear(self.embed_dim, self.embed_dim)
        # self.link_logvar = nn.Linear(self.embed_dim, self.embed_dim)

        # ────────────────────────────────── Decoders ──────────────────────────────────
        self.curve_decoder = ViTDecoderUsingTimmBlock(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.link_decoder = ViTDecoderUsingTimmBlock(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

    # ─────────────────────────────────── helpers ─────────────────────────────────────

    def reparameterize(self,mu, logvar):
        """Reparameterisation trick:  z = μ + σ ⊙ ε,  ε ∼ 𝒩(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encode(
        self,
        encoder,
        x,
        mu_mlp,
        logvar_mlp,
    ):
        """Encode an image, return (z, μ, log σ²)."""
        # ViT forward_features returns [B, 197, D] with [CLS] + 196 patches
        tokens = encoder.forward_features(x)          # [B,197,192]
        patch_tokens = tokens[:, 1:, :]                    # [B,196,192]
        mu = mu_mlp(patch_tokens)
        logvar = logvar_mlp(patch_tokens)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    # ─────────────────────────────────── forward ─────────────────────────────────────
    def forward(
        self,
        curve_img: torch.Tensor,
        link_img: torch.Tensor,
    ):
        """Forward pass returning everything you need for loss computation."""
        z_curve, mu_curve, log_curve = self.encode(
            self.curve_encoder, curve_img, self.curve_mu_mlp, self.curve_logvar_mlp
        )
        z_link, mu_link, log_link = self.encode(
            self.link_encoder, link_img, self.link_mu_mlp, self.link_logvar_mlp
        )

        rec_curve = self.curve_decoder(z_curve)
        rec_link = self.link_decoder(z_link)

        # return rec_curve,z_curve,mu_curve,log_curve,rec_link,z_link,mu_link,log_link
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
