import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

class ViTDecoderUsingTimmBlock(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        # drop_rate=0.0,
        # attn_drop_rate=0.0,
        # drop_path_rate=0.0,
    ):

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                # drop=drop_rate,
                # attn_drop=attn_drop_rate,
                # drop_path=drop_path_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        self.patch_out = nn.Linear(embed_dim, 3 * patch_size * patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.patch_out.weight, std=0.02)
        if self.patch_out.bias is not None:
            nn.init.zeros_(self.patch_out.bias)

    def forward(self, x):
        """
        x: shape [B, N, C], N = num_patches, C = embed_dim
        """
        B, N, C = x.shape
        assert N == self.num_patches, \
            f"tokens num does not match, got {N}, expect {self.num_patches}"

        # pos
        x = x + self.pos_embed  # => [B, N, embed_dim]


        for blk in self.blocks:
            x = blk(x)

        x = self.patch_out(x)   # => [B, N, 3 * p * p]

        # [B, N, 3*p*p] to [B, 3, H, W]
        #  H = W = patch_size * sqrt(N)
        H = W = int(math.sqrt(N))   
        p = self.patch_size         # patch_size

        #  reshape [B, N, 3, p, p]

        x = x.view(B, H, W, 3, p, p)               # => [B, 14, 14, 3, 16, 16]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()# => [B, 3, 14, 16, 14, 16]
        x = x.view(B, 3, H * p, W * p)              # => [B, 3, 224, 224]

        return x
