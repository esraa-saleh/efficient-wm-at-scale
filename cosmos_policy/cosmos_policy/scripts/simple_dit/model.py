# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A small, self-contained AdaLN-conditioned DiT for LIBERO BC -- no VAE tokenizer, no text
conditioning, no rope3d/adaln_lora/sparse-attention machinery. Operates directly on raw RGB
camera frames and predicts a flow-matching velocity over an action chunk.

Architecture, following the original DiT paper's AdaLN-Zero recipe:
    - a small conv image encoder (shared weights across the two camera views)
    - a small proprio MLP
    - a task embedder projecting a pooled T5 instruction embedding (see dataset.py's
      `load_t5_embeddings`) into the conditioning space -- the model's only signal for which task
      it's doing, needed once training spans more than one task
    - conditioning vector c = agentview_emb + wrist_emb + proprio_emb + timestep_emb + task_emb
    - `chunk_size` action tokens (linear-embedded per-timestep actions + learned positional
      embedding), refined by `num_blocks` self-attention DiT blocks modulated by `c` via
      AdaLN-Zero (no cross-attention -- context enters purely through the conditioning vector)
    - a final AdaLN + linear layer projecting back to per-timestep action-dim velocity
"""

import math

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """t: (B,) float (e.g. in [0, 1]) -> (B, dim). `scale` widens the otherwise-tiny [0,1] range."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * scale * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class SpatialSoftmax(nn.Module):
    """Per-channel expected (x, y) keypoint location (Finn et al., "Deep Spatial Autoencoders for
    Visuomotor Learning") -- unlike global average pooling, this preserves *where* each channel's
    activation is concentrated, which manipulation policies need to reach/grasp precisely.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) -> (B, 2*C)
        b, c, h, w = x.shape
        attn = torch.softmax(x.view(b, c, -1), dim=-1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        exp_x = (attn * grid_x).sum(dim=(-2, -1))
        exp_y = (attn * grid_y).sum(dim=(-2, -1))
        return torch.cat([exp_x, exp_y], dim=-1)


class SmallImageEncoder(nn.Module):
    """4-layer stride-2 conv stem -> spatial softmax -> linear projection to embed_dim."""

    def __init__(self, embed_dim: int, in_channels: int = 3):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
            SpatialSoftmax(),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor, return_pre_proj: bool = False):
        # (B, 3, H, W) -> (B, embed_dim), optionally also the raw 256-d spatial-softmax feature
        # (pre-projection) -- used by SimpleDiTWorldModel as a fixed-size regression target, since
        # it's the same representation SmallImageEncoder itself would produce for a real frame.
        feat = self.net(x)
        proj = self.proj(feat)
        if return_pre_proj:
            return proj, feat
        return proj


class ProprioEncoder(nn.Module):
    def __init__(self, proprio_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionChunkEncoder(nn.Module):
    """Flattens a candidate (chunk_size, action_dim) action chunk and MLPs it to embed_dim -- the
    world model's conditioning input for "what action is being evaluated", mirroring
    ProprioEncoder's pattern exactly."""

    def __init__(self, chunk_size: int, action_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chunk_size * action_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, chunk_size, action_dim) -> (B, embed_dim)
        return self.net(x.flatten(start_dim=1))


class TaskEmbedder(nn.Module):
    """Projects a pooled T5 instruction embedding (see dataset.py's `load_t5_embeddings`) into the
    conditioning space -- the model's only signal for which task it's currently doing."""

    def __init__(self, task_emb_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_emb_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.embed_dim))


class DiTBlock(nn.Module):
    """Self-attention (over action tokens only) + MLP, each AdaLN-Zero modulated by `c`."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim))
        # Zero-init so each block starts as an identity function (standard AdaLN-Zero init).
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(embed_dim, action_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class SimpleDiTBC(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 9,
        task_emb_dim: int = 1024,
        chunk_size: int = 16,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.image_encoder = SmallImageEncoder(embed_dim)
        # Concatenates rather than sums the two camera views' embeddings before projecting back to
        # embed_dim -- summing lets one view's signal cancel or dilute the other's, whereas the
        # wrist camera's close-up view is often the one that actually disambiguates a precise
        # grasp point, so the model needs the freedom to weight it independently of agentview.
        self.vision_combiner = nn.Linear(2 * embed_dim, embed_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim, embed_dim)
        self.task_embedder = TaskEmbedder(task_emb_dim, embed_dim)
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.action_embedder = nn.Linear(action_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, chunk_size, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.final_layer = FinalLayer(embed_dim, action_dim)

    def forward(
        self,
        agentview_img: torch.Tensor,
        wrist_img: torch.Tensor,
        proprio: torch.Tensor,
        noisy_action_chunk: torch.Tensor,
        t: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Shared image encoder across both camera views (fewer params than separate encoders).
        agentview_emb = self.image_encoder(agentview_img)
        wrist_emb = self.image_encoder(wrist_img)
        vision_emb = self.vision_combiner(torch.cat([agentview_emb, wrist_emb], dim=-1))
        proprio_emb = self.proprio_encoder(proprio)
        t_emb = self.t_embedder(t)
        task_c = self.task_embedder(task_emb)
        c = vision_emb + proprio_emb + t_emb + task_c  # (B, embed_dim)

        x = self.action_embedder(noisy_action_chunk) + self.pos_embed  # (B, chunk_size, embed_dim)
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)  # (B, chunk_size, action_dim) predicted velocity


class SimpleDiTWorldModel(nn.Module):
    """Predicts a candidate action chunk's likely future state (as SmallImageEncoder's own 256-d
    spatial-softmax feature, not raw pixels) and value, distilled from the full Cosmos Policy
    model's imagined rollouts -- see simple_dit_bc/precompute_teacher_targets.py.

    Unlike SimpleDiTBC, this is direct MSE regression rather than flow-matching: there's no
    distributional-multimodality concern for a value scalar or a teacher-decoded image embedding
    the way there is for actions, so no timestep/noise input is needed. The candidate action chunk
    becomes conditioning input instead of the thing being predicted -- the model answers "given
    this state and this action, what happens?" rather than "what action should I take?".
    """

    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 9,
        task_emb_dim: int = 1024,
        chunk_size: int = 16,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        image_feat_dim: int = 256,
    ):
        super().__init__()
        self.image_encoder = SmallImageEncoder(embed_dim)
        self.vision_combiner = nn.Linear(2 * embed_dim, embed_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim, embed_dim)
        self.task_embedder = TaskEmbedder(task_emb_dim, embed_dim)
        self.action_encoder = ActionChunkEncoder(chunk_size, action_dim, embed_dim)
        # 3 fixed learned query tokens (future-agentview, future-wrist, value), refined by the same
        # AdaLN-Zero DiT block stack SimpleDiTBC uses over action tokens.
        self.query_tokens = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.blocks = nn.ModuleList([DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)])
        # Separate small heads per token rather than one shared head, since the three targets have
        # different output dims (256, 256, 1) -- FinalLayer works fine over a length-1 sequence.
        self.future_agentview_head = FinalLayer(embed_dim, image_feat_dim)
        self.future_wrist_head = FinalLayer(embed_dim, image_feat_dim)
        self.value_head = FinalLayer(embed_dim, 1)

    def forward(
        self,
        agentview_img: torch.Tensor,
        wrist_img: torch.Tensor,
        proprio: torch.Tensor,
        action_chunk: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        agentview_emb = self.image_encoder(agentview_img)
        wrist_emb = self.image_encoder(wrist_img)
        vision_emb = self.vision_combiner(torch.cat([agentview_emb, wrist_emb], dim=-1))
        proprio_emb = self.proprio_encoder(proprio)
        task_c = self.task_embedder(task_emb)
        action_c = self.action_encoder(action_chunk)
        c = vision_emb + proprio_emb + task_c + action_c  # (B, embed_dim)

        x = self.query_tokens.expand(c.shape[0], -1, -1)  # (B, 3, embed_dim)
        for block in self.blocks:
            x = block(x, c)

        future_agentview_feat = self.future_agentview_head(x[:, 0:1], c).squeeze(1)  # (B, image_feat_dim)
        future_wrist_feat = self.future_wrist_head(x[:, 1:2], c).squeeze(1)  # (B, image_feat_dim)
        value = self.value_head(x[:, 2:3], c).squeeze(1)  # (B, 1)
        return future_agentview_feat, future_wrist_feat, value
