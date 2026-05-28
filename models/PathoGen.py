import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention

from models.VL import PromptLearner


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PathoGen(nn.Module):
    def __init__(self, n_classes=2):
        super(PathoGen, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )


        self.classifier = nn.Sequential(
            nn.Linear(3 * n_classes, n_classes),
        )

        self.apply(initialize_weights)

        self.prompt_learner = PromptLearner('cuda',context_leangth=384)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cross_attention_1 = MultiheadAttention(embed_dim=512, num_heads=1)
        # self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

        self.encoder = nn.Sequential(*[
            LPABlock(512) for _ in range(4)
        ])
        self.proto_proj = nn.Linear(512, 512)
        proto = torch.randn(n_classes, 512) / 512 ** 0.5
        self.register_buffer("vis_prototypes", proto)
        self.register_buffer("text_prototypes", proto)



    def reset(self):
        initialize_weights(self)

    def forward(self, x, text=None):
        x = self.feature(x.squeeze(0))

        with torch.no_grad():
            text_prompt = self.prompt_learner(text).float()  # [B, 512]


        x = self.encoder(x)  # [n, d]
        proto = self.proto_proj(self.vis_prototypes)  # [C, d]
        attn_logits = torch.matmul(x, proto.T) / x.shape[-1] ** 0.5  # [n, C]
        attn_weights = F.softmax(attn_logits, dim=0)  # [n, C]


        slide_features = torch.einsum('nc,nd->cd', attn_weights, x)  # [C, d]

        text_q = torch.cat([slide_features, self.text_prototypes], dim=0)

        text_components, _ = self.cross_attention_1(text_q, text_prompt, text_prompt)

        text_features = torch.cat([self.text_prototypes, text_components], dim=0)

        text_features = self.norm2(text_features)

        logit_scale = self.logit_scale.exp()
        x = logit_scale * slide_features @ text_features.t()

        x = x.mean(dim=0, keepdim=True)


        egfr_logits = self.classifier(x)



        return egfr_logits


class LPABlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * expand_factor)
        self.act = nn.GELU()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.state_proj = nn.Linear(d_model * expand_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [n, d]
        residual = x
        x = self.act(self.in_proj(x))  # [n, d*]
        x = x.transpose(0, 1).unsqueeze(0)  # → [1, d*, n]
        x = self.pool(x).squeeze(0).transpose(0, 1)  # → [n, d*]
        x = self.dropout(self.state_proj(x))  # [n, d]
        return self.norm(x + residual)




