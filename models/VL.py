# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin

import models.clip.clip as clip


logger = logging.getLogger(__name__)
import torch
import torch.nn as nn


from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()





class PromptLearner(nn.Module):
    def __init__(self,device,context_leangth):
        super().__init__()
        clip_model, _ = clip.load("RN50", device=device)
        self.clip_model = clip_model
        self.n_ctx = 16
        self.ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if self.ctx_init:
            self.ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(self.ctx_init.split(" "))
            prompt = clip.tokenize(self.ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            self.ctx_vectors = embedding[0, 1: 1 + self.n_ctx, :]
            self.prompt_prefix = self.ctx_init
        else:
            self.ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(self.ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        self.ctx = nn.Parameter(self.ctx_vectors).to(device)
        self.class_token_position = "end"
        self.device = device


        self.context_leangth = context_leangth

    def forward(self, classnames):
        # Ensure classnames is a list of strings
        n_cls = len(classnames)
        dtype = self.clip_model.dtype

        # Tokenizing classnames and processing them
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p,self.context_leangth) for p in prompts]).to(self.device)

        with torch.no_grad():
            # print(tokenized_prompts.device)
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Registering token prefix and suffix buffers
        self.register_buffer("token_prefix", embedding[:, :1, :].squeeze(0))
        self.register_buffer("token_suffix", embedding[:, 1:-self.n_ctx, :].squeeze(0))

        # Expanding the context vector if needed
        ctx = self.ctx


        prefix = self.token_prefix
        suffix = self.token_suffix

        # Constructing prompts based on class_token_position
        prompts = []
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=0,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            for i in range(n_cls):
                name_len = name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            for i in range(n_cls):
                name_len = name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError("Invalid class_token_position value")

        return prompts

