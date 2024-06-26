# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict


class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
    
        self.code_dim = code_dim

        self.encoder = Encoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        print("\n\nEntered Forward of mgpt_vq.py")
        print("Features: ",features.size())
        # Preprocess
        x_in = self.preprocess(features)
        print("\nFeatures preprocessed: ",x_in.size())
        # Encode
        x_encoder = self.encoder(x_in)
        print("\nEncoded: ",x_encoder.size())
        # quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        print("\nQuantized: ",x_quantized.size())
        # decoder
        x_decoder = self.decoder(x_quantized)
        print("\nDecoded: ",x_decoder.size())
        x_out = self.postprocess(x_decoder)
        print("Output postprocessed: ",x_out.size())
        return x_out, loss, perplexity
    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:
        print("\n\n\n\nEntered Encodeeerrrr")
        print("features: ",features.size())
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        print("x_in (after preprocess): ",x_in.size())
        x_encoder = self.encoder(x_in)
        print("X encoder1 (after encoder): ",x_encoder.size())
        x_encoder = self.postprocess(x_encoder)
        print("x_encoder2 (after post process): ",x_encoder.size())
        x_encoder = x_encoder.contiguous().view(-1,
                                                x_encoder.shape[-1])  # (NT, C)
        print("x_encoder (after 7arakat): ",x_encoder.size())
        code_idx = self.quantizer.quantize(x_encoder)
        print("code idx (after quantize): ",code_idx.size())
        code_idx = code_idx.view(N, -1)
        print("code idx (after 7arakat): ",code_idx.size())
        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):
        print("\n\n\n\n\nEntered Decode Function in mgpt_vq.py")
        print("\nZ: ",z.size())
        x_d = self.quantizer.dequantize(z)
        print("\nDequantized: ",x_d.size())
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        print("\nSome stuff: ",x_d.size())
        # decoder
        x_decoder = self.decoder(x_d)
        print("\nDecoded: ",x_decoder.size())
        x_out = self.postprocess(x_decoder)
        print("\nOutput after postprocess: ",x_out.size())
        print("/////////////////////////////////")

        return x_out


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        print("input_emb_width: ", input_emb_width)
        print("output_emb_width: ", output_emb_width)
        print("down_t: ", down_t)
        print("stride_t: ", stride_t)
        print("width: ", width)
        print("depth: ", depth)
        print("dilation_growth_rate: ", dilation_growth_rate)
        print("activation: ", activation)
        print("norm: ", norm)
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        print("input_emb_width: ", input_emb_width)
        print("output_emb_width: ", output_emb_width)
        print("down_t: ", down_t)
        print("stride_t: ", stride_t)
        print("width: ", width)
        print("depth: ", depth)
        print("dilation_growth_rate: ", dilation_growth_rate)
        print("activation: ", activation)
        print("norm: ", norm)

    def forward(self, x):
        return self.model(x)
