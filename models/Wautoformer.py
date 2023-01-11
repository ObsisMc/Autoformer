import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


# author: Obsismc
class DownSampleLayer(nn.Module):
    def __init__(self, down_sample_scale, d_model):
        super(DownSampleLayer, self).__init__()
        self.localConv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, stride=1)
        self.down_sample_norm = nn.BatchNorm1d(d_model)
        self.down_sample_activation = nn.ELU()
        self.localMax = nn.MaxPool1d(kernel_size=down_sample_scale)

    def forward(self, x: torch.Tensor):
        """

        :param x: (B,L,D)
        :return: (B,L/self.down_sample_scale,D)
        """
        x = self.localConv(x.permute(0, 2, 1))
        x = self.down_sample_norm(x)
        x = self.down_sample_activation(x)
        x = self.localMax(x)
        return x.permute(0, 2, 1)


# author: Obsismc
class UpSampleLayer(nn.Module):
    def __init__(self, down_sample_scale, d_model, padding=1, output_padding=1):
        super(UpSampleLayer, self).__init__()
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        # self.upSampleNorm = nn.LayerNorm(d_model)

        kern_size = down_sample_scale + 2 * padding - output_padding  # formula of ConvTranspose1d
        self.upSample = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, padding=padding,
                                           kernel_size=kern_size, stride=down_sample_scale,
                                           output_padding=output_padding)  # need to restore the length
        # self.upActivation = nn.ELU()

    def forward(self, x):
        """

        :param x: (B,L,D)
        :return: (B,self.down_sample_scale * L,D)
        """
        x = self.proj(x.permute(0, 2, 1))
        # x = self.upSampleNorm(x.transpose(2, 1))
        x = self.upSample(x)
        # x = self.upActivation(x)
        return x.transpose(2, 1)


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # U-net part
        # Important: can only handle even
        # down sample step
        self.depth_max = 2  # depth
        self.scale = 2
        self.downSamples = nn.ModuleList(
            [DownSampleLayer(down_sample_scale=self.scale, d_model=configs.d_model) for _ in
             range(self.depth_max)]
        )
        self.downSamples_trend = nn.ModuleList(
            [DownSampleLayer(down_sample_scale=self.scale, d_model=configs.c_out) for _ in
             range(self.depth_max)]
        )
        self.downSamples.append(nn.Identity())
        self.downSamples_trend.append(nn.Identity())
        # up sample step: refer to Yformer's method
        self.upSamples = nn.ModuleList(
            [UpSampleLayer(down_sample_scale=self.scale, d_model=configs.c_out) for _ in
             range(self.depth_max)])
        self.upSamples.insert(0, nn.Identity())

        # final projection
        self.sampled_projection = nn.Linear(configs.c_out, configs.c_out, bias=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)  # Obsismc: decoder input seasonal series

        attns_all = []
        dec_out_all = []
        enc_out_last, dec_out_last = enc_out, dec_out
        trend_init_last = trend_init
        for depth in range(self.depth_max + 1):
            enc_out_cross, attns = self.encoder(enc_out_last,
                                                attn_mask=enc_self_mask)  # Obsismc: first autto-correlation then decomp, return seasonal series
            seasonal_part, trend_part = self.decoder(dec_out_last, enc_out_cross, x_mask=dec_self_mask,
                                                     cross_mask=dec_enc_mask,
                                                     trend=trend_init_last)
            # final
            dec_out = trend_part + seasonal_part
            dec_out_all.append(dec_out)
            attns_all.append(attns)

            # update
            enc_out_last = self.downSamples[depth](enc_out_last)
            dec_out_last = self.downSamples[depth](dec_out_last)
            trend_init_last = self.downSamples_trend[depth](trend_init_last)

        # up sampling
        for depth in range(self.depth_max, 0, -1):
            dec_out_up = self.upSamples[depth](dec_out_all[depth])
            dec_out_all[depth - 1] += dec_out_up

        dec_out = dec_out_all[0]
        # dec_out = self.sampled_projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns_all
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
