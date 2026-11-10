import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionFactory, ChannelAttention


class AnisotropicFlowBalanceBlock(nn.Module):

    def __init__(
        self,
        channel: int,
        flow_radius: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.flow_radius = flow_radius
        self.eps = eps

        # 每个通道控制细节补偿强度
        self.detail_gain = nn.Parameter(
            torch.full((1, channel, 1, 1), 0.1)
        )

        # 最终残差强度
        self.res_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def _get_neighbors(self, x):
        r = self.flow_radius
        h, w = x.shape[-2:]

        x_pad = F.pad(
            x,
            (r, r, r, r),
            mode="replicate"
        )

        # 上下左右
        left = x_pad[:, :, r:r+h, 0:w]
        right = x_pad[:, :, r:r+h, 2*r:2*r+w]
        up = x_pad[:, :, 0:h, r:r+w]
        down = x_pad[:, :, 2*r:2*r+h, r:r+w]

        # 四个对角邻居
        up_left = x_pad[:, :, 0:h, 0:w]
        up_right = x_pad[:, :, 0:h, 2*r:2*r+w]
        down_left = x_pad[:, :, 2*r:2*r+h, 0:w]
        down_right = x_pad[:, :, 2*r:2*r+h, 2*r:2*r+w]

        return (
            left,
            right,
            up,
            down,
            up_left,
            up_right,
            down_left,
            down_right,
        )

    def forward(self, x):

        (
            left,
            right,
            up,
            down,
            up_left,
            up_right,
            down_left,
            down_right,
        ) = self._get_neighbors(x)

        # ---------------------------------
        # 1. 水平 / 垂直修正流
        # ---------------------------------

        horizontal_flow = (
            0.5 * (left + right) - x
        )

        vertical_flow = (
            0.5 * (up + down) - x
        )

        # ---------------------------------
        # 2. 方向变化强度
        # ---------------------------------

        horizontal_cue = (
            right - left
        ).abs()

        vertical_cue = (
            down - up
        ).abs()

        cue_sum = (
            horizontal_cue
            + vertical_cue
            + self.eps
        )

        # 变化越强的方向，传导权重越低
        horizontal_weight = (
            vertical_cue / cue_sum
        )

        vertical_weight = (
            horizontal_cue / cue_sum
        )

        # ---------------------------------
        # 3. 各向异性方向流
        # ---------------------------------

        directional_flow = (
            horizontal_weight * horizontal_flow
            + vertical_weight * vertical_flow
        )

        # ---------------------------------
        # 4. 独立对角细节补偿
        # ---------------------------------

        diagonal_mean = 0.25 * (
            up_left
            + up_right
            + down_left
            + down_right
        )

        detail_flow = (
            x - diagonal_mean
        )

        detail_gain = torch.tanh(
            self.detail_gain
        )

        # ---------------------------------
        # 5. 最终输出
        # ---------------------------------

        balanced_flow = (
            directional_flow
            + detail_gain * detail_flow
        )

        res_scale = torch.tanh(
            self.res_scale
        )

        return (
            x
            + res_scale * balanced_flow
        )


AFB = AnisotropicFlowBalanceBlock


class AquaEncoderBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        flow_radius=1,
        is_fft=True,
        attention_type="none",
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c),
            nn.Conv2d(in_c, out_c, 1),
        )

        self.frequency = AttentionFactory.create_frequency_attention(is_fft, out_c)
        self.att_layer = AttentionFactory.create_attention(attention_type, out_c)
        self.afb = AFB(out_c, flow_radius=flow_radius)

        self.norm = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d(2)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, in_c, 1, groups=in_c),
                nn.Conv2d(in_c, out_c, 1),
            )
            if in_c != out_c
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.frequency(x)
        x = self.att_layer(x)
        x = self.afb(x)
        x = self.norm(x)
        x = self.conv2(x)
        skip = x + identity
        return self.down(skip), skip


class AquaDecoderBlock(nn.Module):
    def __init__(
        self,
        in_c,
        skip_c,
        out_c,
        flow_radius=1,
        is_fft=True,
        attention_type="none",
    ):
        super().__init__()

        # Lightweight learnable upsampling requested by the user.
        mid_c = out_c
        self.up = nn.Sequential(
            nn.Conv2d(in_c, mid_c * 4, 1),
            nn.PixelShuffle(2),
        )

        self.merge = nn.Sequential(
            nn.Conv2d(out_c + skip_c, out_c, 1),
            nn.ReLU(inplace=True),
        )

        self.frequency = AttentionFactory.create_frequency_attention(is_fft, out_c)
        self.att_layer = AttentionFactory.create_attention(attention_type, out_c)
        self.afb = AFB(out_c, flow_radius=flow_radius)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = self.merge(torch.cat([x, skip], dim=1))
        x = self.frequency(x)
        x = self.att_layer(x)
        x = self.afb(x)
        x = self.norm(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, channel, flow_radius=3):
        super().__init__()
        self.ca = ChannelAttention(channel)
        self.afb = AFB(channel, flow_radius=flow_radius)
        self.fusion = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        channel_out = self.ca(x)
        flow_out = self.afb(x)
        return self.fusion(torch.cat([channel_out, flow_out], dim=1)) + x


class Model(nn.Module):
    def __init__(self, n=2, in_channels=3, out_channels=3):
        super().__init__()
        base_channel = 4 + (n - 1) * 6

        self.input_layer = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    3,
                    padding=1,
                    groups=in_channels,
                ),
                nn.Conv2d(in_channels, base_channel, 1),
            ),
            nn.BatchNorm2d(base_channel),
        )

        self.encoder1 = AquaEncoderBlock(
            base_channel,
            base_channel * 2,
            flow_radius=1,
            attention_type="spatial",
            is_fft=True,
        )
        self.encoder2 = AquaEncoderBlock(
            base_channel * 2,
            base_channel * 4,
            flow_radius=2,
            attention_type="mixed",
        )

        self.bottleneck = Bottleneck(
            base_channel * 4,
            flow_radius=3,
        )

        self.decoder1 = AquaDecoderBlock(
            base_channel * 4,
            base_channel * 4,
            base_channel * 2,
            flow_radius=2,
            attention_type="mixed",
        )
        self.decoder2 = AquaDecoderBlock(
            base_channel * 2,
            base_channel * 2,
            base_channel,
            flow_radius=1,
            attention_type="spatial",
            is_fft=True,
        )

        self.output_layer = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(base_channel, base_channel, 1, groups=base_channel),
                nn.Conv2d(base_channel, out_channels, 1),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.input_layer(x)
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x = self.bottleneck(x2)
        x = self.decoder1(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.output_layer(x)

        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return x