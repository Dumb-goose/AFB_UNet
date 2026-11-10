import torch
from torch import nn
import torch.nn.functional as F
from torch.fft import rfft2


class LA_FSA(nn.Module):
    """
    LA-FSA:
    Local Adaptive Frequency Selection Attention

    流程：
        Input
          ↓
        Multi-scale Average Pooling
          ↓
        Adjacent-scale Differential Frequency Decomposition
          ↓
        High / Mid / Low / Base
          ↓
        Channel-wise Band Energy
          ↓
        Grouped Dynamic Frequency Router
          ↓
        Dynamic Frequency Weights
          ↓
        Adaptive Frequency Reconstruction
          ↓
        Residual Update
    """

    def __init__(
        self,
        channel: int,
        init_res_scale: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()

        if channel <= 0:
            raise ValueError("channel must be positive.")

        self.channel = channel
        self.eps = eps

        # -----------------------------------------------------
        # Dynamic Frequency Router
        #
        # 每个通道拥有 4 个频率能量：
        # [High, Mid, Low, Base]
        #
        # groups=channel 表示：
        # 每个通道只根据自己的 4 个频率能量
        # 预测自己的 4 个动态频率权重。
        # -----------------------------------------------------
        self.freq_router = nn.Conv2d(
            in_channels=channel * 4,
            out_channels=channel * 4,
            kernel_size=1,
            groups=channel,
            bias=True,
        )

        # 控制 LA-FSA 对原始特征的修改强度
        self.res_scale = nn.Parameter(
            torch.tensor(
                init_res_scale,
                dtype=torch.float32
            )
        )

    @staticmethod
    def _avg_pool(
        x: torch.Tensor,
        kernel_size: int,
    ) -> torch.Tensor:
        """
        使用不同尺度的平均池化获得不同平滑程度的特征。
        """

        return F.avg_pool2d(
            x,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def _frequency_decomposition(
        self,
        x: torch.Tensor,
    ):
        """
        相邻尺度差分频率分解。

        设：
            L3 = Pool3(X)
            L5 = Pool5(X)
            L7 = Pool7(X)

        则：
            High = X  - L3
            Mid  = L3 - L5
            Low  = L5 - L7
            Base = L7

        并且严格满足：
            High + Mid + Low + Base = X
        """

        # 多尺度局部平滑
        low_3 = self._avg_pool(
            x,
            kernel_size=3
        )

        low_5 = self._avg_pool(
            x,
            kernel_size=5
        )

        low_7 = self._avg_pool(
            x,
            kernel_size=7
        )

        # 相邻尺度差分
        high = x - low_3

        mid = low_3 -  low_5

        low =  low_5 - low_7

        base = low_7

        return high, mid, low, base

    @staticmethod
    def _band_energy(
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个通道的频带能量。

        Input:
            B × C × H × W

        Output:
            B × C × 1 × 1
        """

        return F.adaptive_avg_pool2d(
            feature.abs(),
            output_size=1
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, c, _, _ = x.shape

        # =====================================================
        # 1. 相邻尺度差分频率分解
        # =====================================================

        (
            high,
            mid,
            low,
            base,
        ) = self._frequency_decomposition(x)

        # =====================================================
        # 2. 计算各频带的通道能量
        # =====================================================

        high_energy = self._band_energy(high)
        mid_energy = self._band_energy(mid)
        low_energy = self._band_energy(low)
        base_energy = self._band_energy(base)

        # -----------------------------------------------------
        # 按照每个通道组织：
        #
        # Channel 1:
        # [High, Mid, Low, Base]
        #
        # Channel 2:
        # [High, Mid, Low, Base]
        #
        # ...
        #
        # Shape:
        # B × C × 4 × 1 × 1
        # -----------------------------------------------------
        energies = torch.stack(
            [
                high_energy,
                mid_energy,
                low_energy,
                base_energy,
            ],
            dim=2,
        )

        # B × C × 4 × 1 × 1
        #        ↓
        # B × 4C × 1 × 1
        #
        # 保证 grouped 1×1 Conv 中，
        # 每一个 group 正好读取同一个通道的 4 个频带能量。
        energies = energies.reshape(
            b,
            c * 4,
            1,
            1,
        )

        # =====================================================
        # 3. Dynamic Frequency Router
        # =====================================================

        frequency_logits = self.freq_router(
            energies
        )

        # B × 4C × 1 × 1
        #        ↓
        # B × C × 4 × 1 × 1
        frequency_logits = frequency_logits.view(
            b,
            c,
            4,
            1,
            1,
        )

        # -----------------------------------------------------
        # 对 High / Mid / Low / Base 做 Softmax
        #
        # 对每一个样本、每一个通道：
        #
        # Wh + Wm + Wl + Wb = 1
        # -----------------------------------------------------
        frequency_weights = torch.softmax(
            frequency_logits,
            dim=2,
        )

        high_weight = frequency_weights[:, :, 0]
        mid_weight = frequency_weights[:, :, 1]
        low_weight = frequency_weights[:, :, 2]
        base_weight = frequency_weights[:, :, 3]

        # =====================================================
        # 4. Adaptive Frequency Reconstruction
        # =====================================================

        adaptive_frequency = (
            high_weight * high
            + mid_weight * mid
            + low_weight * low
            + base_weight * base
        )

        # =====================================================
        # 5. Residual Frequency Update
        #
        # 均匀频率权重：
        # Wh = Wm = Wl = Wb = 0.25
        #
        # 作为中性频率状态。
        #
        # 动态频率重构与中性状态之间的差异，
        # 才作为真正的频率注意力修正量。
        # =====================================================

        uniform_frequency = 0.25 * (
            high
            + mid
            + low
            + base
        )

        # 因为：
        # High + Mid + Low + Base = X
        #
        # 所以：
        # uniform_frequency = X / 4

        frequency_correction = (
            adaptive_frequency
            - uniform_frequency
        )

        # 有界残差强度
        gamma = torch.tanh(
            self.res_scale
        ).to(dtype=x.dtype)

        output = (
            x
            + gamma * frequency_correction
        )

        return output

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        hidden = max(2, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = x.mean(1, keepdim=True)
        max_out = x.max(1, keepdim=True)[0]
        return x * self.conv(torch.cat([avg_out, max_out], dim=1))


class MixedAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channel, reduction)
        self.spatial_att = SpatialAttention()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ch_att = self.channel_att(x)
        sp_att = self.spatial_att(x)
        gate = self.gate(x)
        return gate * ch_att + (1 - gate) * sp_att


class AttentionFactory:
    _ATTENTION_MAP = {
        'none': lambda channel: nn.Identity(),
        'spatial': lambda channel: SpatialAttention(),
        'channel': lambda channel: ChannelAttention(channel),
        'mixed': lambda channel: MixedAttention(channel),
    }

    @classmethod
    def create_attention(cls, attention_type, channel):
        """Create a spatial/channel/mixed attention module."""
        return cls._ATTENTION_MAP.get(
            attention_type, lambda channel: nn.Identity()
        )(channel)

    @classmethod
    def create_frequency_attention(cls, is_fft, channel):
        """Create LA-FSA when frequency-spatial attention is enabled."""
        return LA_FSA(channel) if is_fft else nn.Identity()