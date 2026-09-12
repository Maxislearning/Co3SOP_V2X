# ---------------------------------------------
# Stage 3: OccupancyBeamEncoder -- plain nn.Module (not HEADS-registered;
# nothing in mmdet's registry ever builds this, it's driven by
# tools/train_occ_beam.py directly). Small Cartesian 3D CNN baseline per the
# approved Stage 3A plan: no attention, no Transformer, no Polar transform.
#
# TENSOR LAYOUT (locked, asserted at the call site in OccBeamDataset, not
# re-derived here): input is [B,C_in,16,200,200] -- PyTorch Conv3d's
# [B,C,D,H,W] with D=Z(16), H=Y(200), W=X(200), matching the permute
# OccBeamDataset.__getitem__ already applies to the on-disk [C,X,Y,Z]
# volumes. This module itself is layout-agnostic (every op below is
# elementwise/conv over whatever 3 spatial dims it's given), but documented
# here anyway since a future reader must not assume the disk order.
# ---------------------------------------------
import torch.nn as nn


def _num_groups(out_ch, max_groups=8):
    """Largest divisor of out_ch that's <= max_groups -- GroupNorm requires
    an exact divisor (108, one of this module's channel counts, isn't
    divisible by 8)."""
    for g in range(min(max_groups, out_ch), 0, -1):
        if out_ch % g == 0:
            return g
    return 1


def _conv_gn_relu(in_ch, out_ch, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.GroupNorm(_num_groups(out_ch), out_ch),
        nn.ReLU(inplace=True),
    )


class OccupancyBeamEncoder(nn.Module):
    def __init__(self, in_channels, num_tx_beams=64, num_rx_beams=64, embed_dim=256):
        super().__init__()
        self.stem = _conv_gn_relu(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            _conv_gn_relu(32, 48, kernel_size=3, stride=2, padding=1),    # D:16->8  H/W:200->100
            _conv_gn_relu(48, 72, kernel_size=3, stride=2, padding=1),    # D:8->4   H/W:100->50
            _conv_gn_relu(72, 108, kernel_size=3, stride=2, padding=1),   # D:4->2   H/W:50->25
            _conv_gn_relu(108, 160, kernel_size=3, stride=2, padding=1),  # D:2->1   H/W:25->13
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.embed = nn.Sequential(nn.Linear(160, embed_dim), nn.ReLU(inplace=True))
        self.tx_head = nn.Linear(embed_dim, num_tx_beams)
        self.rx_head = nn.Linear(embed_dim, num_rx_beams)

    def forward(self, x):
        # x: [B, C_in, 16, 200, 200]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)  # [B,160]
        emb = self.embed(x)          # [B,embed_dim]
        return self.tx_head(emb), self.rx_head(emb)
