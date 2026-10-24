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
import torch
import torch.nn.functional as F
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


class _Backbone(nn.Module):
    """Shared Cartesian 3D CNN trunk -- identical for both OccupancyBeamEncoder
    (flat 64-way head) and HierarchicalOccupancyBeamEncoder (4-way panel +
    16-way in-panel heads, 任务22 A2), so the two only differ in their heads,
    never in the features they're built on.

    coord_channels (任务35): append 3 normalized (z,y,x) coordinate channels
    in [-1,1] to the input (CoordConv). Without them the trunk is
    conv + global-average-pool, i.e. nearly translation-invariant, so a
    sparse binary GT target mask (a ~400-voxel box whose only informative
    property is WHERE it sits relative to the RX ego at the grid centre)
    loses its position -- the diagnosed cause of gt/target underfitting.
    'dense' (or True): coordinates everywhere. They swamp the sparse input
    at init, so training sits on the majority-class plateau for 3-8+ epochs.
    'gated': coordinates multiplied by the voxel's occupancy (max over input
    channels) -- input stays sparse and the pooled features carry the
    occupied voxels' centroid directly."""
    def __init__(self, in_channels, embed_dim, coord_channels=False):
        super().__init__()
        if coord_channels is True:
            coord_channels = 'dense'
        assert coord_channels in (False, None, 'dense', 'gated'), coord_channels
        self.coord_channels = coord_channels or False
        if self.coord_channels:
            in_channels += 3
        self.stem = _conv_gn_relu(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            _conv_gn_relu(32, 48, kernel_size=3, stride=2, padding=1),    # D:16->8  H/W:200->100
            _conv_gn_relu(48, 72, kernel_size=3, stride=2, padding=1),    # D:8->4   H/W:100->50
            _conv_gn_relu(72, 108, kernel_size=3, stride=2, padding=1),   # D:4->2   H/W:50->25
            _conv_gn_relu(108, 160, kernel_size=3, stride=2, padding=1),  # D:2->1   H/W:25->13
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.embed = nn.Sequential(nn.Linear(160, embed_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        # x: [B, C_in, 16, 200, 200]
        if self.coord_channels:
            B, _, D, H, W = x.shape
            axes = [torch.linspace(-1.0, 1.0, n, device=x.device, dtype=x.dtype) for n in (D, H, W)]
            grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=0)  # [3,D,H,W]
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
            if self.coord_channels == 'gated':
                grid = grid * x.amax(dim=1, keepdim=True)
            x = torch.cat([x, grid], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)  # [B,160]
        return self.embed(x)         # [B,embed_dim]


class OccupancyBeamEncoder(nn.Module):
    def __init__(self, in_channels, num_tx_beams=64, num_rx_beams=64, embed_dim=256, coord_channels=False):
        super().__init__()
        self.backbone = _Backbone(in_channels, embed_dim, coord_channels)
        self.tx_head = nn.Linear(embed_dim, num_tx_beams)
        self.rx_head = nn.Linear(embed_dim, num_rx_beams)

    def forward(self, x):
        emb = self.backbone(x)
        return self.tx_head(emb), self.rx_head(emb)


class HierarchicalOccupancyBeamEncoder(nn.Module):
    """任务22 A2: main candidate fix for the diagnosed panel-level class-
    imbalance collapse -- separates panel (4-way) selection from in-panel
    (16-way) fine steering into two heads per side, so the panel loss can be
    reweighted independently of in-panel loss (train_occ_beam.py weights it
    by panel frequency; in-panel loss stays unweighted since the diagnosis
    -- 任务22's full_eval_occ_beam.py run -- showed in-panel discrimination
    already generalizes well with a flat head, nothing there to fix).

    The in-panel head is ONE shared 16-way classifier per side, not 4
    separate per-panel heads -- reused across all 4 panels regardless of
    which one wins, deliberately mirroring panel_beamforming.py's own
    locked design (ONE shared local DFT codebook reused by all 4 panels,
    never rotated/panel-specific -- see that module's docstring). Combined
    64-way logits (for compatibility with existing eval code expecting one
    flat distribution) are produced by log-probability addition:
    log P(panel, local) = log P(panel) + log P(local) -- treating panel and
    in-panel choice as independent given the shared embedding, which is the
    natural reading of "one shared in-panel classifier with no panel input".
    combined_logits[:, p*16+k] = log_softmax(panel_logits)[:, p] +
    log_softmax(inpanel_logits)[:, k] -- NOT a true joint likelihood in
    general, but preserves the correct argmax/topk ranking for the
    combined-metric eval code, which is all it's used for."""
    def __init__(self, in_channels, num_panels=4, beams_per_panel=16, embed_dim=256, coord_channels=False):
        super().__init__()
        self.num_panels = num_panels
        self.beams_per_panel = beams_per_panel
        self.backbone = _Backbone(in_channels, embed_dim, coord_channels)
        self.tx_panel_head = nn.Linear(embed_dim, num_panels)
        self.tx_inpanel_head = nn.Linear(embed_dim, beams_per_panel)
        self.rx_panel_head = nn.Linear(embed_dim, num_panels)
        self.rx_inpanel_head = nn.Linear(embed_dim, beams_per_panel)

    def forward(self, x):
        emb = self.backbone(x)
        return (self.tx_panel_head(emb), self.tx_inpanel_head(emb),
                self.rx_panel_head(emb), self.rx_inpanel_head(emb))

    @staticmethod
    def combine_to_flat_logits(panel_logits, inpanel_logits, num_panels, beams_per_panel):
        """[B,num_panels],[B,beams_per_panel] -> [B,num_panels*beams_per_panel],
        via log-probability addition (see class docstring). Used for
        combined-metric eval (Top1/3/5, majority baselines) so downstream
        code doesn't need to know this model is hierarchical."""
        log_p_panel = F.log_softmax(panel_logits, dim=1)      # [B,P]
        log_p_local = F.log_softmax(inpanel_logits, dim=1)    # [B,K]
        combined = log_p_panel.unsqueeze(2) + log_p_local.unsqueeze(1)  # [B,P,K]
        return combined.reshape(combined.shape[0], num_panels * beams_per_panel)
