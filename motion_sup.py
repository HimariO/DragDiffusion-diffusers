from itertools import product

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from termcolor import colored


def unet_feat_hook(pipeline, module, xin, xout):
    pipeline.unet_feat_cache = xout


class MotionSup:
    """
    Motion supervision for 1 image
    """
    
    def __init__(
            self, 
            src_points: Tensor,
            dst_points: Tensor,
            refernce_feat: Tensor,
            refernce_latent: Tensor,
            mask: Tensor,
            steps=150,
            debug_log=False,
        ):
        self.init_latent = refernce_latent.float().clone().detach()
        self.ref_latent = torch.nn.Parameter(refernce_latent.float())
        self.ref_feat = refernce_feat.clone().detach()
        self.mask = mask.float()
        self.adam = torch.optim.Adam([self.ref_latent], lr=1e-2)
        
        self.steps = steps
        self._debug_log = debug_log
        
        if not ((src_points <= 1).all() and (src_points >= -1).all()):
            raise ValueError("Value of point coordinate should be within range of [-1, 1]")
        if not ((dst_points <= 1).all() and (dst_points >= -1).all()):
            raise ValueError("Value of point coordinate should be within range of [-1, 1]")
        self.src_points = src_points
        self.dst_points = dst_points
        self.handle_points = src_points.clone()
        
        c, h, w = refernce_feat.shape
        self.radius = max(1, min(h, w) // 8)
        self.sample_offsets = self._radius_offset(1)
        self.search_offsets = self._radius_offset(self.radius, include_origin=True)
    
    def _radius_offset(self, radius, include_origin=True):
        c, h, w = self.ref_feat.shape
        offsets = [
            [x / w, y / h]
            for y, x in product(
                range(-radius, radius + 1),
                range(-radius, radius + 1))
            if (y**2 + x**2)**0.5 <= radius and (not(x == 0 and y == 0) or include_origin)
        ]
        offsets = torch.tensor(
            offsets,
            dtype=self.src_points.dtype,
            device=self.src_points.device
        )
        return offsets
    
    def _radius_sample(self, feat_map, points, offsets):
        n = points.shape[0]
        points = points.reshape([1, 1, n, 2])
        points = points.repeat([1, len(offsets), 1, 1])
        points += offsets.reshape([1, len(offsets), 1, 2])
        feats = F.grid_sample(feat_map, points)
        return feats[0]
    
    def _print(self, *args):
        if self._debug_log:
            print(*args)
    
    @torch.no_grad()
    def search_handle(self, prev_feat: Tensor, feat_map: Tensor):
        n = self.src_points.shape[0]
        feat_map = torch.unsqueeze(feat_map, 0).float()
        prev_feat = torch.unsqueeze(prev_feat, 0).float()
        
        # src_pts_4d = self.src_points.reshape([1, 1, n, 2])
        src_pts_4d = self.handle_points.reshape([1, 1, n, 2])
        src_feats = F.grid_sample(prev_feat, src_pts_4d)
        src_feats = rearrange(src_feats, "b c h w -> (b h w) c")

        offsets = self.search_offsets
        radius_feats = self._radius_sample(feat_map, self.handle_points, offsets)
        radius_feats = rearrange(radius_feats, "c offsets pts -> pts offsets c")
        
        new_handles = []
        for i, (src_pt, src_feat, rad_feat) in enumerate(zip(self.handle_points, src_feats, radius_feats)):
            A = F.normalize(src_feat.unsqueeze(0), dim=1)
            B = F.normalize(rad_feat, dim=1)
            dot = A @ B.T
            nearest = torch.argmax(dot, dim=1)
            new_handles.append(src_pt + offsets[nearest[0]])
            self._print(f"handle point[{i}] moved: ", offsets[nearest[0]])
        
        self.handle_points = torch.stack(new_handles)
        self._print(
            "abs(handle_points - dst_points) = ", 
            torch.abs(self.handle_points - self.dst_points).sum(dim=1).cpu().numpy(),
            torch.abs(self.ref_latent - self.init_latent).mean().cpu().numpy(),
        )

    def step(self, feat_map: Tensor, prev_feat=None, reset_backbound=False):
        c, h, w = feat_map.shape
        offsets = self.sample_offsets
        feat_map = torch.unsqueeze(feat_map, dim=0).float()
        prev_feat = torch.unsqueeze(prev_feat, dim=0).float() if prev_feat is not None else feat_map
        
        src_feats = self._radius_sample(prev_feat, self.handle_points, offsets)

        norm_direct = F.normalize(self.dst_points - self.handle_points)
        track_points = self.handle_points + norm_direct * 2 / min(h, w) * 4  # NOTE: try to move one "pixel" toward dst point on feature map
        # track_points = self.handle_points + (self.dst_points - self.handle_points) / self.steps
        tar_feats = self._radius_sample(feat_map, track_points, offsets)
        
        self.adam.zero_grad()
        motion_l1_loss = F.l1_loss(tar_feats, src_feats.detach())
        motion_l1_loss += F.l1_loss(self.ref_latent * self.mask, self.init_latent * self.mask) * 0.1
        motion_l1_loss.backward()
        self.adam.step()
        
        if reset_backbound:
            self.ref_latent.data = self.ref_latent * (1 - self.mask) + self.init_latent * self.mask
        
        self._print(colored("motion_l1_loss: ", color='green'), float(motion_l1_loss))
        return float(motion_l1_loss.detach().cpu())