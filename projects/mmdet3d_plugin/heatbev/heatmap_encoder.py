from mmcv.runner.base_module import BaseModule
from mmdet3d.models.builder import HEADS

from custom_modules.coord_utils import denormalize_coords, generate_3d_grid, point_sampling


@HEADS.register_module()
class HeatmapEncoder(BaseModule):
    def __init__(self, sampling_grid_size, sampling_grid_range):
        super().__init__()

        self.sampling_grid_size = sampling_grid_size
        self.sampling_grid_range = sampling_grid_range

        self.register_buffer('ref_3d', self.generate_ref_3d(sampling_grid_size, sampling_grid_range))  # (P, 3)

    def generate_ref_3d(self, grid_size, grid_range):
        xyz_norm = generate_3d_grid(*grid_size)
        xyz_real = denormalize_coords(xyz_norm, pc_range=grid_range)

        return xyz_real

    def forward(self, img_feats, img_metas, **kwargs):
        B, N, C, H, W = img_feats.shape

        ref_pixel, cam_mask = point_sampling(
            coords_3d=self.ref_3d.unsqueeze(0).repeat(B, 1, 1),  # (B, P, 3)
            img_metas=img_metas
        )                                                        # (B, N, P, 2), (B, N, P)

        ref_pixel_ = ref_pixel * ref_pixel.new_tensor([W, H])
        ref_pixel_ = ref_pixel_.long()

        volume_feats = img_feats.new_zeros([B, C, len(self.ref_3d)])  # (B, C, P)
        for batch in range(B):
            for cam in range(N):
                valid = cam_mask[batch, cam, :]

                x = ref_pixel_[..., 0][batch, cam, valid]
                y = ref_pixel_[..., 1][batch, cam, valid]

                assert ((x >= 0) & (x < W) & (y >= 0) & (y < H)).all()

                volume_feats[batch, :, valid] = img_feats[batch, cam, :, y, x]

        output = volume_feats.reshape(B, C, *self.sampling_grid_size)  # (B, C, Dh, Hh, Wh)

        return output
