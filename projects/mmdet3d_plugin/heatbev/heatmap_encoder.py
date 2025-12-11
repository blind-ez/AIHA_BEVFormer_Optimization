import copy

from mmcv import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import build_from_cfg

from custom_modules.coord_utils import denormalize_coords, generate_3d_grid, point_sampling


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class HeatmapEncoder(TransformerLayerSequence):
    def __init__(self,
                 sampling_grid_size,
                 sampling_grid_range,
                 **kwargs):
        super().__init__(**kwargs)

        self.register_buffer('ref_3d', self.generate_ref_3d(sampling_grid_size, sampling_grid_range))  # (P, 3)

    def generate_ref_3d(self, grid_size, grid_range):
        xyz_norm = generate_3d_grid(*grid_size)
        xyz_real = denormalize_coords(xyz_norm, pc_range=grid_range)
        return xyz_real

    def forward(self,
                value,              # (B, N, ΣHW, C)
                spatial_shapes,     # (S, 2)
                level_start_index,  # (S,)
                img_metas,
                **kwargs):
        B, _, _, _ = value.shape

        ref_pixel, cam_mask = point_sampling(
            coords_3d=self.ref_3d.unsqueeze(0).repeat(B, 1, 1),  # (B, P, 3)
            img_metas=img_metas
        )                                                        # (B, N, P, 2), (B, N, P)

        for lid, layer in enumerate(self.layers):
            output = layer(
                value=value,                          # (B, N, ΣHW, C)
                ref_pixel=ref_pixel,                  # (B, N, P, 2)
                cam_mask=cam_mask,                    # (B, N, P)
                spatial_shapes=spatial_shapes,        # (S, 2)
                level_start_index=level_start_index,  # (S,)
                ref_3d=self.ref_3d,                   # (P, 3)
                **kwargs
            )

        return output


@TRANSFORMER_LAYER.register_module()
class HeatmapEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 operation_order,
                 attn_cfg=None,
                 ffn_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.operation_order = operation_order

        self.pre_norm = (self.operation_order[0] == 'norm')

        if attn_cfg is not None:
            attn_name_list = list()
            for operation_name in self.operation_order:
                if 'attn' in operation_name:
                    attn_name_list.append(operation_name)
            num_attns = len(attn_name_list)
            if isinstance(attn_cfg, dict):
                attn_cfg = ConfigDict(attn_cfg)
            if isinstance(attn_cfg, dict):
                attn_cfg = [copy.deepcopy(attn_cfg) for _ in range(num_attns)]
            assert len(attn_cfg) == num_attns
            self.attentions = ModuleList()
            for attn_index in range(num_attns):
                if 'embed_dims' not in attn_cfg[attn_index]:
                    attn_cfg[attn_index]['embed_dims'] = self.embed_dims
                else:
                    assert attn_cfg[attn_index]['embed_dims'] == self.embed_dims
                attention = build_from_cfg(attn_cfg[attn_index], ATTENTION)
                attention.operation_name = attn_name_list[attn_index]
                self.attentions.append(attention)

        if ffn_cfg is not None:
            num_ffns = self.operation_order.count('ffn')
            if isinstance(ffn_cfg, dict):
                ffn_cfg = ConfigDict(ffn_cfg)
            if isinstance(ffn_cfg, dict):
                ffn_cfg = [copy.deepcopy(ffn_cfg) for _ in range(num_ffns)]
            assert len(ffn_cfg) == num_ffns
            self.ffns = ModuleList()
            for ffn_index in range(num_ffns):
                if 'embed_dims' not in ffn_cfg[ffn_index]:
                    ffn_cfg[ffn_index]['embed_dims'] = self.embed_dims
                else:
                    assert ffn_cfg[ffn_index]['embed_dims'] == self.embed_dims
                ffn = build_from_cfg(ffn_cfg[ffn_index], FEEDFORWARD_NETWORK)
                self.ffns.append(ffn)

        if norm_cfg is not None:
            num_norms = self.operation_order.count('norm')
            self.norms = ModuleList()
            for _ in range(num_norms):
                norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
                self.norms.append(norm)

    def forward(self,
                query=None,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                **kwargs): 
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        for layer in self.operation_order:
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query=query,
                    key=query,
                    value=query,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    **kwargs
                )
                attn_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query=query,
                    key=key,
                    value=value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    **kwargs
                )
                attn_index += 1

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query)
                ffn_index += 1

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

        return query
