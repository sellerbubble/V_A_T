from datetime import datetime
from typing import Callable, Optional, Sequence, Tuple, Type, Union, List, Dict
from timm.models.vision_transformer import VisionTransformer, Block
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
import math
import torch.nn.functional as F
import logging

import os
import numpy as np
from typing import Dict, Any

class AttentionSaveContext:
    _context: Dict[str, Any] = {}
    
    @classmethod
    def set(cls, key: str, value: Any):
        cls._context[key] = value
        
    @classmethod
    def get(cls, key: str, default: Any = None):
        return cls._context.get(key, default)

AttentionSaveContext.set("save_root", "/limx/tos/users/wenhao/saved_attention")


__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)

class OptimizedHierarchicalPE(nn.Module):
    def __init__(self, action_chunk, action_dim, embed_dim):
        super().__init__()
        self.action_embed = nn.Parameter(
            torch.randn(action_chunk, embed_dim) * (embed_dim**-0.5) * 5
        )
        
        self.dim_embed = nn.Parameter(
            torch.randn(action_dim, embed_dim) * (embed_dim**-0.5) * 0.5
        )

        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self.embed_dim = embed_dim
    
    def forward(self):
     
        matrix = self.action_embed.unsqueeze(1) + self.dim_embed.unsqueeze(0)
        return matrix.reshape(1, -1, self.embed_dim)  # [1, A*B, D]


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SeparatedAttention(nn.Module):
    fused_attn: Final[bool]
    
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        vat_tokens=57,
        action_num=56,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False
        self.action_num = action_num
        self.vat_tokens = vat_tokens
        
        self.q_act = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_vis = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj_act = nn.Linear(dim, dim)
        self.proj_drop_act = nn.Dropout(proj_drop)

    def forward(self, action_tokens, vision_features):
        """
        Args:
            action_tokens: [B, vat_tokens, C] Action tokens
            vision_features: [B, num_vision_tokens, C] Vision tokens
        Returns:
            x_act: [B, vat_tokens, C] Updated action tokens
        """
        B, N_act, C = action_tokens.shape
        B, N_vis, C = vision_features.shape
        
        q_act = self.q_act(action_tokens)
        q_act = q_act.reshape(B, self.vat_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        kv_vis = self.kv_vis(vision_features)
        kv_vis = kv_vis.reshape(B, N_vis, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_vis, v_vis = kv_vis.unbind(0)
        
        attn_act = (q_act @ k_vis.transpose(-2, -1)) * self.scale
        attn_act = attn_act.softmax(dim=-1)
        attn_act = self.attn_drop(attn_act)
        x_act = (attn_act @ v_vis).transpose(1, 2).reshape(B, self.vat_tokens, C)
        
        x_act = self.proj_act(x_act)
        x_act = self.proj_drop_act(x_act)
        
        save_dir = AttentionSaveContext.get("save_dir")
        block_idx = AttentionSaveContext.get("current_block")
        if save_dir and block_idx is not None and AttentionSaveContext.get("save_attention"):
            self._save_attention_scores(attn_act, save_dir, block_idx)

        return x_act
    

    def _save_attention_scores(self, attn_tensor, save_dir, block_idx):
      
        attn_np = attn_tensor.detach().cpu().numpy()
        
        layer_dir = os.path.join(save_dir, f"block_{block_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        
        for b in range(attn_np.shape[0]):
            batch_path = os.path.join(layer_dir, f"batch_{b}.npy")
            np.save(batch_path, attn_np[b])
    
class SeparatedAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        vat_tokens=57,
        action_num=56,
        embed_dim=1152,
        use_film=True
    ):
        super().__init__()
        self.vat_tokens = vat_tokens
        self.action_num = action_num
        self.embed_dim = embed_dim
        self.norm1_act = norm_layer(dim)
        self.norm1_vis = norm_layer(dim)
        self.use_film = use_film
        
        self.attn = SeparatedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            vat_tokens=vat_tokens,
            action_num=action_num,
        )
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_act = norm_layer(dim)
        
        self.mlp_act = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.film_modulator = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, 2 * self.embed_dim)
        )
        nn.init.normal_(self.film_modulator[0].weight, std=0.02)
        nn.init.normal_(self.film_modulator[2].weight, std=0.02)
        nn.init.zeros_(self.film_modulator[0].bias)
        nn.init.zeros_(self.film_modulator[2].bias)

    def forward(self, action_tensor, task_embedding, vision_features):
        """
        Args:
            action_tensor: [B, vat_tokens, dim] Action tokens
            task_embedding: [B, 1, embed_dim] Task embedding
            vision_features: [B, num_vision_tokens, dim] Vision tokens
        Returns:
            action_tensor: [B, vat_tokens, dim] Updated action tokens
        """
        if self.use_film:
            film_params = self.film_modulator(task_embedding)  # [B, 1, 2*embed_dim]
            gamma, beta = torch.split(film_params, self.embed_dim, dim=2)  
            action_tensor = action_tensor * (gamma + 1) + beta
        
        norm_act = self.norm1_act(action_tensor)
        norm_vis = self.norm1_vis(vision_features)
        
        attn_out = self.attn(norm_act, norm_vis)
        attn_out = self.ls1(attn_out)
        
        action_tensor = action_tensor + self.drop_path1(attn_out)
        
        norm_act2 = self.norm2_act(action_tensor)
        
        mlp_out = self.mlp_act(norm_act2)
        mlp_out = self.ls2(mlp_out)
        
        action_tensor = action_tensor + self.drop_path2(mlp_out)
        
        return action_tensor 
    
class ActionModuleTransformer(VisionTransformer):
    """ Vision Transformer with multi-scale feature extraction
    
    Extends the standard ViT to produce feature maps at multiple scales,
    similar to a Feature Pyramid Network (FPN) architecture.
    """
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        in_chans: int = 3,
        num_classes: int = 0,
        global_pool: str = 'map',
        embed_dim: int = 1152,
        depth: int = 27,
        num_heads: int = 16,
        mlp_ratio: float = 3.7362,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = False,
        # set no embed class to True for now, so action token has no pos embed
        no_embed_class: bool = True,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: str = '',
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        action_dim: int = 7,
        action_chunk: int = 1,
        use_diffusion: bool = False,
        use_proprio: bool = False,
        use_film: bool = True,
        taskembedding_add: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )

        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token

        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.action_num = action_chunk * action_dim
        self.num_prefix_tokens += self.action_num
        self.num_prefix_tokens += 1 if use_proprio else 0
        self.use_diffusion = use_diffusion
        self.num_prefix_tokens += 1 if self.use_diffusion else 0
        self.taskembedding_add = taskembedding_add
        assert not (self.taskembedding_add and use_film), "taskembedding_add and use_film cannot be True at the same time"
        
        # consider self.action_num and self.use_proprio and self.use_task_id and self.use_diffusion
        self.vat_tokens = sum([
            self.action_num,
            use_proprio,
            self.use_diffusion,
        ])
        # dynamic img size is True for we need to interpolate the original pos embed
        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        self.action_token = nn.Parameter(torch.zeros(1, self.action_num, embed_dim)) 
        # only calculate cls token and reg token because the embed len of siglip patch_embed is up to cls and reg token
        # and we need to load siglip patch_embed parameter so the embed len of actionvision vit should be the same
        # embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens - self.action_num
        embed_len = num_patches # for siglip and vit large embed len = num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        # only add action pos embed to action token, not all prefix tokens
        # self.action_pos_embed = nn.Parameter(torch.randn(1, self.action_num, embed_dim) *.02)
        self.action_pos_embed_generator = OptimizedHierarchicalPE(action_chunk, action_dim, embed_dim)
        

        self.task_embedding = nn.Embedding(num_embeddings=30, embedding_dim=embed_dim)

        # same with VisionTransformer
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            SeparatedAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                vat_tokens=self.vat_tokens,
                action_num=self.action_num,
                embed_dim=embed_dim,
                use_film=use_film,
            )
            for i in range(depth)])

        # keep norm as identity, for Father class VisionTransformer need norm as trainable params
        self.norm = nn.Identity()
        self.norm_non_act = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.norm_act = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
    
    def _intermediate_layers(
            self,
            vision_features: torch.Tensor,
            n: Union[int, Sequence] = 1,
            diffusion_timestep_embeddings: Optional[torch.Tensor] = None,
            noisy_action_features: Optional[torch.Tensor] = None,
            proprio: Optional[torch.Tensor] = None,
            task_id: Optional[torch.Tensor] = None,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        # vision_features is a list of tensors, each shape [B, image_num*token_num, embed_dim]
        task_embedding = self.task_embedding(task_id)
        action_pos_embed = self.action_pos_embed_generator()
        action_tensor = self.action_token.expand(vision_features[0].shape[0], -1, -1) + action_pos_embed           
        if task_embedding is not None and self.taskembedding_add:
            action_tensor = action_tensor + task_embedding
        
        
        save_attention = False
        AttentionSaveContext.set("save_attention", save_attention)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.attention_save_dir = os.path.join(
            AttentionSaveContext.get("save_root"),
            f"attentions_{timestamp}_object"
        )

        for i, blk in enumerate(self.blocks):

            AttentionSaveContext.set("current_block", i)
            AttentionSaveContext.set("save_dir", self.attention_save_dir)

            select_features_mode = "second last layer features"
            # print(f"select_features_mode: {select_features_mode}")
            select_features = vision_features[-2]
            
            action_tensor = blk(action_tensor, task_embedding, select_features)
            if i in take_indices:
                outputs.append(action_tensor)
                break

            AttentionSaveContext.set("save_dir", None)

        return outputs

    def get_intermediate_layers(
            self,
            vision_features: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            diffusion_timestep_embeddings: Optional[torch.Tensor] = None,
            noisy_action_features: Optional[torch.Tensor] = None,
            proprio: Optional[torch.Tensor] = None,
            task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(vision_features, n, diffusion_timestep_embeddings, noisy_action_features, proprio, task_id)

        action_tokens = [out[:, : self.action_num] for out in outputs]

        return tuple(zip([None], action_tokens))
