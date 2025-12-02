from datetime import datetime
from typing import Callable, Optional, Sequence, Tuple, Type, Union, Dict, Protocol
from timm.models.vision_transformer import VisionTransformer, Block

from functools import partial

import torch
import torch.nn as nn
from torch.jit import Final

from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType

import os
import numpy as np
from typing import Dict, Any
from PIL.Image import Image

import timm
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.actionmodule_modeling import ActionModuleTransformer
from prismatic.models.backbones.vision.vat_small_modeling import VATSMALL
from prismatic.models.backbones.vision.vat_vit_modeling import VATVIT

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


class OptimizedHierarchicalPositionEmbedding(nn.Module):
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
        
      
        self.qkv_non_act = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_non = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_non = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        assert qk_norm == False, "qk_norm must be False for right now code doesn't support qk norm for both non act token and act token"
        
    
        self.q_act = nn.Linear(dim, dim, bias=qkv_bias) 
        self.kv_act = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
    
        self.proj_non_act = nn.Linear(dim, dim)
        self.proj_drop_non_act = nn.Dropout(proj_drop)
        self.proj_act = nn.Linear(dim, dim)     
        self.proj_drop_act = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        num_non_act = N - self.vat_tokens
        
        if num_non_act == 256 or num_non_act == 261:
      
            non_act_tokens = x[:, self.vat_tokens:]  
            qkv_non_act = self.qkv_non_act(non_act_tokens)
            qkv_non_act = qkv_non_act.reshape(B, num_non_act, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q_non, k_non, v_non = qkv_non_act.unbind(0)
         

            attn_non = (q_non @ k_non.transpose(-2, -1)) * self.scale
            attn_non = attn_non.softmax(dim=-1)
            attn_non = self.attn_drop(attn_non)
            x_non = (attn_non @ v_non).transpose(1, 2).reshape(B, num_non_act, C)
            
  
            act_tokens = x[:, :self.vat_tokens]
            q_act = self.q_act(act_tokens)
            q_act = q_act.reshape(B, self.vat_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
 
            kv_act = self.kv_act(x) 
            kv_act = kv_act.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k_act, v_act = kv_act.unbind(0)
            
    
            k_act = k_act[:, :, self.action_num:]
            v_act = v_act[:, :, self.action_num:]

 
            attn_act = (q_act @ k_act.transpose(-2, -1)) * self.scale
            attn_act = attn_act.softmax(dim=-1)
            attn_act = self.attn_drop(attn_act)
            x_act = (attn_act @ v_act).transpose(1, 2).reshape(B, self.vat_tokens, C)
    
            x_non = self.proj_non_act(x_non)
            x_non = self.proj_drop_non_act(x_non)
            x_act = self.proj_act(x_act)
            x_act = self.proj_drop_act(x_act)
            
  
            x_out = torch.cat([x_act, x_non], dim=1) 
        else:

            non_act_tokens = x[:, self.vat_tokens:]  # [B, num_non_act, C]

       
            if num_non_act in [512, 522]:
                n_groups = 2
            elif num_non_act in [768, 783]:
                n_groups = 3
            else:
            
                raise ValueError(f"Unsupported num_non_act value: {num_non_act}. "
                                "Expected 512, 522, 768, or 783.")

            group_size = num_non_act // n_groups 

       
            assert num_non_act % n_groups == 0, (
                f"num_non_act ({num_non_act}) must be divisible by n_groups ({n_groups})"
            )

          
            group_outputs = []

      
            for i in range(n_groups):
       
                start_idx = i * group_size
                end_idx = (i + 1) * group_size
                group_tokens = non_act_tokens[:, start_idx:end_idx]  # [B, group_size, C]
                
        
                qkv = self.qkv_non_act(group_tokens).reshape(
                    B, group_size, 3, self.num_heads, self.head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)  # [B, num_heads, group_size, head_dim]
                
            
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
          
                group_out = (attn @ v).transpose(1, 2).reshape(B, group_size, C)
                group_outputs.append(group_out)

        
            x_non = torch.cat(group_outputs, dim=1)  # [B, num_non_act, C]

            
      
            act_tokens = x[:, :self.vat_tokens]  
            q_act = self.q_act(act_tokens)
            q_act = q_act.reshape(B, self.vat_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
           
            kv_act = self.kv_act(x) 
            kv_act = kv_act.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k_act, v_act = kv_act.unbind(0)

        
            k_act = k_act[:, :, self.action_num:]
            v_act = v_act[:, :, self.action_num:]
            
       
            attn_act = (q_act @ k_act.transpose(-2, -1)) * self.scale
            attn_act = attn_act.softmax(dim=-1)
            attn_act = self.attn_drop(attn_act)
            x_act = (attn_act @ v_act).transpose(1, 2).reshape(B, self.vat_tokens, C)
            
       
            x_non = self.proj_non_act(x_non)
            x_non = self.proj_drop_non_act(x_non)
            x_act = self.proj_act(x_act)
            x_act = self.proj_drop_act(x_act)
            
         
            x_out = torch.cat([x_act, x_non], dim=1)

            save_dir = AttentionSaveContext.get("save_dir")
            block_idx = AttentionSaveContext.get("current_block")
            if save_dir and block_idx is not None and AttentionSaveContext.get("save_attention"):
                self._save_attention_scores(attn_act, save_dir, block_idx)

        return x_out     
    

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
        self.norm1_non_act = norm_layer(dim)  
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
        
   
        # assert init_values is None, "init_values should be None for Identity LayerScale"
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    
        self.norm2_act = norm_layer(dim)
        self.norm2_non_act = norm_layer(dim)
        
   
        self.mlp_non_act = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
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

    def forward(self, x, task_embedding):

 
        act_tokens = x[:, :self.vat_tokens] 
        non_act_tokens = x[:, self.vat_tokens:]  

        # 1.5 FILM
        if self.use_film:
            film_params = self.film_modulator(task_embedding) 
            gamma, beta = torch.split(film_params, self.embed_dim, dim=2)  
         
            act_tokens = act_tokens * (gamma + 1) + beta
        
     
      
        norm_act = self.norm1_act(act_tokens)
        norm_non_act = self.norm1_non_act(non_act_tokens)
        
 
        norm_x = torch.cat([norm_act, norm_non_act], dim=1)
        
     
        attn_out = self.attn(norm_x)
        attn_out = self.ls1(attn_out)
      
        x = x + self.drop_path1(attn_out)
        
    
    
        act_tokens = x[:, :self.vat_tokens]
        non_act_tokens = x[:, self.vat_tokens:]
        
    
        norm_act2 = self.norm2_act(act_tokens)
        norm_non_act2 = self.norm2_non_act(non_act_tokens)
        
   
        act_out = self.mlp_act(norm_act2)
        non_act_out = self.mlp_non_act(norm_non_act2)
        
   
        mlp_out = torch.cat([act_out, non_act_out], dim=1)
        mlp_out = self.ls2(mlp_out)

       
        x = x + self.drop_path2(mlp_out)
        
        return x   
    
class VisionActionTransformer(nn.Module):
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
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        num_classes = num_classes
        global_pool = global_pool
    

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
        # consider action_num and use_proprio and use_task_id and use_diffusion
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
        # embed_len = num_patches if no_embed_class else num_patches + num_prefix_tokens - action_num
        embed_len = num_patches # for siglip and vit large embed len = num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        # only add action pos embed to action token, not all prefix tokens
        # self.action_pos_embed = nn.Parameter(torch.randn(1, self.action_num, embed_dim) *.02)
        self.action_pos_embed_generator = OptimizedHierarchicalPositionEmbedding(action_chunk, action_dim, embed_dim)
        

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


    def _pos_embed(self, x, diffusion_timestep_embeddings: Optional[torch.Tensor] = None,
            noisy_action_features: Optional[torch.Tensor] = None,
            proprio: Optional[torch.Tensor] = None,
            task_embedding: Optional[torch.Tensor] = None,
            ):
      
        pos_embed = self.pos_embed
        action_pos_embed = self.action_pos_embed_generator()

        to_cat = []
        if self.action_token is not None:
            if noisy_action_features is not None:
                action_tensor = noisy_action_features + action_pos_embed
            else:
                action_tensor = self.action_token.expand(x.shape[0], -1, -1) + action_pos_embed    
            if task_embedding is not None and self.taskembedding_add:
                action_tensor = action_tensor + task_embedding
            to_cat.append(action_tensor)

        if diffusion_timestep_embeddings is not None:
            to_cat.append(diffusion_timestep_embeddings)
        if proprio is not None:
            to_cat.append(proprio)
      
        assert x.shape[1] % 256 == 0, f"Input channels should be multiple of 256, got {x.shape[1]}"
        
  
        num_groups = x.shape[1] // 256
        group_tokens = []
        
      
        for i in range(num_groups):
            group = x[:, i*256:(i+1)*256, :]
            
          
            group = group + pos_embed
            
          
            special_tokens = []
            if self.cls_token is not None:
                special_tokens.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                special_tokens.append(self.reg_token.expand(x.shape[0], -1, -1))
            
  
            group_tokens.append(torch.cat(special_tokens + [group], dim=1))

      
        x = torch.cat(group_tokens, dim=1)
            
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
  

        return self.pos_drop(x)
    
    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            diffusion_timestep_embeddings: Optional[torch.Tensor] = None,
            noisy_action_features: Optional[torch.Tensor] = None,
            proprio: Optional[torch.Tensor] = None,
            task_id: Optional[torch.Tensor] = None,
            save_attention: Optional[bool] = False,
            attention_save_path: Optional[str] = None,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        if x.shape[1] != 3:
       
            assert x.shape[1] % 3 == 0, f"Input channels should be multiple of 3, got {x.shape[1]}"
            
       
            num_groups = x.shape[1] // 3
            token_groups = []
            
      
            for i in range(num_groups):
         
                x_group = x[:, i*3:(i+1)*3, :, :]
           
                tokens_group = self.patch_embed(x_group)
                token_groups.append(tokens_group)
            
          
            x = torch.cat(token_groups, dim=1)  # [B, num_groups*num_patches, embed_dim]
        else:
            x = self.patch_embed(x)
        task_embedding = self.task_embedding(task_id)
        x = self._pos_embed(x, diffusion_timestep_embeddings, noisy_action_features, proprio, task_embedding)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
   

        AttentionSaveContext.set("save_attention", save_attention)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.attention_save_dir = os.path.join(
            AttentionSaveContext.get("save_root"),
            f"{timestamp}_{attention_save_path}"
        )

        for i, blk in enumerate(self.blocks):

            AttentionSaveContext.set("current_block", i)
            AttentionSaveContext.set("save_dir", self.attention_save_dir)

            x = blk(x, task_embedding)
            if i in take_indices:
                outputs.append(x)
                break

     
            AttentionSaveContext.set("save_dir", None)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            diffusion_timestep_embeddings: Optional[torch.Tensor] = None,
            noisy_action_features: Optional[torch.Tensor] = None,
            proprio: Optional[torch.Tensor] = None,
            task_id: Optional[torch.Tensor] = None,
            save_attention: Optional[bool] = False,
            attention_save_path: Optional[str] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n, diffusion_timestep_embeddings, noisy_action_features, proprio, task_id, save_attention, attention_save_path)
        
        prefix_tokens = [out[:, : self.action_num] for out in outputs]
        outputs = [out[:, self.action_num:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        
        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

vit_large_param_names = {
    "img_size": 256,
    "patch_size": 16, 
    "in_chans": 3, 
    "num_classes": 0, 
    "global_pool": "map", 
    "embed_dim": 1536, 
    "depth": 40, 
    "num_heads": 16, 
    "mlp_ratio": 4.0, 
    "qkv_bias": True, 
    "qk_norm": False, 
    "init_values": None, 
    "class_token": False, 
    "no_embed_class": False,
    "reg_tokens": 0, 
    "pre_norm": False, 
    "fc_norm": None,
    "dynamic_img_size": False, 
    "dynamic_img_pad": False, 
    "drop_rate": 0,
    "pos_drop_rate": 0, 
    "patch_drop_rate": 0, 
    "proj_drop_rate": 0,
    "attn_drop_rate": 0, 
    "drop_path_rate": 0, 
    "weight_init": '', 
}
 
dinov2_param_names = {
    "img_size": 224,
    "patch_size": 14, 
    "in_chans": 3, 
    "num_classes": 0, 
    "global_pool": "token", 
    "embed_dim": 1024, 
    "depth": 24, 
    "num_heads": 16, 
    "mlp_ratio": 4.0, 
    "qkv_bias": True, 
    "qk_norm": False, 
    "init_values": 1e-05, 
    "class_token": True, 
    "no_embed_class": True,
    "reg_tokens": 4, 
    "pre_norm": False, 
    "fc_norm": None,
    "dynamic_img_size": False, 
    "dynamic_img_pad": False, 
    "drop_rate": 0,
    "pos_drop_rate": 0, 
    "patch_drop_rate": 0, 
    "proj_drop_rate": 0,
    "attn_drop_rate": 0, 
    "drop_path_rate": 0, 

}

def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper

DEFAULT_IMAGE_SIZE = 224

def get_vat_backbone_and_transform(
    vat_backbone_id: str, 
    image_resize_strategy: str, 
    action_dim: int, 
    action_chunk: int, 
    use_diffusion: bool, 
    use_proprio: bool = False, 
    end_lastlayer: int = 2,
    use_film: bool = True,
    taskembedding_add: bool = False,
    baseline: bool = False,
    vat_small_factor: int = 1,
    vat_vit: bool = False,
) -> Tuple[VisionActionTransformer, ImageTransform]:

    vat_backbone_params = {}
    if 'giantopt' in vat_backbone_id: 
        vat_backbone_params = vit_large_param_names
    elif 'dinov2' in vat_backbone_id:
        vat_backbone_params = dinov2_param_names
    
    if baseline == False:
        if vat_small_factor != 1:
            vat_backbone = VATSMALL(action_dim=action_dim, 
                            action_chunk=action_chunk, 
                            use_diffusion=use_diffusion,
                            use_proprio=use_proprio,
                            use_film=use_film,
                            taskembedding_add=taskembedding_add,
                            embed_scaling_factor=vat_small_factor,
                            **vat_backbone_params
                            )
        else:
            if vat_vit:
                vat_backbone = VATVIT(
                            action_dim=action_dim, 
                            action_chunk=action_chunk, 
                            use_diffusion=use_diffusion,
                            use_proprio=use_proprio,
                            use_film=use_film,
                            taskembedding_add=taskembedding_add,
                            **vat_backbone_params
                            )
            else:
                vat_backbone = VisionActionTransformer(
                            action_dim=action_dim, 
                            action_chunk=action_chunk, 
                            use_diffusion=use_diffusion,
                            use_proprio=use_proprio,
                            use_film=use_film,
                            taskembedding_add=taskembedding_add,
                            **vat_backbone_params
                            )
        vat_backbone.forward = unpack_tuple(
                    partial(vat_backbone.get_intermediate_layers, n={len(vat_backbone.blocks) - end_lastlayer}, 
                            return_prefix_tokens=True,)
                )
    else:
        vit = timm.create_model(
                        "vit_so400m_patch14_siglip_224", 
                        pretrained=False, 
                        num_classes=0, 
                        img_size=DEFAULT_IMAGE_SIZE,       
                    )
        vit.forward = lambda x: vit.get_intermediate_layers(
                x, n=list(range(len(vit.blocks))), return_prefix_tokens=False
                )
        vat = ActionModuleTransformer(
                        action_dim=action_dim, 
                        action_chunk=action_chunk, 
                        use_diffusion=use_diffusion,
                        use_proprio=use_proprio,
                        use_film=use_film,
                        taskembedding_add=taskembedding_add,
                        **vat_backbone_params
                )
        vat.forward = unpack_tuple(
                partial(vat.get_intermediate_layers, n={len(vat.blocks) - end_lastlayer}, 
                        return_prefix_tokens=True,)
            )
    # Get Config =>> Note :: Override default image size to ensure correct image transform
    if not baseline:
        data_cfg = timm.data.resolve_model_data_config(vat_backbone)
    else:
        data_cfg = timm.data.resolve_model_data_config(vat)
    data_cfg["input_size"] = (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    # Initialize Default Image Transform --> Modified by `image_resize_strategy`
    default_image_transform = timm.data.create_transform(**data_cfg, is_training=False)

    # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `default_image_size` (crops image)!
    if "siglip" in vat_backbone_id or "in1k" in vat_backbone_id:
        assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(default_image_transform.transforms[0], Resize)
        default_image_transform = Compose(
            [
                Resize(DEFAULT_IMAGE_SIZE, interpolation=default_image_transform.transforms[0].interpolation),
                *default_image_transform.transforms[1:],
            ]
        )

    # Switch on `image_resize_strategy`
    if image_resize_strategy == "resize-naive":
        assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
        assert isinstance(default_image_transform.transforms[0], Resize)

        target_size = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        image_transform = Compose(
            [
                Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                *default_image_transform.transforms[1:],
            ]
        )

    if not baseline:
        return vat_backbone, image_transform
    else:
        return (vit, vat), image_transform