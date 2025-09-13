import functools
from typing import List, Optional, Union, Tuple, Dict, Any
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import torch.distributed as dist
import torch
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode


def parallelize_transformer(transformer: QwenImageTransformer2DModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()
    world_size = dist.get_world_size()


    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None, 
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        temp1 = image_rotary_emb[0]
        temp1 = DP.get_assigned_chunk(temp1, dim=0, group=seq_mesh)
        temp2 = image_rotary_emb[1] 

        remainder = image_rotary_emb[1].shape[0] % world_size
        if remainder != 0:
            pad_amount = world_size - remainder
            temp2 = F.pad(temp2, (0, 0, 0, pad_amount))


        temp2= DP.get_assigned_chunk(temp2 , dim=0, group=seq_mesh)
        image_rotary_emb = (temp1,temp2)
        remainder = encoder_hidden_states.shape[1] % world_size

        if remainder != 0:
            pad_amount = world_size - remainder
            encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, pad_amount))

        remainder = encoder_hidden_states_mask.shape[1] % world_size
        if remainder != 0:
            pad_amount = world_size - remainder
            encoder_hidden_states_mask = F.pad(encoder_hidden_states_mask, (0, pad_amount))




        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = DP.get_assigned_chunk(timestep, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)

        encoder_hidden_states_mask = DP.get_assigned_chunk(encoder_hidden_states_mask, dim=1, group=seq_mesh)



        with UnifiedAttnMode(mesh):
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)
        hidden_states = self.norm_out(hidden_states, temb)

        output = self.proj_out(hidden_states)

        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    transformer.forward = new_forward.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_parallelized = True

    if not shallow_patch:
        parallelize_transformer(pipe.transformer, **kwargs)

    return pipe