import torch, time
import torch.distributed as dist
from diffusers import DiffusionPipeline

model_name = "Qwen/Qwen-Image"


dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

pipe = DiffusionPipeline.from_pretrained(model_name,  torch_dtype=torch.bfloat16,).to("cuda")

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt,
    "zh": "超清，4K，电影级构图" # for chinese prompt,
}

prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

negative_prompt = " " # using an empty string if you do not have specific concept to remove


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
width, height = aspect_ratios["16:9"]



from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
    max_ring_dim_size=2,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

# from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# apply_cache_on_pipe(pipe)

# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
st = time.time()

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
     num_inference_steps=50,


    true_cfg_scale=4.0,
    output_type="pil" if dist.get_rank() == 0 else "pt",
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

print("Inference time: ", time.time()-st)
if dist.get_rank() == 0:
    print("Saving image to qwen.png")
    image.save(f"qwen_{dist.get_world_size()}.png")

dist.destroy_process_group()