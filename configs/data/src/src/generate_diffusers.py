import os
import json
import yaml
import torch
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, StableDiffusion3Pipeline, FluxPipeline
from syntax import WeightSyntax

def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for cat, items in data.items():
        for it in items:
            prompts.append(it)
    return prompts

def get_pipeline(model_key, cfg, device="cuda"):
    if model_key == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["models"]["sdxl"],
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
    elif model_key == "sd35":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            cfg["models"]["sd35"],
            torch_dtype=torch.float16
        ).to(device)
    elif model_key == "flux":
        pipe = FluxPipeline.from_pretrained(
            cfg["models"]["flux"],
            torch_dtype=torch.bfloat16
        ).to(device)
    else:
        raise ValueError(model_key)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def generate_main(cfg, prompts, syntax, out_dir):
    pipe = get_pipeline("sdxl", cfg)
    generator = torch.Generator("cuda").manual_seed(cfg["project"]["seed"])
    os.makedirs(out_dir, exist_ok=True)
    ws = WeightSyntax(cfg)
    for i, item in enumerate(tqdm(prompts)):
        prompt = item["prompt"]
        target = item["target"]
        for level in cfg["experiments"]["main"]["levels"]:
            weighted = ws.apply(prompt, target, level, syntax)
            image = pipe(
                prompt=weighted,
                negative_prompt=cfg["generation"]["negative_prompt"],
                num_inference_steps=cfg["generation"]["steps"],
                guidance_scale=cfg["generation"]["cfg_main"],
                generator=generator,
                width=cfg["generation"]["width"],
                height=cfg["generation"]["height"]
            ).images[0]
            fname = f"{syntax}_{level}_{i:03d}.png"
            image.save(os.path.join(out_dir, fname))
            # 保存元数据
            meta = {
                "prompt_id": i,
                "syntax": syntax,
                "level": level,
                "weight_value": cfg["weight_levels"][level][syntax.split("_")[0]],
                "target": target,
                "prompt": prompt,
                "weighted_prompt": weighted
            }
            with open(os.path.join(out_dir, fname.replace(".png", ".json")), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    cfg = load_config()
    prompts = load_prompts("data/prompts.json")
    for syntax in ["sd_native", "comfyui", "novelai"]:
        generate_main(cfg, prompts, syntax, f"outputs/images/main/{syntax}")
