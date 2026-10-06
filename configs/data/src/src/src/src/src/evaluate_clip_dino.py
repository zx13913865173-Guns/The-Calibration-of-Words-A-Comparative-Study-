import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, Dinov2Model

def load_clip(device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

def load_dino(device="cuda"):
    model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    return model, processor

def clip_score(image, text, model, processor, device="cuda"):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return float(outputs.logits_per_image[0][0].cpu().numpy())

def dino_feature(image, model, processor, device="cuda"):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def evaluate_folder(folder, target_word, out_json):
    clip_model, clip_proc = load_clip()
    dino_model, dino_proc = load_dino()
    results = []
    for fname in tqdm(sorted(os.listdir(folder))):
        if not fname.endswith(".png"):
            continue
        path = os.path.join(folder, fname)
        image = Image.open(path).convert("RGB")
        cs = clip_score(image, target_word, clip_model, clip_proc)
        feat = dino_feature(image, dino_model, dino_proc)
        results.append({"file": fname, "clip": cs, "dino": feat.tolist()})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 示例：对所有输出文件夹循环
    pass
