import json
import requests
import yaml
import os

def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_workflow(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def queue_prompt(workflow, server="127.0.0.1:8188"):
    url = f"http://{server}/prompt"
    data = {"prompt": workflow}
    r = requests.post(url, json=data)
    r.raise_for_status()
    return r.json()

def generate_comfyui(cfg, workflow_path, out_dir):
    wf = load_workflow(workflow_path)
    os.makedirs(out_dir, exist_ok=True)
    # 这里需要根据具体 workflow 修改节点 ID 和输入
    # 示例：假设节点 3 是 KSampler，节点 6 是 CLIPTextEncode
    # 实际使用时请从 ComfyUI 导出 API 格式的 workflow
    for level in cfg["experiments"]["main"]["levels"]:
        for syntax in ["comfyui", "sd_native", "novelai"]:
            # 构造 weighted prompt
            # 这里省略具体替换逻辑，调用 syntax.py
            pass
    print("请根据 workflows/comfyui_sdxl_api.json 配置节点 ID 后运行。")

if __name__ == "__main__":
    cfg = load_config()
    generate_comfyui(cfg, "workflows/comfyui_sdxl_api.json", "outputs/images/comfyui")
