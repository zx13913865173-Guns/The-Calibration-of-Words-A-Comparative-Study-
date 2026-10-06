import re
from typing import List, Dict

class WeightSyntax:
    def __init__(self, config: dict):
        self.cfg = config

    def apply_sd_native(self, prompt: str, target: str, level: str) -> str:
        mapping = {
            "W1": ("[", "]"),   # 两层降权
            "W2": ("[", "]"),   # 一层降权
            "W3": ("(", ")"),   # 一层升权
            "W4": ("((", "))"), # 两层升权
            "W5": ("(((", ")))")# 三层升权
        }
        left, right = mapping[level]
        return prompt.replace(target, f"{left}{target}{right}")

    def apply_comfyui(self, prompt: str, target: str, level: str) -> str:
        value = self.cfg["weight_levels"][level]["comfy"]
        return prompt.replace(target, f"({target}:{value})")

    def apply_novelai(self, prompt: str, target: str, level: str) -> str:
        mapping = {
            "W1": ("[[", "]]"),
            "W2": ("[", "]"),
            "W3": ("{", "}"),
            "W4": ("{{", "}}"),
            "W5": ("{{{", "}}}")
        }
        left, right = mapping[level]
        return prompt.replace(target, f"{left}{target}{right}")

    def apply(self, prompt: str, target: str, level: str, syntax: str) -> str:
        if syntax == "sd_native":
            return self.apply_sd_native(prompt, target, level)
        if syntax == "comfyui":
            return self.apply_comfyui(prompt, target, level)
        if syntax == "novelai":
            return self.apply_novelai(prompt, target, level)
        raise ValueError(f"Unknown syntax: {syntax}")
