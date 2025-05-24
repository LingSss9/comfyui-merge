"""
@author: cyberblackcat
@title: merge
@nickname: CBC
@description: This extension provides some nodes to support merge lora, adjust Lora Block Weight.

"""
# comfyui-merge-lora/__init__.py
import importlib

# 可用的模块列表
module_list = [
    "mergetools.merge_lora_tools",  # 修正路径
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_path in module_list:
    try:
        imported_module = importlib.import_module(f".{module_path}", __name__)
        NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
    except ImportError as e:
        print(f"警告：无法导入模块 {module_path}: {e}")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

