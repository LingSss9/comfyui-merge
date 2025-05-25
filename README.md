
English | [中文](README_zh.md)      

# ComfyUI Merge Plugin

An advanced plugin for ComfyUI, designed for merging up to 4 LoRA models with balanced features. This plugin reproduces the behavior of WebUI's SuperMerger, ensuring that the output is independent of model order and avoids overfitting.     


## Features
- Supports merging 2 to 4 LoRA models.
- No base model required.
- Merging behavior is independent of model order.
- Enhanced logic to avoid dominant weight bias.
- Built-in filtering and folder UI component.

## Installation
1. Copy the folder `comfyui-merge` into your ComfyUI/custom_nodes/ directory.
2. Restart ComfyUI.

## Node List
- **MergeLoRAsKohyaSSLike**: Main merge logic node with order-independent behavior.
- **LoraFolderFilter**: Web UI component for folder filtering.

## Usage
1. Drag and drop the node into your ComfyUI workflow.
2. Connect up to four LoRA models with specified ratios.
3. Adjust sliders to fine-tune merge weights.
4. Output will be a single LoRA-style model node usable in downstream workflows.

Workflow-examples:      
![Plugin Preview](https://github.com/user-attachments/assets/6d0a02e6-a92e-40b3-9f1d-156fde787ff4)
