English | 插件说明（English version）

ComfyUI Merge Plugin

An advanced plugin for ComfyUI, designed for merging up to 4 LoRA models with balanced features. This plugin reproduces the behavior of WebUI's SuperMerger, ensuring that the output is independent of model order and avoids overfitting.

Features
Supports merging 2 to 4 LoRA models.

No base model required.

Merging behavior is independent of model order.

Enhanced logic to avoid dominant weight bias.

Built-in filtering and folder UI component.

Installation
Copy the folder comfyui-merge into your ComfyUI/custom_nodes/ directory.

Restart ComfyUI.

Node List
MergeLoRAsKohyaSSLike: Main merge logic node with order-independent behavior.

LoraFolderFilter: Web UI component for folder filtering.

Usage
Drag and drop the node into your ComfyUI workflow.

Connect up to four LoRA models with specified ratios.

Adjust sliders to fine-tune merge weights.

Output will be a single LoRA-style model node usable in downstream workflows.

中文说明 | 中文版使用文档

ComfyUI Merge 插件

本插件是为 ComfyUI 开发的高级LoRA模型融合插件，可支持最多4个LoRA模型融合，逻辑借鉴WebUI中的SuperMerger插件，重点优化合并顺序影响与权重分配失衡的问题。

插件特色
支持2～4个LoRA模型融合

不需要底模（base model）

合并顺序不会影响输出结果

自动平衡模型之间的影响力，避免一方权重过大

内含用于文件夹筛选的Web界面模块

安装方法
将 comfyui-merge 文件夹复制到你的 ComfyUI/custom_nodes/ 目录下。

重启 ComfyUI。

节点说明
MergeLoRAsKohyaSSLike：主融合逻辑节点，实现顺序无关的合并效果。

LoraFolderFilter：提供文件夹过滤功能的Web界面控件。

使用方法
将节点拖入工作流。

输入最多4个LoRA模型与对应权重。

调整权重参数以控制合并比例。

输出为可用于其他流程的LoRA模型节点。
