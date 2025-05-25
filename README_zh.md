
[English](README.md) | 中文

# ComfyUI模型融合插件    
本插件是为 ComfyUI 开发的LoRA模型融合插件，可支持最多4个LoRA模型融合，逻辑借鉴WebUI中的SuperMerger插件，重点优化合并顺序影响与权重分配失衡的问题。   
### 工作流预览：    
![插件预览](https://github.com/user-attachments/assets/6d0a02e6-a92e-40b3-9f1d-156fde787ff4 )    

## 插件特色
- 支持2～4个LoRA模型融合
- 不需要底模（base model）
- 合并顺序不会影响输出结果
- 自动平衡模型之间的影响力，避免一方权重过大
- 内含用于文件夹筛选的Web界面模块

## 安装方法
1. 将 `comfyui-merge` 文件夹复制到你的 ComfyUI/custom_nodes/ 目录下。
2. 重启 ComfyUI。

## 节点说明
- **MergeLoRAsKohyaSSLike**：主融合逻辑节点，实现顺序无关的合并效果。
- **LoraFolderFilter**：提供文件夹过滤功能的Web界面控件。

## 使用方法
1. 将节点拖入工作流。
2. 输入最多4个LoRA模型与对应权重。
3. 调整权重参数以控制合并比例。
4. 输出为可用于其他流程的LoRA模型节点。
