// custom_nodes/comfyui-merge/web/lora_folder_filter.js
import { ComfyApp, app } from "../../../scripts/app.js";

app.registerExtension({
  name: "Comfy.LoraFolderFilter",

  nodeCreated(node, app) {
    if (node.comfyClass !== "OnlyLoadLoRAsModel") return;

    console.log("[LoRA folder filter] extension loaded for node:", node.comfyClass);

    // 找到 lora_name 和 category_filter widgets
    const loraNamesWidget = node.widgets.find(w => w.name === 'lora_name');
    const categoryFilterWidget = node.widgets.find(w => w.name === 'category_filter');

    if (!loraNamesWidget || !categoryFilterWidget) {
      console.log("[LoRA folder filter] 未找到必需的 widgets");
      return;
    }

    // 保存原始列表的拷贝
    const fullLoraList = [...loraNamesWidget.options.values];

    // 初始化时确保当前过滤状态生效
    updateLoraList();

    // 监听分类过滤器变化
    const oldCallback = categoryFilterWidget.callback || (() => {});
    categoryFilterWidget.callback = (value, ...args) => {
      oldCallback(value, ...args);
      updateLoraList();
    };

    function updateLoraList() {
      const filter = categoryFilterWidget.value;
    
      let filteredList;
      if (filter === 'All') {
        filteredList = fullLoraList;
      } else {
        const normalizedFilter = filter.replace(/[.*+?^${}()|[$$\$$]/g, '\\$&');
        const pattern = new RegExp(`^${normalizedFilter}[\\\\/]`);
        filteredList = fullLoraList.filter(x => pattern.test(x));
      }
    
      // 更新LoRA名称列表
      loraNamesWidget.options.values = filteredList;
    
      // 默认选择第一个匹配项（如果存在）
      if (filteredList.length > 0) {
        loraNamesWidget.value = filteredList[0];
      } else {
        loraNamesWidget.value = '';
      }
    
      // 触发UI更新
      if (loraNamesWidget.widget?.inputEl) {
        loraNamesWidget.widget.inputEl.dispatchEvent(new Event('input'));
      }
    }

    console.log("[LoRA folder filter] widgets 过滤器已设置完成");
  }
});