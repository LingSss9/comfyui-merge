// custom_nodes/comfyui-merge/web/lora_folder_filter.js
import { ComfyApp, app } from "../../../scripts/app.js";

app.registerExtension({
  name: "Comfy.LoraFolderFilter",

  nodeCreated(node, app) {
    if (node.comfyClass !== "OnlyLoadLoRAsModel") return;

    console.log("[LoRA folder filter] extension loaded for node:", node.comfyClass);

    // 找到 lora_name 和 category_filter widgets
    const lora_names_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'lora_name')];
    const category_filter_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'category_filter')];

    if (!lora_names_widget || !category_filter_widget) {
      console.log("[LoRA folder filter] 未找到必需的 widgets");
      return;
    }

    // 备份完整的 lora 列表
    var full_lora_list = lora_names_widget.options.values;

    // 重新定义 lora_names_widget.options.values 的 getter/setter
    Object.defineProperty(lora_names_widget.options, "values", {
      set: (x) => {
        full_lora_list = x;
      },
      get: () => {
        if (category_filter_widget.value == 'All')
          return full_lora_list;

        // 过滤出选定文件夹中的 lora 文件
        // 支持 Windows (\) 和 Unix (/) 路径分隔符
        let prefix = category_filter_widget.value;
        let filtered_list = full_lora_list.filter(x => 
          x.startsWith(prefix + '\\') || x.startsWith(prefix + '/')
        );
        return filtered_list;
      }
    });

    console.log("[LoRA folder filter] widgets 过滤器已设置完成");
  }
});
