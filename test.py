import os

# 1. 路径设置
save_path = r"D:\huggingface_cache"
os.environ['HF_HOME'] = save_path

# 2. 替代方案：从 ModelScope 下载（通常比 HF 镜像快得多）
# 如果 7B 还是觉得大，可以把模型 ID 换成 "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

local_dir = snapshot_download(model_id, cache_dir=save_path)
print(f"下载完成，本地路径: {local_dir}")

# 3. 最新量化配置 (解决你的弃用警告)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 4. 加载模型
print("正在加载模型到显存...")
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    quantization_config=quantization_config, # 使用新配置
    torch_dtype=torch.float16 # 解决 torch_dtype 警告
)

# 5. 推理逻辑
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "什么是孙权"
system_prompt = f"<｜begin of sentence｜>{prompt}<｜Assistant｜>"

print("--- 开始生成 ---")
# 4060 建议设置适当的 temperature
response = pipe(system_prompt, max_new_tokens=2000, do_sample=True, temperature=0.6)
print(response[0]['generated_text'])