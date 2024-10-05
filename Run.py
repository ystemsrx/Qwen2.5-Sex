from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 可调节参数
TOP_P = 0.9        # Top-p (nucleus sampling) 范围在0到1之间
TOP_K = 70         # Top-k 采样的K值
TEMPERATURE = 0.8  # 温度参数，控制生成文本的随机性

device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    current_directory,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(current_directory)

messages = [
    {"role": "system", "content": ""}
]

print("欢迎使用聊天机器人！您可以开始对话。")
print(f"当前设置 - Top-p: {TOP_P}, Top-k: {TOP_K}, 温度: {TEMPERATURE}")

while True:
    # 获取用户输入
    user_input = input("User: ").strip()

    # 将用户输入添加到对话中
    messages.append({"role": "user", "content": user_input})

    # 准备输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成响应
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        top_p=TOP_P,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        do_sample=True,  # 确保使用采样方法
        pad_token_id=tokenizer.eos_token_id  # 避免警告
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码并打印响应
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Assistant: {response}")

    # 将生成的响应添加到对话中
    messages.append({"role": "assistant", "content": response})
