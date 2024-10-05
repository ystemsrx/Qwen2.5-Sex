import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import set_seed
from threading import Thread
import random
import os
import gradio as gr

# 默认参数
DEFAULT_TOP_P = 0.9        # Top-p (nucleus sampling) 范围在0到1之间
DEFAULT_TOP_K = 80         # Top-k 采样的K值
DEFAULT_TEMPERATURE = 0.8  # 温度参数，控制生成文本的随机性
DEFAULT_MAX_NEW_TOKENS = 512  # 生成的最大新令牌数
DEFAULT_SYSTEM_MESSAGE = ""  # 默认系统消息

# 检查是否有可用的 GPU，默认使用 GPU，如果不可用则使用 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    cpu_only = False
    print("检测到 GPU，使用 GPU 进行推理。")
else:
    DEVICE = "cpu"
    cpu_only = True
    print("未检测到 GPU，使用 CPU 进行推理。")

DEFAULT_CKPT_PATH = os.path.dirname(os.path.abspath(__file__))

def _load_model_tokenizer(checkpoint_path, cpu_only):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, resume_download=True)

    device_map = "cpu" if cpu_only else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if not cpu_only else torch.float32,  # 使用更低的精度以节省显存
        device_map=device_map,
        resume_download=True,
    ).eval()

    # 如果使用 GPU，确保模型使用半精度以节省显存（如果模型支持）
    if not cpu_only and torch.cuda.is_available():
        try:
            model.half()
            print("模型已切换为半精度（float16）。")
        except:
            print("无法切换模型为半精度，继续使用默认精度。")

    model.generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS  # 设置生成的最大新令牌数

    return model, tokenizer

def _chat_stream(model, tokenizer, query, history, system_message, top_p, top_k, temperature, max_new_tokens):
    conversation = [
        {'role': 'system', 'content': system_message},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    
    # 准备输入
    try:
        # 尝试使用 apply_chat_template 方法
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,               # 确保返回的是字符串
            add_generation_prompt=True
        )
    except AttributeError:
        # 如果没有 apply_chat_template 方法，使用标准方法构建对话
        print("[WARNING] `apply_chat_template` 方法不存在，使用标准对话格式。")
        text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])
        text += "\nAssistant:"
    
    # 确保 text 是字符串
    if not isinstance(text, str):
        raise ValueError("apply_chat_template 应返回字符串类型的文本。")
    
    # Tokenize 输入
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    
    # 生成参数
    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        do_sample=True,  # 确保使用采样方法
        pad_token_id=tokenizer.eos_token_id,  # 避免警告
        streamer=streamer,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield new_text
    return generated_text

def initialize_model():
    checkpoint_path = DEFAULT_CKPT_PATH
    seed = random.randint(0, 2**32 - 1)  # 随机生成一个种子
    set_seed(seed)  # 设置随机种子

    model, tokenizer = _load_model_tokenizer(checkpoint_path, cpu_only)

    return model, tokenizer

# 初始化模型和分词器
model, tokenizer = initialize_model()

def chat_interface(user_input, history, system_message, top_p, top_k, temperature, max_new_tokens):
    if user_input.strip() == "":
        yield history, history, system_message, ""
        return
    
    # 创建一个新的助手回复条目，初始为空
    history.append((user_input, ""))
    yield history, history, system_message, ""  # 更新 Chatbot 组件和状态

    # 获取模型生成的回复
    generator = _chat_stream(model, tokenizer, user_input, history[:-1], system_message, top_p, top_k, temperature, max_new_tokens)
    assistant_reply = ""
    for new_text in generator:
        assistant_reply += new_text
        # 更新最后一条助手回复
        updated_history = history.copy()
        updated_history[-1] = (user_input, assistant_reply)
        yield updated_history, updated_history, system_message, ""  # 更新 Chatbot 组件和状态

def clear_history():
    return [], [], DEFAULT_SYSTEM_MESSAGE, ""

# Gradio 接口
with gr.Blocks() as demo:
    # CSS
    gr.HTML("""
    <style>
        #chat-container {
            height: 500px;
            overflow-y: auto;
        }
        .settings-column {
            padding-left: 20px;
            border-left: 1px solid #ddd;
        }
        .send-button {
            margin-top: 10px;
            width: 100%;
        }
    </style>
    """)

    gr.Markdown("# Qwen2.5 Boundless")

    with gr.Row():
        # 左侧栏：聊天记录和用户输入
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chat-container")
            user_input = gr.Textbox(
                show_label=False, 
                placeholder="输入你的问题...", 
                lines=2,
                interactive=True
            )
            send_btn = gr.Button("发送", elem_classes=["send-button"])
        
        # 右侧栏：清空历史按钮、系统消息输入框和生成参数滑块
        with gr.Column(scale=1, elem_classes=["settings-column"]):
            gr.Markdown("### 设置")
            clear_btn = gr.Button("清空历史")
            gr.Markdown("#### 系统消息")
            system_message = gr.Textbox(
                label="系统消息",
                value=DEFAULT_SYSTEM_MESSAGE,
                placeholder="输入系统消息...",
                lines=2
            )
            gr.Markdown("#### 生成参数")
            top_p_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=DEFAULT_TOP_P, step=0.05,
                label="Top-p (nucleus sampling)"
            )
            top_k_slider = gr.Slider(
                minimum=0, maximum=100, value=DEFAULT_TOP_K, step=1,
                label="Top-k"
            )
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.5, value=DEFAULT_TEMPERATURE, step=0.05,
                label="Temperature"
            )
            max_new_tokens_slider = gr.Slider(
                minimum=50, maximum=2048, value=DEFAULT_MAX_NEW_TOKENS, step=2,
                label="Max New Tokens"
            )

    # 状态管理
    state = gr.State([])

    # 绑定事件
    # 回车chat_interface
    user_input.submit(
        chat_interface, 
        inputs=[user_input, state, system_message, top_p_slider, top_k_slider, temperature_slider, max_new_tokens_slider], 
        outputs=[chatbot, state, system_message, user_input],
        queue=True
    )
    # 发送chat_interface
    send_btn.click(
        chat_interface, 
        inputs=[user_input, state, system_message, top_p_slider, top_k_slider, temperature_slider, max_new_tokens_slider], 
        outputs=[chatbot, state, system_message, user_input],
        queue=True
    )
    clear_btn.click(
        clear_history, 
        inputs=None, 
        outputs=[chatbot, state, system_message, user_input],
        queue=True
    )

    # JS
    gr.HTML("""
    <script>
        function scrollChat() {
            const chatContainer = document.getElementById('chat-container');
            if(chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        const observer = new MutationObserver(scrollChat);
        const chatContainer = document.getElementById('chat-container');
        if(chatContainer) {
            observer.observe(chatContainer, { childList: true, subtree: true });
        }
    </script>
    """)

demo.launch()
