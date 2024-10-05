[English](README.en.md)

# Qwen2.5 Sex

## 简介

Qwen2.5 Sex 是基于 Qwen2.5-1.5B-Instruct 微调的模型，主要在大量成人文学作品及敏感数据集上进行了训练。由于数据集主要为中文，模型在处理中文文本时效果尤佳。

> **警告**：本模型仅供研究和测试使用。用户需遵循当地法律法规，并自行承担使用模型的责任。

**此模型已在 Hugging Face 上开源，[点击此处](https://huggingface.co/ystemsrx/Qwen2.5-Sex)查看和使用。**

## 使用方法

1. 下载所有文件，并运行以下命令安装依赖：
   ```
   pip install -r requirements.txt
   ```

2. 启动图形化推理界面：
   ```
   python run_webUI.py
   ```

   启动后会显示类似以下的界面：

   ![image](https://github.com/user-attachments/assets/6efe7ba0-4498-40d1-9048-44e14e899e01)

   > **注意**：建议保持模型的参数（如 TOP_P、TOP_K、Temperature 等）为默认值或适当调高，且系统提示词建议保持为空。

## 数据集

Qwen2.5 Sex 模型在大量成人文学和敏感数据集上进行了微调，涉及道德、法律、色情及暴力等主题。由于微调数据集以中文为主，模型在处理中文文本时表现更佳。如欲进一步了解数据集信息，可通过以下链接获取：

- [Bad Data](https://huggingface.co/datasets/ystemsrx/Bad_Data_Alpaca)
- [Toxic-All](https://huggingface.co/datasets/ystemsrx/Toxic-All)
- [Erotic Literature Collection](https://huggingface.co/datasets/ystemsrx/Erotic_Literature_Collection)

## 免责声明

该模型在包含潜在敏感或争议内容的数据集上进行了微调，包括暴力、色情、违法行为和不道德行为。用户在使用该模型时应充分意识到这些内容，建议在受控环境下应用此模型。

Qwen2.5 Sex 的创建者不认可或支持任何非法或不道德的使用行为。该模型仅供研究用途，用户应确保其使用符合所有适用的法律和道德规范。
