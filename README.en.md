[中文](README.md)

# Qwen2.5 Sex

## Introduction

Qwen2.5 Sex is a model fine-tuned based on Qwen2.5-1.5B-Instruct, primarily trained on a large amount of adult literature and sensitive datasets. Since the dataset is mainly in Chinese, the model performs particularly well when processing Chinese text.

> **Warning**: This model is intended for research and testing purposes only. Users must comply with local laws and regulations and assume responsibility for their use of the model.

This model is open-sourced on Hugging Face. [Click here](https://huggingface.co/ystemsrx/Qwen2.5 Sex) to view and use it.

## Usage

1. Download all files and run the following command to install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Launch the graphical inference interface:
   ```
   python run_webUI.py
   ```

   After launching, a similar interface will be displayed:

   ![image](https://github.com/user-attachments/assets/6efe7ba0-4498-40d1-9048-44e14e899e01)

   > **Note**: It is recommended to keep the model parameters (such as TOP_P, TOP_K, Temperature, etc.) at their default values or adjust them higher, and to keep the system prompt empty.

## Dataset

The Qwen2.5 Sex model has been fine-tuned on a large amount of adult literature and sensitive datasets, covering topics such as morality, law, sexuality, and violence. Since the fine-tuned dataset is mainly in Chinese, the model performs better when processing Chinese text. For more information about the dataset, you can access the following links:

- [Bad Data](https://huggingface.co/datasets/ystemsrx/Bad_Data_Alpaca)
- [Toxic-All](https://huggingface.co/datasets/ystemsrx/Toxic-All)
- [Erotic Literature Collection](https://huggingface.co/datasets/ystemsrx/Erotic_Literature_Collection)

## Disclaimer

This model has been fine-tuned on datasets containing potentially sensitive or controversial content, including violence, pornography, illegal activities, and unethical behavior. Users should be fully aware of these contents when using this model, and it is recommended to apply it in a controlled environment.

The creators of Qwen2.5 Sex do not endorse or support any illegal or unethical use. This model is intended for research purposes only, and users must ensure that their usage complies with all applicable laws and ethical guidelines.
