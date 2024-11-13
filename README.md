# Project Title: **Research Paper Title Generation with Llama 1B Fine-Tuning**

This project fine-tunes the Llama 1B model to generate titles for research papers based on their abstracts, using a custom dataset of title-abstract pairs. By leveraging techniques like LoRA (Low-Rank Adaptation) quantization, the model is optimized for efficient training and inference.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model and Techniques](#model-and-techniques)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)
8. [Acknowledgments](#acknowledgments)

---

### Project Overview

The aim of this project is to generate accurate and relevant research paper titles by training a language model to understand the abstract and context of each paper. By employing Llama 1B as the base model, this fine-tuning process demonstrates how pre-trained language models can be adapted for specialized NLP tasks such as title generation.

### Dataset

- **Dataset**: A dataset of research papers containing two columns: `title` and `abstract`.
- **Data Preprocessing**: The dataset is preprocessed to ensure high-quality input, and tokenization is performed using the Llama tokenizer. 

### Model and Techniques

1. **Model**: Llama 1B model by Meta AI, chosen for its balance between performance and efficiency.
2. **Quantization**: LoRA quantization is applied to make fine-tuning feasible on smaller hardware setups by reducing memory usage.
3. **Training**: The model is fine-tuned using Hugging Face's Trainer API, which simplifies the training loop, handling evaluation metrics and model checkpoints.
4. **Evaluation**: The model is evaluated based on title generation accuracy and loss metrics, which help measure its ability to generalize to unseen abstracts.

### Installation

To replicate this project, set up the environment by installing the necessary libraries:

```bash
# Clone the repository
git clone https://github.com/your_username/llama-title-generator.git
cd llama-title-generator

# Install dependencies
pip install -r requirements.txt
```

Requirements can be generated from your environment using:
```bash
pip freeze > requirements.txt
```

### Usage

1. **Data Preparation**:
   - Ensure your dataset is structured with `title` and `abstract` columns.
   - Save the dataset as `data/titles_abstracts.csv`.

2. **Training the Model**:
   - Use the Jupyter notebook to load and preprocess the dataset, initialize the model, and start fine-tuning:
     ```bash
     jupyter notebook llm_llama_1b_finetune_generate_title.ipynb
     ```

3. **Evaluating the Model**:
   - After training, evaluate the model on a validation dataset to verify its performance.

4. **Inference**:
   - Use the model to generate titles from new abstracts by running the inference section of the notebook.

### Project Structure

```
.
├── __pycache__/                                         # Python bytecode cache directory
├── llama_1B_lora_finetuned/                            # Directory containing the fine-tuned Llama model using LoRA
├── Hugginface_prasun.py                                # Script for Hugging Face model integration and utilities
├── llm_llama_1b_finetune_generate_title.ipynb          # Jupyter notebook for fine-tuning Llama and generating titles
├── requirements.txt                                     # Project dependencies and their versions
└── title_maker.py                                      # Core script for title generation functionality
```

### Results

The fine-tuned model shows promising results in generating titles that are contextually relevant to the provided abstracts. Further evaluation metrics are saved in the notebook.

### Acknowledgments

- [Meta AI's Llama model](https://www.llama.com/) and its [Hugging Face page for Llama 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) for providing the foundational pre-trained language model used in this project.
- Hugging Face for their Trainer API, which simplifies model training and deployment.
- LoRA quantization technique for memory-efficient training.
