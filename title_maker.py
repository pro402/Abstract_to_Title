pip install torch transformers peft huggingface_hub

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

# Log in to Hugging Face
login(hf_read_access_key)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\PRASUN MAITY\Desktop\LLM_title_maker\llama_1B_lora_finetuned")

# Load the base LLaMA model from Hugging Face
base_model_path = "meta-llama/Llama-3.2-1B"  # Update to the model's Hugging Face ID
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Load the LoRA fine-tuned model
model = PeftModel.from_pretrained(base_model, r"C:\Users\PRASUN MAITY\Desktop\LLM_title_maker\llama_1B_lora_finetuned")

def generate_title(abstract, temperature, top_k, top_p):
    prompt_template = f"Generate a short title by reading the following abstract:\n\nAbstract: {abstract}\n\nTitle: "
    # Tokenize the prompt input
    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device)
    
    output = model.generate(
        input_ids,
        max_new_tokens=20,                  # Limit the length to a short title
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature,             # Use the temperature from slider
        top_k=top_k,                        # Use the top_k from slider
        top_p=top_p,                        # Use the top_p from slider
        pad_token_id=tokenizer.eos_token_id, # Prevents issues with padding
        attention_mask=torch.ones_like(input_ids)  # Ensure attention mask is used
    )
    title = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return title

# Streamlit UI
st.title("Abstract to Title Generator")

# Text input for abstract
abstract = st.text_area("Enter the abstract:")

# Sliders for parameters with default values
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
top_k = st.slider("Top-k", min_value=1, max_value=100, value=40, step=1)
top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.85, step=0.05)

# Generate title when the button is clicked
if st.button("Generate Title"):
    if abstract.strip():
        title = generate_title(abstract, temperature, top_k, top_p)
        st.subheader("Generated Title")
        st.write(title)  # Only display the generated title
    else:
        st.warning("Please enter an abstract.")
