import os
import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================
# MODEL CONFIG
# ================================

# 🔧 ADJUST THIS to your model name on Hugging Face
MODEL_ID = "nicpac/FYG"  # example: "nicpac/FYG"

SYSTEM_PROMPT = (
    "You are a videogame assistant that recommends games based on the user's tastes. "
    "Answer each question independently, as if it were a new query. "
    "Be clear, concise, and avoid repeating the same idea or sentence."
)

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device}")
print(f"Loading language model from: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
model.to(device)
model.eval()

# Ensure pad_token_id
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# ================================
# GENERATION FUNCTION (NO RAG, NO HISTORY)
# ================================

def generate_reply(user_message: str, history: list) -> str:
    """
    history is ignored on purpose: each answer is generated independently.
    """
    user_message = user_message.strip()
    if not user_message:
        return "Please type what kind of game you are looking for."

    # 1) Build messages for the chat template (NO history)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # 2) Apply model chat template (TinyLlama Chat)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 3) Tokenize with truncation to avoid huge prompts
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)
    input_ids = inputs["input_ids"]

    # 4) Generate answer with sampling + repetition control
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=6,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 5) Decode only NEW tokens (after the prompt)
    generated_ids = output_ids[0, input_ids.shape[-1]:]
    reply = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not reply:
        reply = "I'm not sure this time, could you please rephrase your question?"

    return reply


# ================================
# GRADIO UI
# ================================

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎮 FindYourGame – Video Game Recommender

        Model: TinyLlama fine-tuned on Steam reviews.

        Each question is answered independently (no conversation memory).

        Ask in English, for example:
        - *"What is the best football game for PC?"*
        - *"I just finished Hollow Knight, what should I play next?"*
        - *"I want a relaxing co-op RPG to play with a friend."*
        """
    )

    chatbot = gr.Chatbot(height=400, label="Chat")
    msg = gr.Textbox(label="Type your question about video games")
    clear = gr.Button("Clear chat")

    def respond(message, chat_history):
        reply = generate_reply(message, chat_history)
        # history is only for UI, NOT used for generation
        chat_history.append((message, reply))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch()
