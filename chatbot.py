import streamlit as st
from transformers import pipeline

# --- Streamlit page setup ---
st.set_page_config(page_title="GPT-2 Chatbot", page_icon="🤖", layout="centered")

# --- Load the GPT-2 model once and reuse ---
@st.cache_resource(show_spinner=True)
def load_text_generator():
    text_generator = pipeline("text-generation", model="openai-community/gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator


# --- System behavior prompt to guide GPT-2 ---
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for software engineering. "
    "Answer concisely and give short code examples when useful. "
    "If unsure, say you are unsure.\n\n"
)


# --- Build the conversation prompt for GPT-2 ---
def build_conversation_prompt(chat_history, user_question):
    formatted_conversation = []
    for previous_question, previous_answer in chat_history:
        formatted_conversation.append(f"Question: {previous_question}\nAnswer: {previous_answer}\n")

    formatted_conversation.append(f"Question: {user_question}\nAnswer:")
    return SYSTEM_INSTRUCTION + "\n".join(formatted_conversation)


# --- Page header ---
st.title("🤖 GPT-2 Chatbot (Hugging Face)")
st.caption(
    "A simple local chatbot using the `openai-community/gpt2` model. "
    "For better answers, you can swap it with a small instruction-tuned model."
)


# --- Sidebar for configuration options ---
with st.sidebar:
    st.header("⚙️ Model Settings")
    max_new_tokens = st.slider("Maximum new tokens", 20, 300, 120, 10)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.0, 0.5, 0.1)
    top_p = st.slider("Top-p sampling", 0.1, 1.0, 0.9, 0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.15, 0.05)

    if st.button("🧹 Clear chat history"):
        st.session_state.chat_history = []


# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Display chat history ---
for user_message, ai_reply in st.session_state.chat_history:
    st.chat_message("user").markdown(user_message)
    st.chat_message("assistant").markdown(ai_reply)


# --- Input box for new question ---
user_input = st.chat_input("Ask me about software engineering...")
if user_input:
    st.chat_message("user").markdown(user_input)

    with st.spinner("Thinking..."):
        text_generator = load_text_generator()
        prompt_text = build_conversation_prompt(st.session_state.chat_history, user_input)

        generation_output = text_generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=text_generator.tokenizer.eos_token_id,
            eos_token_id=text_generator.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Extract the model's answer from the text
        generated_answer = generation_output.split("Answer:")[-1].strip()
        if "Question:" in generated_answer:
            generated_answer = generated_answer.split("Question:")[0].strip()

    # --- Display and store chatbot response ---
    st.chat_message("assistant").markdown(generated_answer)
    st.session_state.chat_history.append((user_input, generated_answer))