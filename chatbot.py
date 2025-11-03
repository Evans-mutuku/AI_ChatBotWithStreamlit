import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Evans - ChatBot")

def load_text_generator():
    text_generator = pipeline("text-generation", model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant for software engineering. "
    "Answer concisely and give short code examples when useful. "
    "If unsure, say you are unsure.\n\n"
)

# Build the convo prompt
def build_conversation_prompt(chat_history, user_question):
    formated_conversation = []
    for previous_question, previous_answer in chat_history:
        formated_conversation.append(f"Question: {previous_question}\nAnswer: {previous_answer}\n")
    
    formated_conversation.append(f"Question: {user_question}\nAnswer:")
    return SYTEM_INSTRUCTIONS + "\n".join(formated_conversation)

st.title("Evans - ChatBot UI")
st.caption("Ask me anything about software Engineering")


# Sidebar for config
with st.sidebar:
    st.header("Model Controls/Config")
    max_new_tokens = st.slider("Max New Tokens", min_value=10, max_value=100, value=50, step=10)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history =[]

# Display chat history
for user_message, ai_reply in st.session_state.chat_history:
    st.chat_message("user").markdown(user_message)
    st.chat_message("assistant").markdown(ai_reply)

# User input
user_input = st.chat_input("As JsMummie Anything....")
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
            pad_token_id=text_generator.tokenizer.eos_token_id,
            eos_token_id=text_generator.tokenizer.eos_token_id,
        )[0]['generated_text']

        # Extracting the model answer from the generated text
        generated_answer = generation_output.split("Answer: ")[-1].strip()
        if "Question:" in generated_answer:
            generated_answer = generated_answer.split("Question:")[0].strip()

    # Displaying and storing chatbot response
    st.chat_message("assistant").markdown(generated_answer)
    st.session_state.chat_history.append((user_input, generated_answer))