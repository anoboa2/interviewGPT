import streamlit as st
from local_functions import InvokeInterviewQuestion, InitializeVectorStore

# Configure which OpenAI model to use
if "openai_model" not in st.session_state:
  st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize session state
if "messages" not in st.session_state:
  st.session_state.messages = []

# Initialize vector store
retriever = InitializeVectorStore()

### Streamlit UI ###
placeholder = st.empty()
with placeholder.container():
  st.info("""
    You are chatting with an LLM trained on my skills, experiences, and interests. You'll learn just as much here as you would from an interview! \n
    Our conversation will persist if you close the chat window. However, if you refresh or leave the page the chat history will be erased.
    """,
    icon="💡"
  )
if st.session_state.messages:
  placeholder.empty()
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("Let's dive in!"):
    
  # Add user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})

  # Display user message in chat message container
  with st.chat_message("user"):
    st.markdown(prompt)
  
  # Display assistant response in chat message container
  with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""

    # Stream assistant responses from GPT API call
    for response in InvokeInterviewQuestion(prompt, retriever):
      full_response += response
      message_placeholder.markdown(full_response)

  st.session_state.messages.append({"role": "assistant", "content": full_response})