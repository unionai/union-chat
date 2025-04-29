from dataclasses import dataclass
from models import get_config

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:")
st.title("Union Serving LLMs")


@dataclass
class ClientInfo:
    client: OpenAI
    model_id: str
    max_tokens: int | None = None

@st.cache_resource
def load_client_infos() -> dict[str, ClientInfo]:
    config = get_config()

    client_infos = {}
    for i, model_config in enumerate(config.models):
        endpoint = model_config.get_endpoint(i)
        client_infos[model_config.display_name] = ClientInfo(
            client=OpenAI(base_url=f"{endpoint}/v1", api_key="ABC"),
            model_id=model_config.model_id,
            max_tokens=model_config.max_tokens,
        )

    return client_infos


client_infos = load_client_infos()


def clear_chat():
    st.session_state["messages"] = []


if "messages" not in st.session_state:
    clear_chat()

with st.sidebar:
    select_box = st.selectbox(
        "Model Selection",
        list(client_infos),
        key="model_selection",
        on_change=clear_chat,
    )

    clear_button = st.button(label="Clear Chat", icon="🗑️")

    if clear_button:
        clear_chat()


st.markdown(f"#### Chatting with {select_box}")

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

client_info = client_infos[select_box]

if prompt := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        stream = client_info.client.chat.completions.create(
            model=client_info.model_id,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            max_tokens=client_info.max_tokens,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
