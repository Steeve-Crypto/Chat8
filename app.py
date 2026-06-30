import streamlit as st
from openai import OpenAI
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Chat 8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for nicer look
st.markdown("""
<style>
    .stChatMessage { padding: 1rem 1.5rem; border-radius: 1rem; margin-bottom: 1rem; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #f0f2f6; }
    .stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #e8f4fd; }
    .sidebar .stSelectbox, .sidebar .stTextInput, .sidebar .stSlider { margin-bottom: 0.75rem; }
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #666; font-size: 1.1rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============== PROVIDER PRESETS ==============
PROVIDERS = {
    "xAI Grok (Recommended)": {
        "base_url": "https://api.x.ai/v1",
        "models": ["grok-4.3", "grok-build-0.1", "grok-4.20-reasoning", "grok-4.20-non-reasoning"],
        "default_model": "grok-4.3",
        "description": "Frontier reasoning model by xAI. Excellent at coding, math, and fun personality."
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "o4-mini", "o3"],
        "default_model": "gpt-4o",
        "description": "Industry-leading models from OpenAI."
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "default_model": "llama-3.3-70b-versatile",
        "description": "Extremely fast inference. Great for real-time apps."
    },
    "Together AI": {
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3"],
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "description": "Open-source models hosted on Together.ai."
    },
    "Custom / Local (OpenAI-compatible)": {
        "base_url": "http://localhost:11434/v1",  # Ollama example
        "models": ["llama3.2", "phi4", "qwen2.5", "custom-model-name"],
        "default_model": "llama3.2",
        "description": "Use with Ollama, vLLM, LM Studio, text-generation-webui, etc."
    }
}

# ============== SESSION STATE INITIALIZATION ==============
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_provider" not in st.session_state:
    st.session_state.current_provider = "xAI Grok (Recommended)"

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful, witty, and maximally truthful AI assistant. Answer concisely but thoroughly. Use markdown when helpful."

if "chat_params" not in st.session_state:
    st.session_state.chat_params = {
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.95,
        "stream": True
    }

# ============== SIDEBAR ==============
with st.sidebar:
    st.header("⚙️ LLM Configuration")
    
    # Provider selection
    provider_name = st.selectbox(
        "Provider",
        list(PROVIDERS.keys()),
        index=list(PROVIDERS.keys()).index(st.session_state.current_provider),
        help="Choose your LLM provider. xAI Grok is pre-selected as it's what powers me!"
    )
    st.session_state.current_provider = provider_name
    
    provider_config = PROVIDERS[provider_name]
    
    # API Key
    api_key = st.text_input(
        "API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-... or xai-...",
        help="Your API key is only stored in this browser session. Never shared."
    )
    st.session_state.api_key = api_key
    
    # Base URL (editable for custom)
    base_url = st.text_input(
        "Base URL",
        value=provider_config["base_url"],
        help="API endpoint. Change for custom/local servers."
    )
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        model_options = provider_config["models"] + ["Custom model..."]
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=0 if provider_config["default_model"] in provider_config["models"] else 0,
            help="Select a model or choose 'Custom model...' to type your own."
        )
    with col2:
        if selected_model == "Custom model...":
            model = st.text_input("Custom Model Name", value=provider_config["default_model"], label_visibility="collapsed")
        else:
            model = selected_model
    
    st.caption(provider_config["description"])
    
    # Advanced Parameters
    with st.expander("🔧 Generation Parameters", expanded=False):
        temperature = st.slider(
            "Temperature", 0.0, 2.0, 
            st.session_state.chat_params["temperature"], 0.05,
            help="Higher = more creative/random. Lower = more deterministic."
        )
        max_tokens = st.slider(
            "Max Tokens", 128, 8192, 
            st.session_state.chat_params["max_tokens"], 128,
            help="Maximum length of the response."
        )
        top_p = st.slider(
            "Top P (nucleus)", 0.0, 1.0, 
            st.session_state.chat_params["top_p"], 0.05,
            help="Controls diversity via nucleus sampling."
        )
        enable_stream = st.toggle(
            "Stream responses", 
            value=st.session_state.chat_params["stream"],
            help="Show tokens as they are generated (recommended)."
        )
        
        # Update session
        st.session_state.chat_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": enable_stream
        }
    
    # System Prompt
    with st.expander("📜 System Prompt", expanded=True):
        new_system = st.text_area(
            "System instructions",
            value=st.session_state.system_prompt,
            height=120,
            help="Defines the personality and behavior of the AI."
        )
        if new_system != st.session_state.system_prompt:
            st.session_state.system_prompt = new_system
            st.info("System prompt updated. Start a new chat to apply it fully.")
    
    st.divider()
    
    # Action buttons
    colA, colB = st.columns(2)
    with colA:
        if st.button("🆕 New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()
    with colB:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Export chat
    if st.session_state.messages:
        if st.button("📥 Export Chat (JSON)", use_container_width=True):
            chat_data = {
                "exported_at": datetime.now().isoformat(),
                "provider": provider_name,
                "model": model,
                "system_prompt": st.session_state.system_prompt,
                "messages": st.session_state.messages
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(chat_data, indent=2),
                file_name=f"llm_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    st.divider()
    st.caption("💡 Tip: For local models use Ollama → `http://localhost:11434/v1`")
    st.caption("Made with Streamlit • Works with any OpenAI-compatible API")

# ============== MAIN CHAT AREA ==============
st.markdown('<h1 class="main-header">Chat 8</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Chat with any LLM • Powered by Streamlit + OpenAI SDK</p>', unsafe_allow_html=True)

# Show current config
if api_key:
    st.success(f"Connected to **{provider_name}** • Model: `{model}` • Streaming: {'On' if enable_stream else 'Off'}", icon="🔌")
else:
    st.warning("⚠️ Please enter your API key in the sidebar to start chatting.", icon="🔑")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "system":
        continue  # Don't show system message in UI (or show as small note)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here... (Press Enter to send)", disabled=not api_key):
    if not api_key:
        st.error("Please add your API key in the sidebar first!")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare messages for API (include current system prompt)
    api_messages = []
    if st.session_state.system_prompt.strip():
        api_messages.append({"role": "system", "content": st.session_state.system_prompt})
    
    # Add conversation history (skip any old system messages to avoid duplicates)
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            api_messages.append(msg)
    
    # Create OpenAI client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        st.error(f"Failed to initialize client: {str(e)}")
        st.stop()
    
    # Assistant response placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Call the API
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=st.session_state.chat_params["temperature"],
                max_tokens=st.session_state.chat_params["max_tokens"],
                top_p=st.session_state.chat_params["top_p"],
                stream=st.session_state.chat_params["stream"]
            )
            
            if st.session_state.chat_params["stream"]:
                # Streaming mode
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")  # blinking cursor
                message_placeholder.markdown(full_response)
            else:
                # Non-streaming
                full_response = stream.choices[0].message.content
                message_placeholder.markdown(full_response)
            
            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"❌ Error calling {provider_name} API: {str(e)}"
            message_placeholder.error(error_msg)
            # Remove the user message if API call failed
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
            st.stop()

# Footer info
st.divider()
st.caption(
    "This app is a universal wrapper for any LLM with an OpenAI-compatible API. "
    "It runs 100% locally on your machine. Your API keys never leave your browser session."
)