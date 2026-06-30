# LLM Wrapper - Streamlit Chat App

A beautiful, flexible Streamlit application that wraps around any **OpenAI-compatible LLM API**.

## Features
- 🔌 Connect to **xAI Grok**, OpenAI, Groq, Together.ai, Ollama (local), vLLM, or any OpenAI-compatible endpoint
- 💬 Modern chat interface with streaming responses
- ⚙️ Configurable: model, temperature, max tokens, system prompt, top_p, etc.
- 🧠 Conversation history with session state
- 🎨 Clean UI with sidebar settings, clear chat, export chat
- 🔒 API key handled securely (password input, session only)
- 📋 Presets for popular providers + fully custom

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get API Keys
- **xAI Grok**: https://console.x.ai/ (recommended - I'm powered by Grok!)
- **OpenAI**: https://platform.openai.com/api-keys
- **Groq**: https://console.groq.com/keys (very fast & cheap)
- **Together.ai**, Fireworks, etc. for open models

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use
1. Open the **Sidebar** (click the arrow if hidden)
2. Select your **Provider** (xAI Grok is default)
3. Paste your **API Key**
4. Choose or type the **Model** name
5. (Optional) Set System Prompt, Temperature, Max Tokens
6. Start chatting in the main chat box!
7. Use **Clear Chat** button to reset conversation

## Supported Providers (Examples)

| Provider       | Base URL                              | Example Models                  | Notes                     |
|----------------|---------------------------------------|---------------------------------|---------------------------|
| xAI Grok       | `https://api.x.ai/v1`                 | `grok-4.3`, `grok-build-0.1`   | Best reasoning & fun personality |
| OpenAI         | `https://api.openai.com/v1`           | `gpt-4o`, `gpt-4o-mini`        | Industry standard        |
| Groq           | `https://api.groq.com/openai/v1`      | `llama-3.3-70b-versatile`, `mixtral-8x7b-32768` | Blazing fast inference |
| Custom         | Your endpoint                         | Any model                      | Ollama, vLLM, LM Studio, etc. |

## Tips
- For **local models**: Use Ollama with `http://localhost:11434/v1` as base URL and model like `llama3.2`
- Enable **Streaming** for real-time token-by-token responses (default: on)
- The app remembers your settings in the current browser session
- Export your chat history as JSON or Markdown from the sidebar

## Advanced
You can extend this app easily:
- Add **file uploads** for RAG (use with LangChain or LlamaIndex)
- Add **tools / function calling**
- Add **image upload** for vision models (Grok supports vision)
- Deploy to Streamlit Cloud, Hugging Face Spaces, or your server

Streamlit + OpenAI SDK (compatible layer)

*Note: This is a local app. Never share your API keys. For production, add authentication.*
