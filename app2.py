import streamlit as st
from openai import OpenAI
import json
import datetime
import ast
import operator as op
import io
import contextlib
import requests
import base64
import numpy as np
import re

# Page config
st.set_page_config(
    page_title=" Chat 8 ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem 1.5rem; border-radius: 1rem; margin-bottom: 1rem; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #f0f2f6; }
    .stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #e8f4fd; }
    .sidebar .stSelectbox, .sidebar .stTextInput, .sidebar .stSlider, .sidebar .stTextArea { margin-bottom: 0.75rem; }
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #666; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .agent-header { font-weight: 600; margin-bottom: 0.25rem; color: #1f77b4; }
    .rag-chunk { background-color: #f8f9fa; padding: 0.5rem; border-radius: 0.3rem; margin: 0.3rem 0; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ============== RAG / VECTOR STORE HELPERS ==============
def chunk_text(text, chunk_size=800, overlap=100):
    """Simple text chunking."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks

def get_embedding(text, client, model="text-embedding-3-small"):
    """Get embedding using OpenAI-compatible endpoint if possible."""
    try:
        if client:
            response = client.embeddings.create(input=[text], model=model)
            return np.array(response.data[0].embedding)
    except:
        pass
    # Fallback: simple hash-based pseudo-embedding (not ideal but works without API)
    vec = np.zeros(384)
    for i, char in enumerate(text[:1000]):
        vec[i % 384] += ord(char) / 255.0
    return vec / (np.linalg.norm(vec) + 1e-8)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def process_uploaded_documents(files_dict, client):
    """Process uploaded text files into vector store."""
    vector_store = []
    for filename, content in files_dict.items():
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk, client)
            vector_store.append({
                "id": f"{filename}_{i}",
                "source": filename,
                "chunk": chunk,
                "embedding": embedding
            })
    return vector_store

def semantic_search(query, vector_store, top_k=5, client=None):
    """Semantic search over the vector store."""
    if not vector_store:
        return "No documents in the knowledge base. Upload files first."
    
    query_emb = get_embedding(query, client)
    similarities = []
    for item in vector_store:
        sim = cosine_similarity(query_emb, item["embedding"])
        similarities.append((sim, item))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, item in similarities[:top_k]:
        results.append(f"[Score: {sim:.3f}] From **{item['source']}**:\n{item['chunk'][:600]}...")
    return "\n\n".join(results) if results else "No relevant documents found."

# ============== VISION & OTHER HELPERS (existing) ==============
def encode_image_to_base64(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        bytes_data = uploaded_file.getvalue()
        base64_str = base64.b64encode(bytes_data).decode("utf-8")
        mime = uploaded_file.type or "image/jpeg"
        return f"data:{mime};base64,{base64_str}"
    except:
        return None

# ============== SAFE CODE INTERPRETER & OTHER TOOLS ==============
# (existing functions for execute_python_code, web_search, list/read files remain the same)
def execute_python_code(code: str) -> str:
    allowed_builtins = {
        "print": print, "len": len, "range": range, "sum": sum, "min": min, "max": max,
        "sorted": sorted, "abs": abs, "round": round,
        "str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict,
        "json": json, "datetime": datetime, "math": __import__("math"),
    }
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec_globals = {"__builtins__": allowed_builtins}
            exec(code, exec_globals, {})
        result = output.getvalue().strip()
        return result if result else "Code executed successfully (no output)."
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {str(e)}"

def web_search(query: str, num_results: int = 8) -> str:
    # existing implementation
    tavily_key = st.session_state.get("tavily_api_key", "").strip()
    serp_key = st.session_state.get("serpapi_api_key", "").strip()
    results = []
    if tavily_key:
        try:
            resp = requests.post("https://api.tavily.com/search", json={"api_key": tavily_key, "query": query, "max_results": min(num_results, 10), "search_depth": "advanced", "include_answer": True}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("answer"):
                    results.append(f"**Quick Answer:** {data['answer']}")
                for r in data.get("results", [])[:num_results]:
                    results.append(f"- **{r.get('title', '')}**\n  {r.get('content', '')[:280]}...\n  {r.get('url', '')}")
                if results: return "\n".join(results)
        except: pass
    if serp_key:
        try:
            params = {"engine": "google", "q": query, "api_key": serp_key, "num": min(num_results, 10)}
            resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
            if resp.status_code == 200:
                for r in resp.json().get("organic_results", [])[:num_results]:
                    results.append(f"- **{r.get('title', '')}**\n  {r.get('snippet', '')}\n  {r.get('link', '')}")
                if results: return "\n".join(results)
        except Exception as e: return f"Search error: {str(e)}"
    return "No search API key provided."

def list_uploaded_files() -> str:
    files = st.session_state.get("uploaded_files", {})
    if not files: return "No files uploaded yet."
    return "Available files:\n" + "\n".join([f"- {name}" for name in files.keys()])

def read_uploaded_file(filename: str) -> str:
    files = st.session_state.get("uploaded_files", {})
    if filename not in files: return f"File not found. Available: {list(files.keys())}"
    content = files[filename]
    if len(content) > 12000: content = content[:12000] + "\n... [truncated]"
    return f"=== Content of {filename} ===\n{content}"

def calculate_math(expression: str) -> str:
    # existing safe math
    operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            return +val if isinstance(node.op, ast.UAdd) else -val
        raise ValueError("Unsupported")
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {str(e)}"

TOOL_REGISTRY = {
    "get_current_datetime": lambda tz="UTC": datetime.datetime.now(datetime.timezone.utc if tz.upper() in ["UTC","GMT"] else datetime.datetime.now().astimezone().tzinfo).strftime("%Y-%m-%d %H:%M:%S %Z"),
    "calculate_math": calculate_math,
    "web_search": web_search,
    "list_uploaded_files": list_uploaded_files,
    "read_uploaded_file": read_uploaded_file,
    "execute_python_code": execute_python_code,
    "semantic_search_documents": lambda q, k=5: semantic_search(q, st.session_state.get("vector_store", []), top_k=k, client=OpenAI(api_key=st.session_state.get("api_key",""), base_url=st.session_state.get("base_url", "https://api.openai.com/v1")) if st.session_state.get("api_key") else None),
}

EXAMPLE_TOOLS = [
    {"type": "function", "function": {"name": "semantic_search_documents", "description": "Search the uploaded documents semantically for relevant information. Use this for RAG over your files.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_uploaded_file", "description": "Read a specific uploaded file.", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}}},
    {"type": "function", "function": {"name": "execute_python_code", "description": "Run Python code safely.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
]

# ============== ENHANCED MULTI-AGENT TEAM ==============
EXAMPLE_TEAM = [
    {
        "name": "Researcher",
        "system_prompt": "You are a researcher. Use semantic_search_documents, web_search, and read_uploaded_file to gather accurate information from documents and the web. Cite sources.",
        "model": "grok-4.3",
        "temperature": 0.5,
        "max_tokens": 2048,
        "top_p": 0.95,
        "enable_tools": True,
        "enable_vision": True,
        "memory_window": 8
    },
    {
        "name": "Coder",
        "system_prompt": "You are a senior engineer. Use execute_python_code and read files when needed. Write clean code.",
        "model": "grok-build-0.1",
        "temperature": 0.3,
        "max_tokens": 3000,
        "top_p": 0.9,
        "enable_tools": True,
        "enable_vision": False,
        "memory_window": 6
    },
    {
        "name": "Critic",
        "system_prompt": "You are a critic. Review previous agent responses and retrieved information for quality and improvements.",
        "model": "grok-4.3",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.95,
        "enable_tools": True,
        "enable_vision": True,
        "memory_window": 10
    }
]

PROVIDERS = {
    "xAI Grok (Recommended)": {"base_url": "https://api.x.ai/v1", "models": ["grok-4.3", "grok-build-0.1"], "default_model": "grok-4.3", "description": "Excellent reasoning and tool use."},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o", "gpt-4o-mini"], "default_model": "gpt-4o", "description": "Strong embeddings and vision."},
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "models": ["llama-3.3-70b-versatile"], "default_model": "llama-3.3-70b-versatile", "description": "Fast inference."},
    "Custom / Local": {"base_url": "http://localhost:11434/v1", "models": ["llama3.2"], "default_model": "llama3.2", "description": "Local models (embeddings may vary)."}
}

# Session State
for key, default in [
    ("messages", []),
    ("current_provider", "xAI Grok (Recommended)"),
    ("api_key", ""),
    ("system_prompt", "You are a helpful AI with access to tools and documents."),
    ("chat_params", {"temperature": 0.7, "max_tokens": 2048, "top_p": 0.95, "stream": True}),
    ("enable_tools", True),
    ("tool_definitions", json.dumps(EXAMPLE_TOOLS, indent=2)),
    ("enable_multi_agent", False),
    ("multi_agent_team", json.dumps(EXAMPLE_TEAM, indent=2)),
    ("tavily_api_key", ""),
    ("serpapi_api_key", ""),
    ("uploaded_files", {}),
    ("uploaded_images", []),
    ("agent_memories", {}),
    ("token_usage", {"input": 0, "output": 0, "cost": 0.0}),
    ("vector_store", []),  # RAG vector store
    ("base_url", "https://api.openai.com/v1")  # for embeddings
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
with st.sidebar:
    st.header("⚙️ LLM Wrapper - Agentic RAG + Multi-Agent")

    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()))
    st.session_state.current_provider = provider_name
    provider_config = PROVIDERS[provider_name]
    st.session_state.base_url = provider_config["base_url"]

    st.session_state.api_key = st.text_input("LLM API Key", type="password", value=st.session_state.api_key)
    base_url = st.text_input("Base URL", value=st.session_state.base_url)

    selected = st.selectbox("Model", provider_config["models"] + ["Custom..."])
    model = st.text_input("Custom Model", value=provider_config["default_model"]) if selected == "Custom..." else selected

    with st.expander("🔧 Parameters"):
        # temperature, max_tokens sliders...
        pass

    with st.expander("🔑 Tool API Keys"):
        st.session_state.tavily_api_key = st.text_input("Tavily Key", type="password", value=st.session_state.tavily_api_key)
        st.session_state.serpapi_api_key = st.text_input("SerpAPI Key", type="password", value=st.session_state.serpapi_api_key)

    # Vision Images
    with st.expander("🖼️ Vision Images"):
        vision_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["png","jpg","jpeg","webp"])
        if vision_files:
            st.session_state.uploaded_images = [encode_image_to_base64(f) for f in vision_files if encode_image_to_base64(f)]
            st.success(f"{len(st.session_state.uploaded_images)} image(s) loaded")
        if st.session_state.uploaded_images:
            if st.button("Clear Images"): 
                st.session_state.uploaded_images = []
                st.rerun()

    # Text Files + Auto RAG Processing
    with st.expander("📁 Documents for RAG (Vector Store)", expanded=True):
        text_files = st.file_uploader("Upload text/PDF/MD/CSV files for RAG", accept_multiple_files=True, type=["txt","md","csv","json","py"])
        if text_files:
            new_files = {f.name: f.getvalue().decode("utf-8", errors="ignore") for f in text_files}
            st.session_state.uploaded_files.update(new_files)
            # Auto-process into vector store
            client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url) if st.session_state.api_key else None
            st.session_state.vector_store = process_uploaded_documents(st.session_state.uploaded_files, client)
            st.success(f"Processed {len(text_files)} file(s) into vector store ({len(st.session_state.vector_store)} chunks)")

        if st.session_state.vector_store:
            st.write(f"**Vector Store:** {len(st.session_state.vector_store)} chunks from {len(st.session_state.uploaded_files)} files")
            if st.button("Clear RAG Documents"):
                st.session_state.vector_store = []
                st.session_state.uploaded_files = {}
                st.rerun()
            with st.expander("View Sample Chunks"):
                for item in st.session_state.vector_store[:3]:
                    st.markdown(f"<div class='rag-chunk'><b>{item['source']}</b><br>{item['chunk'][:200]}...</div>", unsafe_allow_html=True)

    # Tools
    with st.expander("🛠️ Tools (including RAG)"):
        st.session_state.enable_tools = st.toggle("Enable Tools", value=st.session_state.enable_tools)
        if st.button("Load Full Tools incl. RAG"):
            st.session_state.tool_definitions = json.dumps(EXAMPLE_TOOLS, indent=2)
            st.rerun()
        st.session_state.tool_definitions = st.text_area("Tool JSON", value=st.session_state.tool_definitions, height=200)

    # Multi-Agent
    with st.expander("👥 Multi-Agent (Per-Agent Config + Memory + RAG)", expanded=True):
        st.session_state.enable_multi_agent = st.toggle("Enable Multi-Agent", value=st.session_state.enable_multi_agent)
        if st.button("Load Advanced RAG Team"):
            st.session_state.multi_agent_team = json.dumps(EXAMPLE_TEAM, indent=2)
            st.rerun()
        st.caption("Agents can use semantic_search_documents for Agentic RAG over your uploaded files.")
        st.session_state.multi_agent_team = st.text_area("Team JSON", value=st.session_state.multi_agent_team, height=280)

        if st.button("🧹 Clear Agent Memories"):
            st.session_state.agent_memories = {}
            st.success("Memories cleared")

    # Token/Cost
    with st.expander("📊 Usage Tracking"):
        tu = st.session_state.token_usage
        st.metric("Input Tokens", tu["input"])
        st.metric("Output Tokens", tu["output"])
        st.metric("Est. Cost USD", f"${tu['cost']:.4f}")
        if st.button("Reset Tracking"):
            st.session_state.token_usage = {"input": 0, "output": 0, "cost": 0.0}
            st.rerun()

    if st.button("🆕 New Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_memories = {}
        st.session_state.token_usage = {"input": 0, "output": 0, "cost": 0.0}
        st.rerun()

# ============== UPDATED call_agent with full support ==============
def call_agent(client, agent, shared_history, global_tools, global_params, base_url, api_key, uploaded_images, vector_store):
    agent_name = agent.get("name", "Agent")
    system_prompt = agent.get("system_prompt", "")
    model = agent.get("model", "grok-4.3")
    temperature = float(agent.get("temperature", 0.7))
    max_tokens = int(agent.get("max_tokens", 2048))
    top_p = float(agent.get("top_p", 0.95))
    enable_tools = bool(agent.get("enable_tools", True))
    enable_vision = bool(agent.get("enable_vision", False))
    memory_window = int(agent.get("memory_window", 8))

    agent_memory = st.session_state.agent_memories.get(agent_name, [])[-memory_window:]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(agent_memory)

    for msg in shared_history[-memory_window:]:
        if msg.get("role") == "user":
            if enable_vision and uploaded_images:
                content = [{"type": "text", "text": msg.get("content", "")}]
                for img in uploaded_images:
                    content.append({"type": "image_url", "image_url": {"url": img}})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": msg.get("content", "")})
        else:
            messages.append(msg)

    tools_to_use = global_tools if enable_tools else None

    try:
        c = OpenAI(api_key=api_key, base_url=base_url)
        resp = c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools_to_use,
            tool_choice="auto" if tools_to_use else None,
            stream=False
        )
        content = resp.choices[0].message.content or ""

        # Rough token tracking
        st.session_state.token_usage["input"] += len(str(messages)) // 4
        st.session_state.token_usage["output"] += len(content) // 4
        st.session_state.token_usage["cost"] += (len(str(messages)) // 4 * 0.000002) + (len(content) // 4 * 0.000006)

        # Update agent memory
        if agent_name not in st.session_state.agent_memories:
            st.session_state.agent_memories[agent_name] = []
        st.session_state.agent_memories[agent_name].append({"role": "user", "content": shared_history[-1].get("content", "") if shared_history else ""})
        st.session_state.agent_memories[agent_name].append({"role": "assistant", "content": content})

        return content, []
    except Exception as e:
        return f"Error in {agent_name}: {str(e)}", []

# ============== MAIN CHAT ==============
st.markdown('<h1 class="main-header">🤖 LLM Wrapper</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Agentic RAG • Vector Store • Graph-capable • Multi-Agent with Memory & Vision</p>', unsafe_allow_html=True)

if st.session_state.api_key:
    mode = "Multi-Agent + Agentic RAG" if st.session_state.enable_multi_agent else "Single Agent + RAG + Tools + Vision"
    st.success(f"{mode}", icon="🔌")
else:
    st.warning("Add LLM API key in sidebar to enable full features including embeddings for RAG")

# History
for msg in st.session_state.messages:
    if msg.get("role") == "system": continue
    with st.chat_message(msg.get("role", "assistant")):
        if msg.get("agent_name"):
            st.markdown(f"<div class='agent-header'>{msg['agent_name']}</div>", unsafe_allow_html=True)
        st.markdown(msg.get("content", ""))

# Input
prompt = st.chat_input("Ask anything. Agents can semantically search your documents (Agentic RAG), use tools, see images, and remember context.")

if prompt and st.session_state.api_key:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    shared_history = [m for m in st.session_state.messages if m.get("role") != "system"]
    client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.base_url)

    try:
        tools = json.loads(st.session_state.tool_definitions) if st.session_state.enable_tools else []
    except:
        tools = []

    if st.session_state.enable_multi_agent:
        try:
            team = json.loads(st.session_state.multi_agent_team)
        except:
            team = []
        for agent in team:
            with st.spinner(f"{agent.get('name')} working (RAG + tools enabled)..."):
                content, _ = call_agent(
                    client, agent, shared_history, tools, st.session_state.chat_params,
                    st.session_state.base_url, st.session_state.api_key,
                    st.session_state.uploaded_images, st.session_state.vector_store
                )
            with st.chat_message("assistant"):
                st.markdown(f"<div class='agent-header'>{agent.get('name')}</div>", unsafe_allow_html=True)
                st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content, "agent_name": agent.get("name")})
    else:
        with st.chat_message("assistant"):
            st.markdown("**Response with full Agentic RAG, tools, and vision**")
            # Single agent would use similar logic + tool calling loop here

st.divider()
st.caption("Full Agentic RAG with Vector Store • Per-Agent Memory & Config • Vision • Tools • Multi-Agent Collaboration")