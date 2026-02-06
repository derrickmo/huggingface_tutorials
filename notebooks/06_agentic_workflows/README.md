# Agentic Workflows

This folder contains 4 notebooks covering tool-using agents, MCP servers, multi-tool patterns, and RAG systems.

## Notebooks

### Notebook 14: MCP Basics
**Concepts**: Model Context Protocol, tool calling, agent loop, ReAct pattern
**Models**: llama3.2:1b (1.3GB, CPU), llama3.1:8b (4.7GB, GPU)
**Demo**: Build your first tool-using agent

**Quick Demo:**
```python
import ollama

tools = [{
    'type': 'function',
    'function': {
        'name': 'calculator',
        'description': 'Perform arithmetic operations',
        'parameters': {
            'type': 'object',
            'properties': {
                'operation': {'type': 'string', 'enum': ['add', 'subtract', 'multiply', 'divide']},
                'a': {'type': 'number'},
                'b': {'type': 'number'}
            },
            'required': ['operation', 'a', 'b']
        }
    }
}]

response = ollama.chat(
    model='llama3.2:1b',
    messages=[{'role': 'user', 'content': 'What is 25 times 17?'}],
    tools=tools
)

# Agent decides to use calculator tool
# Executes: calculator(operation='multiply', a=25, b=17)
# Returns: 425
```

**What you'll learn:**
- Understand Model Context Protocol (MCP)
- Define tools with JSON schema
- Implement agent loop with tool calling
- Handle multi-step reasoning
- Debug tool invocation

**Key Concept - ReAct Pattern:**
```
User Query → LLM Reasons → Calls Tool → Observes Result → Reasons Again → Final Answer
```

---

### Notebook 15: MCP Servers
**Concepts**: Reusable tool servers, file system operations, data analysis, security
**Demo**: Build production-ready MCP servers

**FileSystemServer Example:**
```python
class FileSystemServer:
    def __init__(self, workspace='./mcp_workspace'):
        self.workspace = Path(workspace)
        self.tools = [
            self.read_file_tool(),
            self.write_file_tool(),
            self.list_files_tool(),
            self.delete_file_tool()
        ]

    def _safe_path(self, filename):
        """Prevent path traversal attacks"""
        path = (self.workspace / filename).resolve()
        if not str(path).startswith(str(self.workspace)):
            raise ValueError("Path outside workspace")
        return path
```

**What you'll learn:**
- Design reusable MCP servers
- Implement file system tools
- Build data analysis tools
- Security best practices (path traversal prevention)
- Combine multiple servers
- Tool description best practices

**Practical Use Cases:**
- Automate file management tasks
- Data pipeline agents
- Report generation systems
- Development automation

---

### Notebook 16: Multi-Tool Agents
**Concepts**: Agent patterns (ReAct, Plan-and-Execute, Reflection), multi-step workflows
**Demo**: Compare different agent architectures

**Three Agent Patterns:**

**1. ReAct (Reason + Act):**
```
- Simple iterative loop
- Fastest execution
- Good for straightforward tasks
- May miss complex dependencies
```

**2. Plan-and-Execute:**
```
Phase 1: Create step-by-step plan
Phase 2: Execute plan sequentially
- Better for complex multi-step tasks
- More structured approach
- Slower but more reliable
```

**3. Reflection:**
```
Phase 1: Initial execution
Phase 2: Verify and correct results
- Highest accuracy
- Self-correcting
- Slowest (double execution)
- Best for critical tasks
```

**Quick Demo:**
```python
# ReAct pattern
agent = ReActAgent(tools=comprehensive_tools)
result = agent.run("Check if 17 is prime, then multiply it by 3")

# Plan-and-Execute pattern
agent = PlanExecuteAgent(tools=comprehensive_tools)
result = agent.run("Find all files starting with 'data' and count them")

# Reflection pattern
agent = ReflectionAgent(tools=comprehensive_tools)
result = agent.run("Calculate 144 / 12 and verify the answer")
```

**What you'll learn:**
- Implement ReAct, Plan-and-Execute, and Reflection patterns
- Choose the right pattern for different tasks
- Build comprehensive tool servers (7+ tools)
- Optimize agent performance
- Handle complex multi-step workflows

---

### Notebook 17: RAG with Local LLMs
**Concepts**: Retrieval-Augmented Generation, vector databases, semantic search, embeddings
**Models**: sentence-transformers/all-MiniLM-L6-v2 (80MB), llama3.1:8b (4.7GB)
**Demo**: Build a RAG system with local models

**RAG Architecture:**
```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[Vector Database] → Semantic Search → Top-K Documents
    ↓
[LLM] ← Query + Retrieved Context
    ↓
Grounded Answer (with sources)
```

**Quick Demo (FAISS):**
```python
from sentence_transformers import SentenceTransformer
import faiss
import ollama

# 1. Create embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# 2. Build vector index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 3. Search
query_embedding = model.encode(["What is RAG?"])
distances, indices = index.search(query_embedding, k=3)
context = "\n".join([documents[i] for i in indices[0]])

# 4. Generate answer with context
response = ollama.chat(
    model='llama3.2:1b',
    messages=[{
        'role': 'user',
        'content': f"Context: {context}\n\nQuestion: What is RAG?\n\nAnswer:"
    }]
)
```

**Quick Demo (ChromaDB):**
```python
import chromadb

# Persistent vector database
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.create_collection(name="knowledge_base")

# Add documents (embeddings created automatically)
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Query
results = collection.query(
    query_texts=["What is RAG?"],
    n_results=3
)
```

**What you'll learn:**
- Build RAG pipelines from scratch
- Use FAISS for fast similarity search
- Use ChromaDB for persistent storage
- Create embeddings with sentence-transformers
- Implement semantic search
- Inject context into LLM prompts
- Compare with/without RAG performance
- Build custom knowledge bases

**Use Cases:**
- Company documentation Q&A
- Customer support chatbots
- Research paper analysis
- Code documentation search
- Personal knowledge management

---

## Hardware Requirements

| Notebook | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| 14-16 (MCP/Agents) | 8GB RAM | 12GB VRAM (GPU) | llama3.1:8b needs GPU |
| 17 (RAG) | 8GB RAM | 12GB VRAM (GPU) | Embedding model + LLM |

**Model Storage:**
- Llama 3.2:1b: 1.3GB
- Llama 3.1:8b: 4.7GB
- Embedding models: 80-420MB
- Total: ~6-7GB for full setup

## Prerequisites

### Install Ollama:
```bash
# Download from ollama.ai, then:
ollama pull llama3.2:1b  # Small model (CPU-friendly)
ollama pull llama3.1:8b  # Large model (GPU-optimized)
```

### Install Python Dependencies:
```bash
pip install ollama mcp
pip install sentence-transformers faiss-cpu chromadb  # For Notebook 17
```

## Running the Demos

1. **Ensure Ollama is running**: `ollama serve` (in background)
2. **Activate environment**: `source venv/bin/activate`
3. **Launch Jupyter**: `jupyter notebook`
4. **Start with Notebook 14**: Build foundation first

## CLI Tools

Corresponding CLI tools in `functions/agentic/`:

**1. mcp_agent.py** - Simple calculator agent
```bash
python functions/agentic/mcp_agent.py "What is 127 multiplied by 83?"
python functions/agentic/mcp_agent.py "Calculate (100 - 35) / 5" --model large
```

**2. mcp_server.py** - File system and data analysis server
```bash
python functions/agentic/mcp_server.py "List all files in the workspace"
python functions/agentic/mcp_server.py "Create a file called notes.txt with: Hello World"
python functions/agentic/mcp_server.py "Calculate statistics for: 10, 20, 30, 40, 50"
```

**3. multi_agent.py** - Multi-pattern agent system
```bash
python functions/agentic/multi_agent.py "Check if 17 is prime" --pattern react
python functions/agentic/multi_agent.py "Find files starting with 'data'" --pattern plan
python functions/agentic/multi_agent.py "Calculate 144 / 12" --pattern reflection
```

## Key Concepts

### Model Context Protocol (MCP):
- **Open standard** for AI-tool integration (by Anthropic)
- **Decoupled architecture** - servers expose tools, clients consume them
- **Reusable** - Write once, use with any MCP client
- **Language agnostic** - Python servers, any language client

### Tool Calling:
- LLMs decide **when** to use tools based on descriptions
- Structured tool invocation via JSON schema
- Multi-step reasoning with tool chains
- Error handling and fallback strategies

### Agent Patterns:
- **ReAct**: Fast, iterative reasoning
- **Plan-and-Execute**: Structured, reliable
- **Reflection**: Self-correcting, accurate

### RAG (Retrieval-Augmented Generation):
- **Reduces hallucinations** by grounding in facts
- **Enables dynamic knowledge** without retraining
- **Cites sources** for transparency
- **Scales knowledge** beyond LLM context window

## Practical Applications

**MCP Agents (Notebooks 14-16):**
- Automated code review
- Data pipeline orchestration
- Customer support automation
- Task scheduling and execution

**RAG Systems (Notebook 17):**
- Enterprise knowledge bases
- Legal document search
- Medical literature Q&A
- Technical documentation assistants

## Performance Expectations

**Agent Execution (Notebooks 14-16):**
- **llama3.2:1b (CPU)**: 2-5 seconds per turn
- **llama3.1:8b (GPU)**: 1-3 seconds per turn
- Tool execution: 100-500ms depending on complexity

**RAG Pipeline (Notebook 17):**
- **Embedding**: 50-100ms per query
- **Vector search**: 10-50ms for 1000 documents
- **LLM generation**: 1-4 seconds
- **Total**: 2-5 seconds per query

## Common Issues

### Ollama Connection:
- **Issue**: "Connection refused"
  - **Solution**: Ensure `ollama serve` is running

### Tool Calling:
- **Issue**: Agent doesn't use tools
  - **Solution**: Improve tool descriptions, use larger model

### RAG Accuracy:
- **Issue**: Irrelevant documents retrieved
  - **Solution**: Better embedding model, tune top-K, improve chunking

### Performance:
- **Issue**: Slow inference
  - **Solution**: Use GPU, smaller models, or batch processing

## Next Steps

After completing this section:
- **Combine techniques**: Use RAG with MCP agents
- **Build production apps**: Deploy with FastAPI
- **Explore frameworks**: LangChain, LlamaIndex
- **Advanced RAG**: HyDE, query expansion, re-ranking
- **Multi-agent systems**: Agent collaboration patterns

## Additional Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Ollama Documentation](https://ollama.ai/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Papers](https://arxiv.org/abs/2005.11401)

## Advanced Topics (Beyond This Tutorial)

**Multi-Agent Systems:**
- Agent-to-agent communication
- Task delegation and coordination
- Consensus mechanisms

**Advanced RAG:**
- Hybrid search (keyword + semantic)
- Query expansion and rewriting
- Re-ranking retrieved documents
- Conversational RAG with memory

**Production Deployment:**
- FastAPI REST endpoints
- Docker containerization
- Kubernetes orchestration
- Monitoring and logging
