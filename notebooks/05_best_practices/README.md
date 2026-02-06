# Best Practices & Production

This folder contains 3 notebooks covering production deployment, optimization, and responsible AI.

## Notebooks

### Notebook 10: Ollama Integration
**Concepts**: Local LLM deployment, model serving, API integration
**Models**: TinyLlama (637MB), Llama 3.2 (1B/3B), Phi-3 (2.4GB)
**Demo**: Run large language models locally without cloud APIs

**Quick Demo:**
```bash
# Install Ollama from ollama.ai, then:
ollama pull tinyllama
ollama run tinyllama "Explain machine learning in simple terms"
```

**Python Integration:**
```python
import ollama

response = ollama.chat(
    model='tinyllama',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])
```

**What you'll learn:**
- Install and configure Ollama
- Pull and manage local models
- Integrate Ollama with HuggingFace workflows
- Compare local vs cloud LLM performance
- Build offline-capable applications

**Use Cases:**
- Privacy-sensitive applications
- Offline/air-gapped environments
- Cost reduction (no API fees)
- Low-latency inference
- Development and testing

---

### Notebook 11: Performance, Caching, and Cost Analysis
**Concepts**: Latency optimization, throughput, memory profiling, cost estimation
**Demo**: Measure and optimize model performance for production

**Key Metrics Covered:**
```python
# Latency measurement
import time
start = time.time()
result = model(input_data)
latency = time.time() - start
print(f"Latency: {latency*1000:.2f}ms")

# Throughput calculation
num_requests = 100
total_time = benchmark(model, num_requests)
throughput = num_requests / total_time
print(f"Throughput: {throughput:.2f} requests/second")

# Memory profiling
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.2f}MB")
```

**Visualizations Included:**
- Cache size bar charts
- Latency vs model size comparisons
- Throughput vs batch size plots
- Memory usage over time
- Cost estimation charts

**What you'll learn:**
- Measure inference latency and throughput
- Profile memory usage
- Understand HuggingFace model caching
- Estimate cloud API costs
- Optimize for production workloads
- Choose between CPU, GPU, and API deployment

**Production Considerations:**
- **Latency**: Target <100ms for interactive apps
- **Throughput**: Scale with batch processing
- **Memory**: Monitor to prevent OOM errors
- **Cost**: Calculate $/1000 requests for budgeting

---

### Notebook 12: Model Cards and Responsible AI
**Concepts**: Bias detection, fairness, transparency, model documentation
**Demo**: Evaluate models for ethical deployment

**Key Topics:**

**1. Model Cards:**
```
Every HuggingFace model has a Model Card that documents:
- Intended use and limitations
- Training data and procedures
- Evaluation metrics
- Ethical considerations
- Licensing and attribution
```

**2. Bias Detection:**
```python
# Example: Testing for gender bias
test_sentences = [
    "The doctor said he would...",
    "The doctor said she would...",
    "The nurse said he would...",
    "The nurse said she would..."
]

for sentence in test_sentences:
    result = model(sentence)
    analyze_bias(result)  # Compare predictions
```

**3. Fairness Evaluation:**
- Test models across demographic groups
- Measure performance disparities
- Identify potential harms
- Document limitations

**What you'll learn:**
- Read and write model cards
- Detect bias in model outputs
- Evaluate fairness across groups
- Document ethical considerations
- Comply with AI regulations (EU AI Act, etc.)
- Make informed deployment decisions

**Critical Thinking Questions:**
- What could go wrong if this model is deployed?
- Who might be harmed by incorrect predictions?
- Is the training data representative?
- Are the limitations clearly communicated?

**No Code Downloads:**
- This notebook is concept-focused
- Minimal model downloads required
- Emphasis on critical analysis

---

## Hardware Requirements

| Notebook | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| 10 (Ollama) | 8GB RAM | 16GB RAM | Models stored separately |
| 11 (Performance) | 8GB RAM | 8GB VRAM (GPU) | Benchmarking multiple models |
| 12 (Ethics) | 4GB RAM | 8GB RAM | Mostly reading/analysis |

## Running the Demos

### Notebook 10 (Ollama):
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)
2. **Pull a model**: `ollama pull tinyllama`
3. **Install Python client**: `pip install ollama`
4. **Run notebook**: Follow along with local model examples

### Notebook 11 (Performance):
1. **Install profiling tools**: `pip install psutil matplotlib seaborn`
2. **Run benchmarks**: Execute timing cells multiple times for accuracy
3. **Analyze visualizations**: Compare different model configurations
4. **Estimate costs**: Use your expected request volume

### Notebook 12 (Ethics):
1. **Read model cards**: Visit HuggingFace model pages
2. **Analyze bias tests**: Run bias detection cells
3. **Discuss findings**: Consider implications with team
4. **Document decisions**: Create model cards for your projects

## Practical Applications

### Ollama (Local LLMs):
- Privacy-compliant chatbots (HIPAA, GDPR)
- Development environments (no internet required)
- Edge deployment (IoT, mobile devices)
- Cost-effective experimentation

### Performance Optimization:
- Production API design
- Resource allocation planning
- Cost-benefit analysis
- SLA compliance (latency targets)

### Responsible AI:
- Model selection for sensitive domains
- Risk assessment before deployment
- Compliance with regulations
- Stakeholder communication

## Key Takeaways

**Notebook 10:**
- ✅ Local LLMs provide privacy and cost benefits
- ✅ Ollama simplifies local model deployment
- ✅ Performance comparable to cloud for many tasks
- ✅ Trade-offs exist (model size, hardware requirements)

**Notebook 11:**
- ✅ Measure latency, throughput, and memory for all deployments
- ✅ HuggingFace caching saves bandwidth and time
- ✅ Batch processing improves throughput
- ✅ Cloud APIs trade cost for convenience
- ✅ Profile before optimizing (data-driven decisions)

**Notebook 12:**
- ✅ All models have biases from training data
- ✅ Model cards provide transparency
- ✅ Test for fairness across demographic groups
- ✅ Document limitations clearly
- ✅ Consider societal impact before deployment

## Common Issues

### Ollama:
- **Issue**: "Model not found"
  - **Solution**: Run `ollama pull <model>` first
- **Issue**: "Connection refused"
  - **Solution**: Ensure Ollama daemon is running: `ollama serve`

### Performance:
- **Issue**: Inconsistent timing results
  - **Solution**: Warm up models first, average multiple runs
- **Issue**: Out of memory during benchmarking
  - **Solution**: Use smaller models or reduce batch size

### Ethics:
- **Issue**: "My model seems unbiased"
  - **Solution**: Test more diverse inputs, consult domain experts
- **Issue**: "How do I fix bias?"
  - **Solution**: Consider data augmentation, fine-tuning, or choosing different models

## Next Steps

After completing this section:
- **Agentic Workflows** (06_agentic_workflows/) - Build autonomous AI agents
- **Fine-Tuning** (Notebooks 04-05 in 01_nlp/) - Customize models for your domain
- **Production Deployment** - Apply learnings to real-world systems

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [HuggingFace Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [EU AI Act](https://artificialintelligenceact.eu/)
- [Partnership on AI](https://partnershiponai.org/)
- [Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit)
