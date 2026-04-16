# HuggingFace Tutorial — Complete Syllabus (38 Notebooks)

## Overview

7 sections, 38 notebooks. Each notebook follows the XX_OO naming convention where XX = section number, OO = notebook number within section.

**Legend**: [EXISTS] = already built | [NEW] = to be created | [RENAME] = existing notebook being renumbered

---

## 00 — Transformer Fundamentals (5 notebooks) [NEW section]

| File | Title | Key Topics | Models/Tools |
|------|-------|------------|--------------|
| `00_01_tokenization_embeddings.ipynb` | Tokenization & Embeddings | BPE, WordPiece, SentencePiece, vocabulary, special tokens, embedding layers, positional encoding | `AutoTokenizer`, `tiktoken` |
| `00_02_transformer_architecture.ipynb` | Transformer Architecture | Self-attention, multi-head attention, encoder vs decoder, feed-forward layers, layer norm — inspect real model internals | `AutoModel`, attention visualization |
| `00_03_huggingface_ecosystem.ipynb` | HuggingFace Ecosystem Tour | Hub navigation, AutoClasses, pipelines, model configs, tokenizer configs, model cards, Spaces | `pipeline`, `AutoModel`, `HfApi` |
| `00_04_preprocessors_feature_extractors.ipynb` | Preprocessors & Feature Extractors | Unified preprocessing layer, padding/truncation strategies, image normalization, mel spectrograms, multimodal processors | `AutoImageProcessor`, `AutoFeatureExtractor`, `AutoProcessor` |
| `00_05_model_configuration_customization.ipynb` | Model Configuration & Customization | config.json anatomy, modifying architectures, creating models from scratch, model surgery (freezing, head swapping), memory estimation | `AutoConfig`, `DistilBertConfig`, `GPT2Config` |

**Learning outcome**: Students understand *what* transformers are and *how* the HuggingFace ecosystem is organized before using specific models.

---

## 01 — Natural Language Processing (8 notebooks)

| File | Title | Status | Key Topics | Models |
|------|-------|--------|------------|--------|
| `01_01_nlp_text_generation.ipynb` | Text Generation | [EXISTS] | Causal LM, greedy/beam/sampling decoding | GPT-2, DistilGPT2 |
| `01_02_nlp_text_classification.ipynb` | Text Classification | [EXISTS] | Sentiment analysis, sequence classification | DistilBERT, BERT |
| `01_03_nlp_text_summarization.ipynb` | Text Summarization | [EXISTS] | Abstractive summarization, seq2seq | DistilBART, BART |
| `01_04_nlp_ner.ipynb` | Named Entity Recognition | [NEW] | Token classification, NER tags (PER/ORG/LOC), span extraction | DistilBERT-NER, BERT-NER |
| `01_05_nlp_question_answering.ipynb` | Question Answering | [NEW] | Extractive QA (SQuAD-style), context + question → answer span | DistilBERT-QA, RoBERTa-QA |
| `01_06_nlp_translation.ipynb` | Translation & Multilingual | [NEW] | Machine translation, multilingual models, language detection | MarianMT, NLLB-200, mBART |
| `01_07_nlp_fine_tuning_unsloth.ipynb` | Fine-tuning (Unsloth) | [RENAME] was 01_04 | 2-5x faster LLM fine-tuning | Llama 3.2/3.1 |
| `01_08_nlp_fine_tuning_lora.ipynb` | Fine-tuning (LoRA) | [RENAME] was 01_05 | Parameter-efficient fine-tuning | GPT-2 + LoRA |

**Changes**: 3 new core NLP notebooks (NER, QA, Translation), fine-tuning notebooks renumbered to 01_07/01_08.

---

## 02 — Computer Vision (4 notebooks)

| File | Title | Status | Key Topics | Models |
|------|-------|--------|------------|--------|
| `02_01_cv_image_classification.ipynb` | Image Classification | [EXISTS] | ViT, image preprocessing, top-k predictions | ViT-base, ViT-large |
| `02_02_cv_object_detection.ipynb` | Object Detection | [EXISTS] | Bounding boxes, COCO categories | DETR, YOLOv8 |
| `02_03_cv_ocr.ipynb` | OCR | [EXISTS] | Text extraction from images | TrOCR, PaddleOCR |
| `02_04_cv_image_segmentation.ipynb` | Image Segmentation | [NEW] | Semantic segmentation, instance segmentation, panoptic | SAM, Mask2Former, SegFormer |

**Changes**: 1 new notebook (segmentation).

---

## 03 — Audio (3 notebooks)

| File | Title | Status | Key Topics | Models |
|------|-------|--------|------------|--------|
| `03_01_audio_speech_recognition.ipynb` | Speech Recognition | [EXISTS] | ASR, transcription, multilingual | Whisper tiny/small |
| `03_02_audio_text_to_speech.ipynb` | Text-to-Speech | [EXISTS] | TTS, speech synthesis | SpeechT5 |
| `03_03_audio_classification.ipynb` | Audio Classification | [NEW] | Environmental sounds, music genre, speaker verification | Wav2Vec2, Audio Spectrogram Transformer |

**Changes**: 1 new notebook (audio classification).

---

## 04 — Multimodal (5 notebooks)

| File | Title | Status | Key Topics | Models |
|------|-------|--------|------------|--------|
| `04_01_multimodal_image_to_text.ipynb` | Image-to-Text (Captioning) | [EXISTS] | Image captioning, visual description | BLIP, BLIP-2 |
| `04_02_multimodal_vqa.ipynb` | Visual Question Answering | [NEW] | Ask questions about images, VQA datasets | BLIP-2, Pix2Struct, ViLT |
| `04_03_multimodal_text_to_image.ipynb` | Text-to-Image Generation | [NEW] | Stable Diffusion, prompt engineering, negative prompts, schedulers | Stable Diffusion (via `diffusers`) |
| `04_04_multimodal_image_editing.ipynb` | Image Editing & Inpainting | [NEW] | Inpainting, img2img, instruct-pix2pix, ControlNet basics | SD Inpainting, InstructPix2Pix |
| `04_05_multimodal_document_understanding.ipynb` | Document Understanding | [NEW] | Form parsing, receipt extraction, document QA | LayoutLM, Donut |

**Changes**: 4 new notebooks. This section goes from 1 to 5 notebooks — the biggest expansion.

---

## 05 — Best Practices & Production (8 notebooks)

| File | Title | Status | Key Topics | Models/Tools |
|------|-------|--------|------------|--------------|
| `05_01_ollama_integration.ipynb` | Ollama Integration | [EXISTS] | Local LLMs, Ollama API, HF tokenizer combo | TinyLlama via Ollama |
| `05_02_performance_caching_costs.ipynb` | Performance & Caching | [EXISTS] | Latency, throughput, batching, memory, cost estimation | DistilBERT benchmark |
| `05_03_model_cards_responsible_ai.ipynb` | Model Cards & Responsible AI | [EXISTS] | Bias detection, ethics, system cards | Various |
| `05_04_huggingface_datasets.ipynb` | HuggingFace Datasets | [EXISTS] | Load, filter, map, custom datasets, DataLoaders | `datasets` library |
| `05_05_gradio_spaces.ipynb` | Gradio & Spaces | [EXISTS] | Interactive demos, Blocks, deployment | `gradio` |
| `05_06_quantization_compression.ipynb` | Quantization Intro | [EXISTS] | FP16, PyTorch dynamic quant, ONNX export | PyTorch, ONNX Runtime |
| `05_07_quantization_deep_dive.ipynb` | Quantization Deep Dive | [NEW] | GPTQ, AWQ, bitsandbytes 4-bit/8-bit, quality evaluation, perplexity comparison | `auto-gptq`, `bitsandbytes`, `autoawq` |
| `05_08_training_best_practices.ipynb` | Training Best Practices | [NEW] | Trainer API, learning rate schedules, mixed precision, gradient accumulation, early stopping, checkpointing | `Trainer`, `TrainingArguments` |

**Changes**: 2 new notebooks. Quantization gets a proper deep dive, and Trainer API gets its own notebook.

---

## 06 — Agentic Workflows (5 notebooks)

| File | Title | Status | Key Topics | Models/Tools |
|------|-------|--------|------------|--------------|
| `06_01_mcp_basics.ipynb` | MCP Basics | [EXISTS] | Model Context Protocol, tool calling, agent loop | Ollama + tools |
| `06_02_mcp_servers.ipynb` | MCP Servers | [EXISTS] | Reusable tool servers, file system tools | Ollama + MCP |
| `06_03_multi_tool_agents.ipynb` | Multi-Tool Agents | [EXISTS] | ReAct, Plan-and-Execute, Reflection patterns | Ollama + tools |
| `06_04_rag_local_llms.ipynb` | RAG with Local LLMs | [EXISTS] | Vector databases, semantic search, context injection | FAISS, ChromaDB, Ollama |
| `06_05_structured_output.ipynb` | Structured Output & Function Calling | [NEW] | JSON mode, Pydantic output parsing, tool use patterns, schema validation | Ollama, `instructor` |

**Changes**: 1 new notebook.

---

## Summary

| Section | Notebooks | Existing | New | Renamed |
|---------|-----------|----------|-----|---------|
| 00 Fundamentals | 5 | 0 | 5 | 0 |
| 01 NLP | 8 | 5 | 3 | 2 (01_04→01_07, 01_05→01_08) |
| 02 Computer Vision | 4 | 3 | 1 | 0 |
| 03 Audio | 3 | 2 | 1 | 0 |
| 04 Multimodal | 5 | 1 | 4 | 0 |
| 05 Best Practices | 8 | 6 | 2 | 0 |
| 06 Agentic | 5 | 4 | 1 | 0 |
| **Total** | **38** | **21** | **17** | **2** |

---

## File Operations Required

### New folder
```
notebooks/00_fundamentals/
```

### Renames (NLP fine-tuning pushed to end)
```
01_04_nlp_fine_tuning_unsloth.ipynb → 01_07_nlp_fine_tuning_unsloth.ipynb
01_05_nlp_fine_tuning_lora.ipynb    → 01_08_nlp_fine_tuning_lora.ipynb
```

### New notebooks to create (17 total)
```
00_01_tokenization_embeddings.ipynb
00_02_transformer_architecture.ipynb
00_03_huggingface_ecosystem.ipynb
00_04_preprocessors_feature_extractors.ipynb
00_05_model_configuration_customization.ipynb
01_04_nlp_ner.ipynb
01_05_nlp_question_answering.ipynb
01_06_nlp_translation.ipynb
02_04_cv_image_segmentation.ipynb
03_03_audio_classification.ipynb
04_02_multimodal_vqa.ipynb
04_03_multimodal_text_to_image.ipynb
04_04_multimodal_image_editing.ipynb
04_05_multimodal_document_understanding.ipynb
05_07_quantization_deep_dive.ipynb
05_08_training_best_practices.ipynb
06_05_structured_output.ipynb
```

---

## Implementation Order (suggested)

**Phase 1 — Foundation + Renames**
1. Create `00_fundamentals/` folder
2. Rename NLP fine-tuning notebooks (01_04→01_07, 01_05→01_08)
3. Build 00_01, 00_02, 00_03

**Phase 2 — Core Task Notebooks**
4. Build 01_04 (NER), 01_05 (QA), 01_06 (Translation)
5. Build 02_04 (Segmentation)
6. Build 03_03 (Audio Classification)

**Phase 3 — Multimodal Expansion**
7. Build 04_02 (VQA), 04_03 (Text-to-Image), 04_04 (Image Editing), 04_05 (Document Understanding)

**Phase 4 — Best Practices + Agentic**
8. Build 05_07 (Quantization Deep Dive), 05_08 (Training Best Practices)
9. Build 06_05 (Structured Output)

**Phase 5 — Update docs**
10. Update CLAUDE.md, README.md, requirements.txt, getting_started.md
