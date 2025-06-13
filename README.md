# WWDC RAG Tools

A set of tools for converting WWDC (Apple Worldwide Developers Conference) talk transcripts into optimized chunks for Retrieval-Augmented Generation (RAG) applications, using LM Studio and local language models.

## Overview

This project demonstrates how to prepare documents for RAG applications with proper metadata preservation. It focuses specifically on WWDC talk transcripts but can be adapted for other document types.

The workflow consists of two main steps:
1. Extracting transcripts from WWDC videos
2. Converting these transcripts into context-rich, semantically meaningful chunks using local language models

## Why This Matters

When using local RAG applications like LM Studio or Open Web UI, you often don't have control over the metadata that accompanies your documents. This metadata provides crucial context that helps language models better understand and retrieve relevant information:

- **Source attribution**: Knowing which WWDC session the information came from
- **Temporal context**: Understanding when the information was published (which year)
- **Topic coherence**: Maintaining semantic boundaries between different concepts
- **Information hierarchy**: Preserving the relationships between main topics and subtopics

By pre-processing the documents with a semantic chunking approach, we can ensure that each chunk contains not just the content itself but also the contextual metadata needed for accurate retrieval.

## Components

### 1. Transcript Extractor (`extract-transcripts.py`)

A tool to scrape WWDC talk transcripts from Apple's developer website. It:
- Takes a base URL as input (e.g., "https://developer.apple.com/videos/wwdc2025/")
- Extracts all video links from that page
- Downloads the transcript for each video
- Saves the transcript with metadata (year, title, URL) in markdown format

### 2. LLM-Assisted Chunker (`assisted-rag.py`)

A tool that uses local language models via LM Studio to:
- Process the extracted transcripts
- Split them into semantically coherent chunks
- Generate a title and summary for each chunk
- Save the resulting chunks as JSON files with preserved metadata
- Handle large documents through progressive splitting

## Setup and Usage

### Prerequisites

- Python 3.7+
- LM Studio (with a language model loaded and the API server running)
- Required Python packages: `pip install beautifulsoup4 requests click`

### Step 1: Extract Transcripts

```bash
python extract-transcripts.py --base_url "https://developer.apple.com/videos/wwdc2023/"
```

This will download transcripts to the `./transcripts` directory.

### Step 2: Process Transcripts into RAG Chunks

```bash
python assisted-rag.py --model_name "qwen/qwen3-14b"
```

Options:
- `--model_name`: Specify which model to use (must be loaded in LM Studio)
- `--restart`: Restart processing from the beginning
- `--file`: Process a single transcript file

The processed chunks will be saved in the `./rag-chunks/{model_name}/` directory.

## Better Ways to Do This?

This project is an exploration of RAG document preparation techniques, and I'm still learning how all of this works. There are likely more efficient approaches or better practices for document chunking and metadata preservation.

Some areas for improvement might include:
- More sophisticated chunking algorithms
- Better token estimation
- Integration with vector stores
- Parallel processing for large document collections

I welcome comments on the repository and pull requests with improvements!

## License

MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
