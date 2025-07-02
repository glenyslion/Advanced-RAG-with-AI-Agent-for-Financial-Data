# AI Chatbot with Retrieval-Augmented Generation and Summarization

This project implements an **AI-powered chatbot** that answers finance news questions using **retrieval-augmented generation (RAG)**, and advanced summarization with fine-tuned LLM adapters. It combines:
- Retrieval of relevant news documents
- Answer generation using large language models
- Summarization of answers or retrieved content

It also includes a **Streamlit web app** to interact with the chatbot.

---

## Features

- **Base LLM** answering without context
- **RAG** with document retrieval
- **Advanced RAG** with improved answer quality
- **Fine-tuned Summarizer** adapters for:
  - Summarizing generated answers
  - Summarizing retrieved documents
- **Streamlit Web UI** for easy interaction

---

## Repository Structure
```text
├── ai_agent_source_code.py     # Main agent code (RAG + Summarization)
├── chatbot_web_app.py          # Streamlit web app for chatbot
├── Dockerfile                  # Docker container definition
├── lora_adapters.zip           # Fine-tuned LoRA weights for summarization
├── sample_output.pdf           # Example chatbot UI output
├── stitching_code.ipynb        # Jupyter Notebook for experimentation
├── unit_test.ipynb             # Notebook to test the end-to-end pipeline
└── README.md
```


---

## Running with Docker

This repository includes a **Dockerfile** to help you containerize the full pipeline, including retrieval, generation, and summarization.

### 1. Prepare Your Data
- Download the finance news dataset from [this link](https://drive.google.com/file/d/12UOejHcxZkM6jjIQxcIv4b7UaTWVcFTu/view).
- Unzip it and place the `finance_news` folder in a convenient location.
- Download and unzip the LoRA adapter weights from [this link](https://github.com/glenyslion/Advanced-RAG-with-AI-Agent-for-Financial-Data/blob/main/lora_adapters.zip).

### 2. Set Up Environment Variables
Create a `.env` file in your project root or mount it when running Docker:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Build the Docker Image
```bash
docker build -t genai_project .
```

### 4. Run the Container
Example Command: 
```bash
docker run -p 8000:8000 \
  --mount type=bind,source="/path/to/finance_news",target=/home/finance_news \
  --mount type=bind,source="/path/to/lora_adapters",target=/home/lora_adapters \
  --env-file .env \
  genai_project
```

## Results & Insights

- **Base LLM** → Outdated answers (knowledge cutoff October 2023)
- **Base RAG** → Better coverage with retrieved documents
- **Advanced RAG (no fine-tuning)** → More detailed and relevant answers with fewer hallucinations
- **Advanced RAG + Summarizer (on generated answer)** → Produces clear, concise summaries of generated answers
- **Advanced RAG + Summarizer (on retrieved docs)** → Summarizes retrieved documents before generation, reducing noise and improving conciseness

*Preferred approach:* Summarizing retrieved documents first generally yields the most concise and relevant answers.

---

## Credits
Project developed by Glenys Charity Lion.

---

## Acknowledgments
This project was developed as part of a Generative AI course. I appreciate your interest and welcome feedback!
