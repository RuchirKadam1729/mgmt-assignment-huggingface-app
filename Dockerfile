FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch — prevents 1.5GB CUDA download
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ML libraries
RUN pip install --no-cache-dir \
    sentence-transformers \
    scikit-learn \
    bertopic \
    hdbscan \
    umap-learn \
    pynndescent

# LangChain stack
RUN pip install --no-cache-dir \
    langchain-core>=0.2.0 \
    langchain-mistralai>=0.1.7 \
    langgraph>=0.0.30

# UI + data — pin gradio, change this layer most often
RUN pip install --no-cache-dir \
    gradio==5.9.1 \
    gradio_client==1.5.2 \
    plotly \
    numpy \
    pandas \
    nltk

RUN python -m nltk.downloader punkt punkt_tab stopwords

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN mkdir -p /app/checkpoints && chmod 777 /app/checkpoints

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]