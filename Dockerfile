FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

SHELL ["/bin/bash", "-c"]
RUN apt-get update -qq && apt-get upgrade -qq &&\
    apt-get install -qq man wget sudo vim tmux

RUN apt update

RUN apt install -y cudnn9

RUN yes | pip install --upgrade pip

# Install required Python libraries
RUN yes | pip install numpy==1.26.4 \
    pandas pydantic typing-extensions \
    transformers accelerate peft datasets \
    scikit-learn bert-score evaluate \
    openai langchain langchain_openai langchain_community \
    langchain_pinecone langgraph fastembed \
    sentence-transformers pinecone-client \
    streamlit dotenv fastembed ipython \
    matplotlib huggingface_hub \
    langchain_text_splitters langchain_core \
    langchain_community

RUN pip cache purge

RUN pip install notebook

WORKDIR /home

COPY unit_test.ipynb /home/

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip", "0.0.0.0","--allow-root", "--no-browser"]
