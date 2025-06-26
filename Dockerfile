FROM python:3.10-slim

WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y curl
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple gradio
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple langchain
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple torch --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple pandas
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple faiss-cpu
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple requests
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple langchain-community
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple transformers
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple accelerate
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple sentence-transformers
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple langchain-huggingface

EXPOSE 8080
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
