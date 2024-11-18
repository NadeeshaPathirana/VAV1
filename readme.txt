qdrant_api_key = zwDHhJD-ZlGfAEYRnNQSTCOKi_eLQFTDa-CbFOS-HMmAmBO00KP89Q

qdrant_api_key_usage = curl \
    -X GET 'https://83a4aa67-c595-4f58-a5e3-f5d6eb107808.us-east4-0.gcp.cloud.qdrant.io:6333' \
    --header 'api-key: zwDHhJD-ZlGfAEYRnNQSTCOKi_eLQFTDa-CbFOS-HMmAmBO00KP89Q'

qdrant_cluster_url = https://83a4aa67-c595-4f58-a5e3-f5d6eb107808.us-east4-0.gcp.cloud.qdrant.io

1. Select and set the python interpreter
2. Install pytorch with pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 (this is working wiht CUDA 11x)
3. Install requirements file
4. Start Ollama local server -> Ollama serve

communicate with local LLM model - curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\": \"mistral\", \"prompt\": \"Hello, how are you?\"}"