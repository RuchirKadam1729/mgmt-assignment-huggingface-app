mkdir -p "$(pwd)/checkpoints"
docker run -it --rm \
  -p 7860:7860 \
  -e GROQ_API_KEY=$(cat secrets/GROQ_API_KEY) \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mgmt-assignment-huggingface-app