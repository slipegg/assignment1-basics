mkdir -p logs

docker run -d \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --cpus="8" \
    --memory="32g" \
    --gpus '"device=1"' \
    --network host \
    --mount type=bind,source="$(pwd)",target=/app \
    --mount type=bind,source=/home/ljw/.vscode-server,target=/root/.vscode-server \
    huahuadan/cs336-assignment1:v1 \
    bash -c "uv sync && .venv/bin/python3 scripts/train_lm.py > /app/logs/train_lm_lr_1e-2.log 2>&1"
