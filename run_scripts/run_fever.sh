
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

python -m src.main --config configs/fever.yaml --mode warmup_shortcut
python -m src.main --config configs/fever.yaml --mode warmup_grounded
python -m src.main --config configs/fever.yaml --mode routing
python -m src.main --config configs/fever.yaml --mode remix
python -m src.main --config configs/fever.yaml --mode eval
