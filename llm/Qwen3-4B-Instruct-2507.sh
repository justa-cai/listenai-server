vllm serve Qwen3-4B-Instruct-2507 --port 9500 --max-model-len 28816 --tool-call-parser hermes --enable-auto-tool-choice --swap-space 8 --enforce-eager --max-num-seqs 8 --gpu-memory-utilization 0.50
