#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  start_online_cpu_server.sh --model PATH [options] [-- VLLM_OPTIONS]

Options:
  --model PATH                 Model path or name. Required.
  --tensor-parallel-size N     Tensor parallel size. Default: 1.
  --max-model-len N            Maximum model length. Default: 16635.
  --host HOST                  Listen address. Default: 0.0.0.0.
  --port PORT                  Listen port. Default: 12346.
  -h, --help                   Show this help message.

The script always disables the MM processor cache, prefix cache, MM encoder
compilation, and MM encoder CUDAGraph. Arguments after -- are passed to vLLM
before these fixed isolation options.
EOF
}

model=""
tensor_parallel_size="1"
max_model_len="16635"
host="0.0.0.0"
port="12346"
extra_args=()

while (($# > 0)); do
    case "$1" in
        --model)
            model=${2:-}
            shift 2
            ;;
        --tensor-parallel-size)
            tensor_parallel_size=${2:-}
            shift 2
            ;;
        --max-model-len)
            max_model_len=${2:-}
            shift 2
            ;;
        --host)
            host=${2:-}
            shift 2
            ;;
        --port)
            port=${2:-}
            shift 2
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$model" ]]; then
    echo "--model is required." >&2
    usage >&2
    exit 2
fi

command -v vllm >/dev/null 2>&1 || {
    echo "vllm was not found in PATH." >&2
    exit 1
}

exec vllm serve "$model" \
    --host "$host" \
    --port "$port" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --max-model-len "$max_model_len" \
    "${extra_args[@]}" \
    --limit-mm-per-prompt '{"image":1}' \
    --mm-processor-cache-gb 0 \
    --no-enable-prefix-caching \
    --compilation-config \
        '{"compile_mm_encoder":false,"cudagraph_mm_encoder":false}' \
    --enable-mm-processor-stats \
    --enable-request-id-headers
