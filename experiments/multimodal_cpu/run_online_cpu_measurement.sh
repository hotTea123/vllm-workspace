#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_online_cpu_measurement.sh --container NAME_OR_ID --model NAME \
    --image FILE --output-prefix PATH [options]

Options:
  --container NAME_OR_ID  Container measured by docker stats. Required.
  --model NAME            Model name served by the API. Required.
  --image FILE            Image sent in every request. Required.
  --output-prefix PATH    Prefix for requests JSONL and host CPU CSV. Required.
  --base-url URL          vLLM base URL. Default: http://127.0.0.1:12346.
  --prompt TEXT           User prompt. Default: Describe this image.
  --warmups N             Warmup requests before CPU sampling. Default: 4.
  --repeats N             Measured requests. Default: 10.
  --output-tokens N       Fixed generated token count. Default: 100.
  --interval SECONDS      Host CPU sampling interval. Default: 1.
  -h, --help              Show this help message.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd -- "$script_dir/../.." && pwd)

container=""
model=""
image=""
output_prefix=""
base_url="http://127.0.0.1:12346"
prompt="Describe this image."
warmups="4"
repeats="10"
output_tokens="100"
interval="1"

while (($# > 0)); do
    case "$1" in
        --container)
            container=${2:-}
            shift 2
            ;;
        --model)
            model=${2:-}
            shift 2
            ;;
        --image)
            image=${2:-}
            shift 2
            ;;
        --output-prefix)
            output_prefix=${2:-}
            shift 2
            ;;
        --base-url)
            base_url=${2:-}
            shift 2
            ;;
        --prompt)
            prompt=${2:-}
            shift 2
            ;;
        --warmups)
            warmups=${2:-}
            shift 2
            ;;
        --repeats)
            repeats=${2:-}
            shift 2
            ;;
        --output-tokens)
            output_tokens=${2:-}
            shift 2
            ;;
        --interval)
            interval=${2:-}
            shift 2
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

if [[ -z "$container" || -z "$model" || -z "$image" || -z "$output_prefix" ]]; then
    echo "--container, --model, --image, and --output-prefix are required." >&2
    usage >&2
    exit 2
fi

if [[ ! -f "$image" ]]; then
    echo "Image does not exist: $image" >&2
    exit 1
fi

requests_output="${output_prefix}.requests.jsonl"
cpu_output="${output_prefix}.host_cpu.csv"
run_id=$(date '+%Y%m%d%H%M%S')-$$
python_bin=${PYTHON:-python}

cd "$workspace"

echo "Warming up the online service (${warmups} requests)." >&2
"$python_bin" -m experiments.multimodal_cpu.run_online_cpu \
    --base-url "$base_url" \
    --model "$model" \
    --image "$image" \
    --prompt "$prompt" \
    --phase warmup \
    --requests "$warmups" \
    --output-tokens "$output_tokens" \
    --run-id "$run_id"

collector_pid=""
stop_collector() {
    if [[ -n "$collector_pid" ]] && kill -0 "$collector_pid" 2>/dev/null; then
        kill -TERM "$collector_pid"
        wait "$collector_pid" || true
    fi
    collector_pid=""
}
trap stop_collector EXIT
trap 'stop_collector; exit 130' INT TERM

bash "$script_dir/collect_host_cpu.sh" \
    --container "$container" \
    --output "$cpu_output" \
    --interval "$interval" &
collector_pid=$!

echo "Measuring ${repeats} requests with host CPU sampling enabled." >&2
"$python_bin" -m experiments.multimodal_cpu.run_online_cpu \
    --base-url "$base_url" \
    --model "$model" \
    --image "$image" \
    --prompt "$prompt" \
    --phase measure \
    --requests "$repeats" \
    --output-tokens "$output_tokens" \
    --output "$requests_output" \
    --run-id "$run_id"

stop_collector
trap - EXIT INT TERM

echo "Requests: $requests_output"
echo "Host CPU: $cpu_output"
