#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  collect_host_cpu.sh --container NAME_OR_ID --output FILE [--interval SECONDS]

Options:
  --container NAME_OR_ID  Docker container name or ID.
  --output FILE           New CSV file to write. Existing files are not overwritten.
  --interval SECONDS      Sampling interval in seconds. Default: 1.
  -h, --help              Show this help message.

Example:
  collect_host_cpu.sh \
    --container vllm-container \
    --output /home/zkx/mm_cpu/results/qwen72b_cpu.csv \
    --interval 1

Press Ctrl+C to stop sampling.
EOF
}

container=""
output=""
interval="1"

while (($# > 0)); do
    case "$1" in
        --container)
            container=${2:-}
            shift 2
            ;;
        --output)
            output=${2:-}
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

if [[ -z "$container" || -z "$output" ]]; then
    echo "--container and --output are required." >&2
    usage >&2
    exit 2
fi

if ! awk -v value="$interval" 'BEGIN { exit !(value > 0) }'; then
    echo "--interval must be greater than zero." >&2
    exit 2
fi

command -v docker >/dev/null 2>&1 || {
    echo "docker was not found in PATH." >&2
    exit 1
}

running=$(docker inspect --format '{{.State.Running}}' "$container")
if [[ "$running" != "true" ]]; then
    echo "Container is not running: $container" >&2
    exit 1
fi

if [[ -e "$output" ]]; then
    echo "Output file already exists: $output" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$output")"
printf '%s\n' \
    'timestamp,container,cpu_percent,effective_cores,memory_usage,pids' \
    > "$output"

trap 'echo; echo "Stopped. Results: $output" >&2; exit 0' INT TERM

echo "Collecting Docker CPU usage every ${interval}s." >&2
echo "Container: $container" >&2
echo "Output:    $output" >&2
echo "Press Ctrl+C to stop." >&2

while true; do
    stats=$(docker stats --no-stream \
        --format '{{.CPUPerc}}|{{.MemUsage}}|{{.PIDs}}' \
        "$container")
    IFS='|' read -r cpu_percent memory_usage pids <<< "$stats"
    cpu_value=${cpu_percent%\%}
    effective_cores=$(awk -v cpu="$cpu_value" 'BEGIN { printf "%.2f", cpu / 100 }')
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    printf '"%s","%s",%s,%s,"%s",%s\n' \
        "$timestamp" \
        "$container" \
        "$cpu_value" \
        "$effective_cores" \
        "$memory_usage" \
        "$pids" \
        >> "$output"

    sleep "$interval"
done
