#!/usr/bin/env bash
set -e

export VLLM_MODEL=$(python3 -c "import yaml;print(yaml.safe_load(open('scripts/config.yaml'))['generation']['vllm_model'])")

docker compose -f docker-compose.backend.prod.yml up -d
