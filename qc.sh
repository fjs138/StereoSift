#!/bin/sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    printf 'Error: virtual environment not found at %s\n' "$PYTHON" >&2
    exit 1
fi

printf '\nStereoSift visual judgment\n'
printf 'Originals are copied by default, never moved.\n\n'
printf 'Input image or directory [%s]: ' "$SCRIPT_DIR/input"
IFS= read -r INPUT_PATH || exit 1
[ -n "$INPUT_PATH" ] || INPUT_PATH="$SCRIPT_DIR/input"

printf 'Results directory [%s]: ' "$SCRIPT_DIR/output/qc"
IFS= read -r OUTPUT_PATH || exit 1
[ -n "$OUTPUT_PATH" ] || OUTPUT_PATH="$SCRIPT_DIR/output/qc"

printf 'Vision backend URL (leave blank for basic non-AI checks): '
IFS= read -r BACKEND_URL || exit 1

set -- "$PYTHON" "$SCRIPT_DIR/qc_pipeline.py" --input "$INPUT_PATH" --output-dir "$OUTPUT_PATH"
if [ -n "$BACKEND_URL" ]; then
    set -- "$@" --backend-url "$BACKEND_URL"
fi

cd "$SCRIPT_DIR" || exit 1
exec "$@"
