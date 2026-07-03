#!/bin/sh

# Friendly interactive launcher for SBS 2D-to-3D conversion.
# Run from any directory: sh sbs.sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    printf 'Error: virtual environment not found.\n' >&2
    printf 'Set one up first:\n' >&2
    printf '  python3 -m venv venv\n' >&2
    printf '  source venv/bin/activate\n' >&2
    printf '  python install_torch.py\n' >&2
    printf '  pip install -r requirements.txt\n' >&2
    exit 1
fi

DEFAULT_INPUT="$SCRIPT_DIR/input"
DEFAULT_OUTPUT="$SCRIPT_DIR/output"

printf '\nStereoSift — image and video SBS converter\n'
printf 'Press Enter to keep each default. Paths may contain spaces.\n\n'

printf 'Input file or directory [%s]: ' "$DEFAULT_INPUT"
IFS= read -r INPUT_PATH || exit 1
[ -n "$INPUT_PATH" ] || INPUT_PATH="$DEFAULT_INPUT"

if [ ! -e "$INPUT_PATH" ]; then
    printf 'Error: input does not exist:\n  %s\n' "$INPUT_PATH" >&2
    exit 1
fi

printf 'Output directory [%s]: ' "$DEFAULT_OUTPUT"
IFS= read -r OUTPUT_DIR || exit 1
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="$DEFAULT_OUTPUT"

if ! mkdir -p "$OUTPUT_DIR"; then
    printf 'Error: could not create output directory:\n  %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi

printf '\nInput:  %s\nOutput: %s\n' "$INPUT_PATH" "$OUTPUT_DIR"
printf 'Start conversion? [Y/n]: '
IFS= read -r CONFIRM || exit 1
case $CONFIRM in
    n|N|no|NO|No) printf 'Cancelled.\n'; exit 0 ;;
esac

printf '\nStarting conversion. Press Ctrl-C once to stop safely.\n\n'
cd "$SCRIPT_DIR" || exit 1

exec "$PYTHON" "$SCRIPT_DIR/convert.py" \
    --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-res -1 \
    --yes
