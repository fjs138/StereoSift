#!/bin/sh

# Friendly launcher for SBS 2D-to-3D conversion.
# Works from any current directory: ./sbs.sh or sh sbs.sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$SCRIPT_DIR
if [ ! -x "$PROJECT_DIR/venv/bin/python" ]; then
    PROJECT_DIR="$HOME/Projects/StereoSift"
fi
PYTHON="$PROJECT_DIR/venv/bin/python"

DEFAULT_INPUT="$HOME/Downloads/dump"
DEFAULT_OUTPUT="/Volumes/SANDISK512/dump_sbs"

if [ ! -x "$PYTHON" ]; then
    printf 'Error: project Python was not found at:\n  %s\n' "$PYTHON" >&2
    printf 'Create the virtual environment and install requirements first.\n' >&2
    exit 1
fi

printf '\nStereoSift — image and video SBS converter\n'
printf 'Press Enter to keep each default. Paths may contain spaces.\n\n'

printf 'Input file or directory [%s]: ' "$DEFAULT_INPUT"
IFS= read -r INPUT_PATH || exit 1
[ -n "$INPUT_PATH" ] || INPUT_PATH=$DEFAULT_INPUT

if [ ! -e "$INPUT_PATH" ]; then
    printf 'Error: input does not exist:\n  %s\n' "$INPUT_PATH" >&2
    exit 1
fi

printf 'Output directory [%s]: ' "$DEFAULT_OUTPUT"
IFS= read -r OUTPUT_DIR || exit 1
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR=$DEFAULT_OUTPUT

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
cd "$PROJECT_DIR" || exit 1

exec "$PYTHON" "$PROJECT_DIR/convert.py" \
    --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-res -1 \
    --yes
