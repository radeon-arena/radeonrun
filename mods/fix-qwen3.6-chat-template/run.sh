#!/bin/bash
set -e
# Mod: fix-qwen3.6-chat-template (SCAFFOLD)
# Installs a corrected chat template into the container workspace.
# The template itself is a placeholder.
cp chat_template.jinja "$WORKSPACE_DIR/fixed_chat_template.jinja"
echo "=======> to apply chat template, use --chat-template fixed_chat_template.jinja"
