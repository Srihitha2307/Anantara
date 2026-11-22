#!/usr/bin/env bash
# Publish the app to a Hugging Face Space (Docker SDK).  Usage:  bash deploy/huggingface/push.sh <hf-username>/<space-name>
# Create the empty Space first at https://huggingface.co/new-space (SDK: Docker, blank). Log in with:  huggingface-cli login
set -euo pipefail
SPACE="${1:?usage: push.sh <hf-username>/<space-name>}"
TMP="$(mktemp -d)"
git clone "https://huggingface.co/spaces/${SPACE}" "$TMP/space"
rsync -a --delete --exclude .git --exclude notebooks --exclude docs --exclude tests --exclude data/cache --exclude '__pycache__' ./ "$TMP/space/"
cp deploy/huggingface/README.md "$TMP/space/README.md"
cd "$TMP/space" && git add -A && git commit -m "Deploy Anantara" && git push
echo "Live at https://${SPACE/\//-}.hf.space  (first build takes a few minutes)"
