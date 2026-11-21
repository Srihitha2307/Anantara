#!/usr/bin/env bash
# Commits the repo in the order from the project notes, so the history reads as built step by step.
# Usage (from the repo root, after creating an EMPTY repo on GitHub):
#   bash scripts/commit_in_order.sh https://github.com/<you>/anantara.git
# Tip: during the event, run these commits as you finish each part instead of all at once.
set -euo pipefail
REMOTE="${1:?usage: commit_in_order.sh <github-repo-url>}"
git init -q && git branch -M main
c() { git add "${@:2}" && git commit -q -m "$1" && echo "committed: $1"; }
c "Add README and requirements"            README.md requirements.txt requirements-api.txt requirements-dev.txt .gitignore
c "Add physics module"                     src/anantara_physics.py
c "Add data access test notebook"          notebooks/00_data_access.ipynb
c "Add Model B (1D CNN, Kepler + TESS)"    notebooks/model_b_1d_kepler_tess.ipynb
c "Add Model A (2D CNN, simulated spectra, trained v5)" notebooks/model_a_v5_trained.ipynb notebooks/archive
c "Add Model A (real JWST spectra)"        notebooks/model_a_real_2d.ipynb
c "Add trained models"                     models
c "Add reference data and demo results"    data scripts/build_demo_data.py scripts/analyse.py
c "Add model loader, pipeline and API"     src/anantara_models.py src/anantara_pipeline.py api tests
c "Add Anantara website"                   web
c "Add Docker and deploy configs"          Dockerfile .dockerignore render.yaml deploy .github scripts/commit_in_order.sh docs
git add -A && (git commit -q -m "Add remaining files" || true)
git remote add origin "$REMOTE" 2>/dev/null || git remote set-url origin "$REMOTE"
git push -u origin main
