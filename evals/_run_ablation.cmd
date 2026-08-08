@echo off
cd /d C:\Users\mathe\Documents\Code\I.S.A.A.C
set ISAAC_MODEL_NAME=gpt-oss:120b-cloud
set ISAAC_OLLAMA_LIGHT_MODEL=gpt-oss:120b-cloud
set ISAAC_OLLAMA_HEAVY_MODEL=gpt-oss:120b-cloud
set OLLAMA_LIGHT_MODEL=gpt-oss:120b-cloud
set OLLAMA_HEAVY_MODEL=gpt-oss:120b-cloud
set ISAAC_FAST_MODEL=gpt-oss:120b-cloud
set ISAAC_STRONG_MODEL=gpt-oss:120b-cloud
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe evals\run_ablation.py --trials 3 --warmup 2 --per-category 2 --task-timeout 180 --out evals\results\ablation_1.5.0.json
echo ABLATION_EXIT=%ERRORLEVEL%
