#!/bin/bash
# Run LLM Fine-Tuning Web UI with GPU (Native Python - No Docker)
# This bypasses Docker GPU detection issues

set -e

echo "============================================================"
echo "🚀 LLM Fine-Tuning Web UI - Native GPU Mode"
echo "============================================================"
echo ""

# Check GPU
echo "Checking GPU..."
python3 -c "import torch; print(f'✅ GPU: {torch.cuda.is_available()}'); print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import gradio; print('✅ Gradio: OK')" 2>/dev/null || { echo "❌ Gradio not found. Install with: pip install gradio python-multipart"; exit 1; }
python3 -c "import transformers; print('✅ Transformers: OK')" 2>/dev/null || { echo "❌ Transformers not found. Install with: pip install -r requirements.txt"; exit 1; }
echo ""

# Start Web UI
echo "Starting Web UI..."
echo "🌐 Access at: http://localhost:7860"
echo "Press Ctrl+C to stop"
echo "============================================================"
echo ""

cd "$(dirname "$0")"
python3 app/gradio_app.py
