# CLAIRAI: Real-Time Probe Activation Visualization

A real-time visualization system for monitoring neural network probe activations during LLM inference. This project captures activation scores from vLLM models via CUDA IPC and streams them to a web frontend for interactive visualization.

## Overview

CLAIRAI consists of three main components:

1. **vLLM Probe Plugin**: A CUDA IPC-based probe that hooks into vLLM's model execution, capturing activation scores from a target layer and writing them directly to GPU memory via a high-performance ring buffer.

2. **Proxy Server**: A FastAPI server that:
   - Reads probe activation data from the CUDA IPC ring buffer
   - Proxies chat completion requests to vLLM
   - Streams both generated tokens and probe scores to the frontend in real-time

3. **Frontend Visualization**: A React application that displays:
   - Token-by-token probe activation heatmaps
   - Interactive line charts showing activation patterns
   - Real-time streaming visualization as tokens are generated

## Architecture

```
┌─────────────────┐
│  vLLM Server    │  (Port 8000)
│  + Probe Plugin │  ────┐
│                 │      │ CUDA IPC
└─────────────────┘      │ Ring Buffer
                         │
┌─────────────────┐      │
│  Proxy Server   │  ◄───┘
│  (FastAPI)      │  ────┐
│  Port 6969      │      │ HTTP/SSE
└─────────────────┘      │
                         │
┌─────────────────┐      │
│  Frontend       │  ◄───┘
│  (React/Vite)   │
│  Port 5173      │
└─────────────────┘
```

## Prerequisites

- **Python 3.8+** with CUDA support
- **CUDA 11.x or 12.x** (match your system)
- **NVIDIA GPU** with sufficient VRAM
- **Node.js 18+** (for frontend)
- **vLLM** installed and working

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `vllm` - LLM serving framework
- `torch` - PyTorch with CUDA support
- `fastapi`, `uvicorn` - Proxy server
- `httpx` - HTTP client for vLLM
- `cupy-cuda12x` - GPU memory operations (adjust version to match your CUDA version)

**Note**: If you have CUDA 11.x, use `cupy-cuda11x` instead of `cupy-cuda12x`.

### 2. Build the CUDA Extension

The probe plugin includes a custom CUDA kernel for high-performance IPC writes:

```bash
cd src/
pip install -e .
```

This will compile the `vllm_ipc` CUDA extension. You should see output about building with `nvcc`.

### 3. Install Frontend Dependencies

```bash
cd frontend/sae-vis/
npm install
```

## Configuration

### Environment Variables

Before running vLLM, set these environment variables:

```bash
export TARGET_LAYER=19              # Which transformer layer to probe
export PROBE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Model identifier for probe directory
```

The probe files should be located at:
```
src/probes/{MODEL_NAME}/*.json
```

For example: `src/probes/Llama-3.1-8B-Instruct/enc_19_*.json`

### Proxy Configuration

The proxy server is configured in `src/proxy.py`. Default settings:
- **vLLM URL**: `http://localhost:8000/v1/chat/completions`
- **Proxy Port**: `6969`
- **IPC Path**: `/tmp/vllm_probe.ipc`
- **Meta Path**: `/tmp/vllm_probe_meta.json`

### Frontend Configuration

In `frontend/App.jsx` or `frontend/sae-vis/src/App.jsx`, ensure:
- **Proxy URL**: `http://localhost:6969/v1/chat/completions`
- **Model Name**: Matches your vLLM model ID exactly

## Running the System

You need **three terminal windows** to run all components.

### Terminal 1: Start vLLM Server

```bash
# Set environment variables
export TARGET_LAYER=19
export PROBE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Start vLLM with the probe plugin
vllm serve $PROBE_MODEL \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
```

**Look for these log messages:**
- `>> [PLUGIN] Initializing Probe for: ...`
- `>> [IPC] Ring Buffer Active: ... MB`
- `>> [IPC] Handle exported to /tmp/vllm_probe.ipc`

The probe plugin will automatically register when vLLM starts.

### Terminal 2: Start Proxy Server

```bash
cd /root/clairai
python src/proxy.py
```

**Look for:**
- `🚀 Starting vLLM Probe Proxy on port 6969...`
- `[IPC] ⏳ Waiting for IPC file at /tmp/vllm_probe.ipc...`
- `[IPC] 🚀 Connected. Ring Size: ...`

The proxy will wait for vLLM to create the IPC handle, then start monitoring the ring buffer.

### Terminal 3: Start Frontend

```bash
cd frontend/sae-vis/
npm run dev
```

The frontend will start on `http://localhost:5173` (or another port if 5173 is busy).

## Usage

1. Open your browser to `http://localhost:5173`
2. Enter a prompt in the chat interface
3. Watch as tokens stream in with real-time probe activation visualizations:
   - **Heatmap**: Token background colors indicate activation intensity
   - **Line Chart**: Shows activation patterns across tokens
   - **Hover**: Hover over tokens to see detailed activation values

## Project Structure

```
clairai/
├── src/
│   ├── proxy.py              # FastAPI proxy server
│   ├── sidecar.py            # Standalone IPC reader (optional)
│   ├── setup.py              # Package setup with CUDA extension
│   ├── vllm_probe/
│   │   ├── __init__.py       # Plugin registration
│   │   ├── model.py          # Probe hook implementation
│   │   └── ipc_writer.cu     # CUDA kernel for IPC writes
│   └── probes/               # Probe weight files
│       └── {MODEL_NAME}/
│           └── *.json
├── frontend/
│   ├── App.jsx               # Standalone frontend (legacy)
│   └── sae-vis/              # Vite + React frontend
│       ├── src/
│       │   └── App.jsx       # Main React component
│       └── package.json
├── benchmarking/             # Performance benchmarking tools
├── requirements.txt          # Python dependencies
└── README.md
```

## How It Works

### vLLM Probe Plugin

1. **Initialization**: When vLLM loads a model, the plugin:
   - Loads probe weights from JSON files
   - Allocates a GPU ring buffer using `cudaMalloc` (IPC-compatible)
   - Creates an IPC handle and exports it to `/tmp/vllm_probe.ipc`
   - Attaches a forward hook to the target transformer layer

2. **During Inference**: 
   - The hook captures activations from the target layer
   - Computes probe scores: `probe_score = ReLU(activations @ probe_weights)`
   - Writes scores to the ring buffer using the CUDA kernel
   - Updates the sequence counter atomically

### Proxy Server

1. **IPC Monitoring**: Background thread continuously:
   - Reads from the GPU ring buffer via IPC
   - Tracks stream IDs to match probe data with requests
   - Buffers probe scores in memory

2. **Request Handling**:
   - Receives chat completion requests from frontend
   - Proxies requests to vLLM (forces streaming mode)
   - Matches incoming tokens with buffered probe data
   - Streams both tokens and probe scores via Server-Sent Events (SSE)

### Frontend

1. **Streaming**: Receives SSE stream with:
   - `token` events: Generated text tokens
   - `probe_update` events: Incremental probe score updates
   - `probe_final` events: Final probe scores after completion
   - `done` event: Complete response with all data

2. **Visualization**:
   - Renders tokens with background colors based on activation intensity
   - Updates line chart in real-time as tokens arrive
   - Supports probe selection and hover interactions

## Troubleshooting

### vLLM Issues

**CUDA out of memory:**
- Reduce `--gpu-memory-utilization` (e.g., `0.85` instead of `0.9`)
- The ring buffer allocates persistent VRAM that vLLM must account for

**ModuleNotFoundError: No module named 'vllm_ipc':**
- Make sure you ran `pip install -e src/` to build the CUDA extension
- Verify you're using the same Python environment for vLLM

**Probe plugin not loading:**
- Check that `TARGET_LAYER` and `PROBE_MODEL` environment variables are set
- Verify probe files exist in `src/probes/{MODEL_NAME}/`
- Look for `>> [PLUGIN]` log messages in vLLM output

### Proxy Issues

**Proxy can't connect to IPC:**
- Ensure vLLM is running first
- Check that `/tmp/vllm_probe.ipc` exists (created by vLLM)
- Verify proxy logs show `[IPC] 🚀 Connected`

**Probe data not matching requests:**
- The matching algorithm uses prompt length estimation
- Very similar prompts may cause mismatches
- Check proxy logs for sequence numbers

**Connection refused to vLLM:**
- Verify vLLM is running on port 8000
- Check `VLLM_URL` in `src/proxy.py` matches your vLLM configuration

### Frontend Issues

**CORS errors:**
- Proxy already includes CORS middleware for `localhost:5173` and `localhost:3000`
- If using a different port, update `allow_origins` in `src/proxy.py`

**No probe visualizations:**
- Check browser console for errors
- Verify proxy is streaming probe data (check Network tab for SSE events)
- Ensure model name in frontend matches vLLM model ID exactly

**Empty responses:**
- Verify proxy URL is correct: `http://localhost:6969/v1/chat/completions`
- Check that vLLM is responding to requests

## Performance Notes

- **CUDA Graphs**: The IPC kernel is designed to work with CUDA Graphs, avoiding Python interpreter overhead
- **Zero-Copy**: Probe data flows from GPU → GPU (IPC) → GPU (CuPy) → CPU (NumPy) with minimal copying
- **Streaming**: Both tokens and probe scores stream incrementally, providing real-time feedback
- **Latency**: Typical latency added by probe system: <1ms per token

## Development

### Building CUDA Extension

To rebuild the CUDA extension after changes:

```bash
cd src/
pip install -e . --force-reinstall --no-deps
```

### Adding New Probes

1. Place probe JSON files in `src/probes/{MODEL_NAME}/`
2. Format: `{"vector": [list of weights]}`
3. Probes are automatically loaded by layer index

### Modifying the Proxy

The proxy is in `src/proxy.py`. Key components:
- `IPCMonitor`: Background thread reading from ring buffer
- `proxy_chat_completions`: FastAPI endpoint handling requests
- `find_matching_stream`: Logic for matching requests with probe data

### Frontend Development

The main frontend is in `frontend/sae-vis/`. To develop:

```bash
cd frontend/sae-vis/
npm run dev  # Development server with hot reload
npm run build  # Production build
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
