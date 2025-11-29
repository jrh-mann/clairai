#!/usr/bin/env python3
"""Minimal FastAPI proxy server that forwards requests to vLLM."""

import asyncio
import ctypes
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import copy

import cupy as cp
import numpy as np
import aiohttp
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

# Configuration
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "6969"))
IPC_PATH = "/tmp/vllm_probe.ipc"
META_PATH = "/tmp/vllm_probe_meta.json"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IPCReader:
    """Reads probe data from IPC shared memory with minimal overhead."""
    
    # Cache cudart library to avoid repeated loading
    _cudart: Optional[ctypes.CDLL] = None
    
    @classmethod
    def _get_cudart(cls) -> ctypes.CDLL:
        """Get or load the cudart library (cached)."""
        if cls._cudart is None:
            cls._cudart = ctypes.CDLL('libcudart.so')
        return cls._cudart
    
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.gpu_mem_ptr: Optional[int] = None
        self.ring_size: int = 0
        self.slot_size_bytes: int = 0
        self.num_probes: int = 0
        self.data_base_ptr: int = 0
        self.last_seq: int = 0
        self.latest_data: Optional[Dict[str, Any]] = None
        # Store probe data by request_id: {request_id: {token_idx: [probe_values]}}
        self.probe_data: Dict[str, Dict[int, List[float]]] = {}
        self._lock = threading.Lock()
        # Track active requests for matching probe data
        self._active_requests: deque = deque(maxlen=100)  # Keep last 100 requests
    
    def _read_head(self) -> int:
        """Read head sequence number from GPU memory."""
        if self.gpu_mem_ptr is None:
            return 0
        
        cp.cuda.Device().synchronize()
        cudart = self._get_cudart()
        host_buf = (ctypes.c_uint64 * 1)()
        cudart.cudaMemcpy(
            ctypes.byref(host_buf),
            ctypes.c_void_p(self.gpu_mem_ptr),
            8,
            ctypes.c_int(2)  # cudaMemcpyDeviceToHost
        )
        return int(host_buf[0])
    
    def _read_slot(self, seq: int) -> Optional[Dict[str, Any]]:
        """Read a single slot from the ring buffer."""
        slot_idx = seq % self.ring_size
        src_ptr = self.data_base_ptr + (slot_idx * self.slot_size_bytes)
        
        cp.cuda.Device().synchronize()
        cudart = self._get_cudart()
        
        # Read batch_id (8 bytes at offset 0)
        host_batch_id = (ctypes.c_uint64 * 1)()
        cudart.cudaMemcpy(
            ctypes.byref(host_batch_id),
            ctypes.c_void_p(src_ptr),
            8,
            ctypes.c_int(2)
        )
        batch_id = int(host_batch_id[0])
        
        # Read num_requests (4 bytes at offset 8)
        host_num_requests = (ctypes.c_int32 * 1)()
        cudart.cudaMemcpy(
            ctypes.byref(host_num_requests),
            ctypes.c_void_p(src_ptr + 8),
            4,
            ctypes.c_int(2)
        )
        num_requests = int(host_num_requests[0])
        
        if num_requests == 0:
            return None
        
        # Read query_start_loc
        query_start_loc_size = num_requests + 1
        host_query_start_loc = (ctypes.c_int32 * query_start_loc_size)()
        cudart.cudaMemcpy(
            ctypes.byref(host_query_start_loc),
            ctypes.c_void_p(src_ptr + 12),
            query_start_loc_size * 4,
            ctypes.c_int(2)
        )
        query_start_loc = np.array([int(host_query_start_loc[i]) for i in range(query_start_loc_size)])
        num_tokens = int(query_start_loc[-1])
        
        # Read probe_scores
        scores_offset = 12 + (query_start_loc_size * 4)
        scores_size = num_tokens * self.num_probes
        
        host_scores = (ctypes.c_float * scores_size)()
        cudart.cudaMemcpy(
            ctypes.byref(host_scores),
            ctypes.c_void_p(src_ptr + scores_offset),
            scores_size * 4,
            ctypes.c_int(2)
        )
        scores_array = np.array([float(host_scores[i]) for i in range(scores_size)])
        probe_scores = scores_array.reshape(num_tokens, self.num_probes)
        
        # Extract per-request scores
        requests = []
        for req_idx in range(num_requests):
            start_token = int(query_start_loc[req_idx])
            end_token = int(query_start_loc[req_idx + 1])
            request_scores = probe_scores[start_token:end_token]
            
            requests.append({
                "request_idx": req_idx,
                "start_token": start_token,
                "end_token": end_token,
                "scores": request_scores.tolist() if request_scores.size > 0 else [],
                "avg_score": float(request_scores.mean()) if request_scores.size > 0 else 0.0
            })
        
        return {
            "seq": seq,
            "batch_id": batch_id,
            "num_requests": num_requests,
            "requests": requests
        }
    
    def _reader_loop(self):
        """Main reading loop running in background thread."""
        # Wait for IPC handle
        logger.info("Waiting for IPC handle...")
        while self.running and not os.path.exists(IPC_PATH):
            time.sleep(0.5)
        
        if not self.running:
            return
        
        try:
            # Load metadata
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            
            self.ring_size = meta["ring_size"]
            self.slot_size_bytes = meta["slot_size_bytes"]
            self.num_probes = meta["num_probes"]
            
            # Open IPC handle
            with open(IPC_PATH, "rb") as f:
                handle_bytes = f.read()
            
            self.gpu_mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
            self.data_base_ptr = self.gpu_mem_ptr + 128
            
            logger.info(f"IPC connected. Ring size: {self.ring_size}, Probes: {self.num_probes}")
            
            # Initialize from current head
            self.last_seq = self._read_head()
            logger.info(f"Starting from sequence: {self.last_seq}")
            
            # Main reading loop
            while self.running:
                curr_seq = self._read_head()
                
                if curr_seq > self.last_seq:
                    delta = curr_seq - self.last_seq
                    
                    # Handle wrap-around
                    start = curr_seq - 1 if delta > self.ring_size else self.last_seq
                    
                    # Read new slots
                    for seq in range(start, curr_seq):
                        slot_data = self._read_slot(seq)
                        if slot_data:
                            with self._lock:
                                self.latest_data = slot_data
                                # Store probe data for active requests
                                self._store_probe_data(slot_data)
                            logger.debug(f"Read seq {seq}: batch_id={slot_data['batch_id']}, requests={slot_data['num_requests']}")
                    
                    self.last_seq = curr_seq
                else:
                    time.sleep(0.001)  # Minimal sleep when no updates
                    
        except Exception as e:
            logger.error(f"IPC reader error: {e}", exc_info=True)
        finally:
            if self.gpu_mem_ptr is not None:
                try:
                    cp.cuda.runtime.ipcCloseMemHandle(self.gpu_mem_ptr)
                except Exception:
                    pass
    
    def start(self):
        """Start the IPC reader thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        logger.info("IPC reader started")
    
    def stop(self):
        """Stop the IPC reader thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("IPC reader stopped")
    
    def _store_probe_data(self, slot_data: Dict[str, Any]):
        """Store probe data for active requests."""
        # Match probe data to active requests by order
        # This assumes probe data arrives in the same order as requests
        requests = slot_data.get("requests", [])
        if not requests:
            return
        
        # Get active requests (most recent first)
        active = list(self._active_requests)
        if not active:
            return
        
        # Match each request in the batch to an active request
        for req_data in requests:
            req_idx = req_data["request_idx"]
            scores = req_data.get("scores", [])
            start_token = req_data.get("start_token", 0)
            
            # Match to active request (simple: match by position in batch)
            # For single requests, match to most recent active request
            # Note: active is guaranteed to be non-empty due to check above
            # Use the most recent request that doesn't have probe data yet
            matched = False
            for request_id in reversed(active):
                if request_id not in self.probe_data:
                    # Store token-to-probe mapping
                    # Note: probe data includes all tokens (prompt + generated)
                    # We'll match streaming tokens starting from the first token
                    # The start_token offset can be used to skip prompt tokens if needed
                    token_probes: Dict[int, List[float]] = {}
                    for local_idx, token_scores in enumerate(scores):
                        if isinstance(token_scores, list):
                            # Store with global token index (start_token + local_idx)
                            # But also store with local index for easier matching
                            global_idx = start_token + local_idx
                            token_probes[global_idx] = token_scores
                            # Also store with local index (0-based for this request)
                            token_probes[local_idx] = token_scores
                    if token_probes:
                        self.probe_data[request_id] = {
                            "token_probes": token_probes,
                            "start_token": start_token,
                            "num_tokens": len(scores)
                        }
                        matched = True
                        break
            
            if not matched and len(active) > req_idx:
                # Fallback: match by index
                request_id = active[-(req_idx + 1)]
                token_probes: Dict[int, List[float]] = {}
                for local_idx, token_scores in enumerate(scores):
                    if isinstance(token_scores, list):
                        global_idx = start_token + local_idx
                        token_probes[global_idx] = token_scores
                        token_probes[local_idx] = token_scores
                if token_probes:
                    self.probe_data[request_id] = {
                        "token_probes": token_probes,
                        "start_token": start_token,
                        "num_tokens": len(scores)
                    }
    
    def register_request(self, request_id: str):
        """Register an active request for probe data matching."""
        with self._lock:
            self._active_requests.append(request_id)
    
    def get_probe_data(self, request_id: str) -> Optional[Dict[int, List[float]]]:
        """Get probe data for a request (thread-safe). Returns token_probes dict."""
        with self._lock:
            data = self.probe_data.get(request_id)
            if data and isinstance(data, dict) and "token_probes" in data:
                return data["token_probes"]
            return None
    
    def clear_probe_data(self, request_id: str):
        """Clear probe data for a request after completion."""
        with self._lock:
            self.probe_data.pop(request_id, None)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get latest probe data (thread-safe)."""
        with self._lock:
            return self.latest_data


# Global IPC reader instance
ipc_reader = IPCReader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage IPC reader lifecycle."""
    ipc_reader.start()
    yield
    ipc_reader.stop()


app = FastAPI(title="vLLM Probe Proxy", lifespan=lifespan)


def _prepare_headers(request: Request) -> Dict[str, str]:
    """Prepare headers for forwarding, excluding host and content-length."""
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    return headers


async def _handle_error_response(response: aiohttp.ClientResponse) -> JSONResponse:
    """Handle error responses from vLLM."""
    try:
        error_data = await response.json()
    except Exception:
        error_text = await response.text()
        error_data = {"error": error_text}
    
    return JSONResponse(
        status_code=response.status,
        content=error_data
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "vllm_url": f"{VLLM_URL}/v1"}


@app.get("/probes/latest")
async def get_latest_probes() -> Dict[str, Any]:
    """Get latest probe data from IPC."""
    latest = ipc_reader.get_latest()
    if latest is None:
        return {"status": "no_data", "data": None}
    return {"status": "ok", "data": latest}


def _parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single SSE line and return JSON data."""
    line = line.strip()
    if not line.startswith("data: "):
        return None
    
    data_str = line[6:]  # Remove 'data: ' prefix
    if data_str == "[DONE]":
        return {"done": True}
    
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def _inject_probe_values(
    chunk_data: Dict[str, Any],
    probe_data: Optional[Dict[int, List[float]]],
    token_index: int
) -> Dict[str, Any]:
    """Inject probe values into a streaming chunk.
    
    Args:
        chunk_data: The SSE chunk data from vLLM
        probe_data: Dict mapping token index to probe values list
        token_index: Current token index (0-based for generated tokens)
    
    Returns:
        Modified chunk_data with probe_values added
    """
    if probe_data is None:
        return chunk_data
    
    # Try to find probe values for this token
    # First try local index (0-based for generated tokens)
    probe_values = probe_data.get(token_index)
    
    # If not found, try to find the closest available token
    # (in case of timing issues or prompt token offset)
    if probe_values is None and probe_data:
        # Find the closest token index
        available_indices = sorted(probe_data.keys())
        if available_indices:
            # Use the token at or just after our index
            for idx in available_indices:
                if idx >= token_index:
                    probe_values = probe_data[idx]
                    break
            # If no token >= token_index, use the last one
            if probe_values is None:
                probe_values = probe_data[available_indices[-1]]
    
    if probe_values is None:
        return chunk_data
    
    # Create a copy to avoid modifying original
    result = copy.deepcopy(chunk_data)
    
    # Add probe values to choices
    if "choices" in result and isinstance(result["choices"], list):
        for choice in result["choices"]:
            if "delta" in choice:
                choice["delta"]["probe_values"] = probe_values
            elif "message" in choice:
                choice["message"]["probe_values"] = probe_values
    
    return result


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """Forward chat completion requests to vLLM with probe value injection."""
    body = await request.json()
    headers = _prepare_headers(request)
    
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())
    ipc_reader.register_request(request_id)
    
    # Track token index for this request
    token_index = 0
    
    # For streaming, we need to keep the session alive until the generator finishes
    # So we create it without async with and manage it manually
    session = aiohttp.ClientSession()
    try:
        vllm_response = await session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=body,
            headers=headers
        )
        
        if vllm_response.status != status.HTTP_200_OK:
            ipc_reader.clear_probe_data(request_id)
            vllm_response.close()
            await session.close()
            return await _handle_error_response(vllm_response)
        
        if not body.get("stream", True):
            # Non-streaming: inject probe values if available
            response_data = await vllm_response.json()
            probe_data = ipc_reader.get_probe_data(request_id)
            
            if probe_data and "choices" in response_data:
                for choice_idx, choice in enumerate(response_data["choices"]):
                    if "message" in choice:
                        # For non-streaming, attach all probe values
                        # Get the last token's probe values as a summary
                        if probe_data:
                            available_indices = sorted(probe_data.keys())
                            if available_indices:
                                # Use the last token's probe values
                                last_idx = available_indices[-1]
                                choice["message"]["probe_values"] = probe_data[last_idx]
                                # Also include all probe values indexed by token
                                choice["message"]["all_probe_values"] = {
                                    str(k): v for k, v in probe_data.items()
                                }
            
            ipc_reader.clear_probe_data(request_id)
            vllm_response.close()
            await session.close()
            return response_data
        
        # Streaming: parse SSE and inject probe values
        # Keep references to session and response so they stay alive during streaming
        async def generate():
            nonlocal token_index
            buffer = ""
            probe_data = None
            chunk_count = 0
            _session = session  # Keep session reference
            _response = vllm_response  # Keep response reference
            
            try:
                logger.debug(f"Starting stream for request {request_id}")
                async for chunk_bytes in _response.content.iter_any():
                    if not chunk_bytes:
                        continue
                    
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.debug(f"Received first chunk from vLLM for request {request_id}")
                    buffer += chunk_bytes.decode('utf-8', errors='ignore')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        chunk_data = _parse_sse_line(line)
                        if chunk_data is None:
                            # Not a data line, forward as-is with proper SSE format
                            yield f"{line}\n\n".encode('utf-8')
                            continue
                        
                        if chunk_data.get("done"):
                            # End of stream
                            yield f"data: [DONE]\n\n".encode('utf-8')
                            ipc_reader.clear_probe_data(request_id)
                            return
                        
                        # Check if this chunk contains a token
                        has_token = False
                        if "choices" in chunk_data:
                            for choice in chunk_data["choices"]:
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:  # Only count non-empty content
                                        has_token = True
                                        break
                        
                        # Get probe data if available (lazy load, non-blocking)
                        if has_token and probe_data is None:
                            # Try to get probe data with minimal delay
                            probe_data = ipc_reader.get_probe_data(request_id)
                            if probe_data is None:
                                # Try once more after a short delay (non-blocking)
                                await asyncio.sleep(0.005)  # 5ms wait
                                probe_data = ipc_reader.get_probe_data(request_id)
                        
                        # Inject probe values if available
                        if has_token:
                            if probe_data:
                                chunk_data = _inject_probe_values(
                                    chunk_data,
                                    probe_data,
                                    token_index
                                )
                            # Always increment token_index when token is detected
                            # This keeps index in sync even if probe_data is missing
                            token_index += 1
                        
                        # Serialize and yield - always yield all chunks, not just ones with tokens
                        chunk_json = json.dumps(chunk_data)
                        yield f"data: {chunk_json}\n\n".encode('utf-8')
                        if chunk_count <= 3:
                            logger.debug(f"Yielded chunk {chunk_count} for request {request_id}")
                    
                    # Note: Remaining buffer stays in buffer for next iteration
                        
            except GeneratorExit:
                raise
            except (aiohttp.ClientError, ConnectionError) as e:
                logger.debug(f"Connection closed: {type(e).__name__}")
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
            finally:
                # Clean up probe data
                # Note: We don't process remaining buffer here as it may be incomplete
                # and could corrupt SSE format. The buffer will be lost if connection
                # closes, which is acceptable for streaming.
                ipc_reader.clear_probe_data(request_id)
                # Clean up session and response when generator finishes
                try:
                    _response.close()
                    await _session.close()
                except Exception:
                    pass
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to vLLM: {e}")
        ipc_reader.clear_probe_data(request_id)
        try:
            await session.close()
        except:
            pass
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": f"Failed to connect to vLLM: {str(e)}"}
        )


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_v1(request: Request, path: str) -> StreamingResponse:
    """Forward other v1 API requests to vLLM."""
    body: Optional[Any] = None
    if request.method in ["POST", "PUT"]:
        try:
            body = await request.json()
        except Exception:
            body = await request.body()
    
    headers = _prepare_headers(request)
    
    async with aiohttp.ClientSession() as session:
        async with session.request(
            request.method,
            f"{VLLM_URL}/v1/{path}",
            json=body if isinstance(body, dict) else None,
            data=body if not isinstance(body, dict) else None,
            headers=headers,
            params=dict(request.query_params)
        ) as vllm_response:
            content = await vllm_response.read()
            excluded_headers = {"content-length", "transfer-encoding"}
            response_headers = {
                k: v for k, v in vllm_response.headers.items()
                if k.lower() not in excluded_headers
            }
            
            return StreamingResponse(
                iter([content]),
                status_code=vllm_response.status,
                media_type=vllm_response.headers.get("content-type", "application/json"),
                headers=response_headers
            )


def main() -> None:
    """Run the proxy server."""
    logger.info(f"Starting proxy on port {PROXY_PORT}")
    logger.info(f"Forwarding to {VLLM_URL}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PROXY_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()