#!/usr/bin/env python3
"""
Minimal script to hit vLLM API and monitor probe outputs from IPC.
Returns the API response and all probe outputs collected during the request.
"""

import asyncio
import aiohttp
import json
import time
import os
import cupy as cp
import numpy as np
import ctypes
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

IPC_PATH = "/tmp/vllm_probe.ipc"
META_PATH = "/tmp/vllm_probe_meta.json"


class IPCReader:
    """Reads probe outputs from IPC shared memory."""
    
    def __init__(self):
        self.gpu_mem_ptr = None
        self.meta = None
        self.last_seq = 0
        self.probe_outputs = []
        
    def connect(self):
        """Connect to IPC shared memory."""
        # Wait for IPC handle
        while not os.path.exists(IPC_PATH):
            time.sleep(0.1)
        
        # Load metadata
        with open(META_PATH, 'r') as f:
            self.meta = json.load(f)
        
        # Open IPC handle
        with open(IPC_PATH, "rb") as f:
            handle_bytes = f.read()
        
        self.gpu_mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
        
        # Initialize last_seq to current head
        self.last_seq = self._read_head()
        
    def _read_head(self) -> int:
        """Read head value from IPC buffer."""
        cp.cuda.Device().synchronize()
        cudart = ctypes.CDLL('libcudart.so')
        host_buf = (ctypes.c_uint64 * 1)()
        cudart.cudaMemcpy(
            ctypes.byref(host_buf),
            ctypes.c_void_p(self.gpu_mem_ptr),
            8,
            ctypes.c_int(2)  # cudaMemcpyDeviceToHost
        )
        return int(host_buf[0])
    
    def read_new_outputs(self) -> List[Dict[str, Any]]:
        """Read all new probe outputs since last call."""
        outputs = []
        curr_seq = self._read_head()
        
        if curr_seq <= self.last_seq:
            return outputs
        
        RING_SIZE = self.meta["ring_size"]
        SLOT_SIZE_BYTES = self.meta["slot_size_bytes"]
        NUM_PROBES = self.meta["num_probes"]
        data_base_ptr = self.gpu_mem_ptr + 128
        
        delta = curr_seq - self.last_seq
        if delta > RING_SIZE:
            start = curr_seq - 1
        else:
            start = self.last_seq
        
        cudart = ctypes.CDLL('libcudart.so')
        
        for seq in range(start, curr_seq):
            slot_idx = seq % RING_SIZE
            src_ptr = data_base_ptr + (slot_idx * SLOT_SIZE_BYTES)
            
            cp.cuda.Device().synchronize()
            
            # Read batch_id
            host_batch_id = (ctypes.c_uint64 * 1)()
            cudart.cudaMemcpy(
                ctypes.byref(host_batch_id),
                ctypes.c_void_p(src_ptr),
                8,
                ctypes.c_int(2)
            )
            batch_id = int(host_batch_id[0])
            
            # Read num_requests
            host_num_requests = (ctypes.c_int32 * 1)()
            cudart.cudaMemcpy(
                ctypes.byref(host_num_requests),
                ctypes.c_void_p(src_ptr + 8),
                4,
                ctypes.c_int(2)
            )
            num_requests = int(host_num_requests[0])
            
            if num_requests == 0:
                continue
            
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
            scores_size = num_tokens * NUM_PROBES
            
            host_scores = (ctypes.c_float * scores_size)()
            cudart.cudaMemcpy(
                ctypes.byref(host_scores),
                ctypes.c_void_p(src_ptr + scores_offset),
                scores_size * 4,
                ctypes.c_int(2)
            )
            scores_array = np.array([float(host_scores[i]) for i in range(scores_size)])
            probe_scores = scores_array.reshape(num_tokens, NUM_PROBES)
            
            # Process each request in batch
            for req_idx in range(num_requests):
                start_token = int(query_start_loc[req_idx])
                end_token = int(query_start_loc[req_idx + 1])
                request_scores = probe_scores[start_token:end_token]
                
                outputs.append({
                    "sequence": seq,
                    "batch_id": batch_id,
                    "request_idx": req_idx,
                    "start_token": start_token,
                    "end_token": end_token,
                    "probe_scores": request_scores.tolist(),
                    "avg_score": float(request_scores.mean()) if request_scores.size > 0 else 0.0
                })
        
        self.last_seq = curr_seq
        return outputs


async def make_vllm_request(url: str, model: str, prompt: str, request_id: int, max_tokens: int = 100) -> Dict[str, Any]:
    """Make a request to vLLM API."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/chat/completions", json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                return {"request_id": request_id, "prompt": prompt, "error": f"HTTP {response.status}: {error_text}"}
            
            content = ""
            content_buffer = ""
            async for chunk in response.content.iter_any():
                if chunk:
                    content_buffer += chunk.decode('utf-8', errors='ignore')
                
                # Process complete lines
                while '\n' in content_buffer:
                    line, content_buffer = content_buffer.split('\n', 1)
                    line = line.strip()
                    
                    if not line or not line.startswith('data: '):
                        continue
                    
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content += delta['content']
                    except json.JSONDecodeError:
                        pass
            
            return {"request_id": request_id, "prompt": prompt, "content": content, "status": response.status}


async def monitor_probes(ipc_reader: IPCReader, stop_event: asyncio.Event) -> List[Dict[str, Any]]:
    """Monitor probe outputs until stop event is set."""
    all_outputs = []
    
    while not stop_event.is_set():
        outputs = ipc_reader.read_new_outputs()
        all_outputs.extend(outputs)
        await asyncio.sleep(0.01)  # Check every 10ms
    
    # Final read to catch any remaining outputs
    await asyncio.sleep(0.1)
    outputs = ipc_reader.read_new_outputs()
    all_outputs.extend(outputs)
    
    return all_outputs


def tokenize_response(tokenizer, response_text: str) -> List[Dict[str, Any]]:
    """Tokenize response text and return tokens with their indices."""
    # Tokenize the response
    tokens = tokenizer.encode(response_text, add_special_tokens=False)
    token_ids = tokens
    
    # Get token strings using the tokenizer's convert_ids_to_tokens for better accuracy
    try:
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    except:
        # Fallback to decode if convert_ids_to_tokens not available
        token_strings = []
        for token_id in token_ids:
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            token_strings.append(token_str)
    
    # Create list of token info
    token_info = []
    for i, (token_id, token_str) in enumerate(zip(token_ids, token_strings)):
        token_info.append({
            "index": i,
            "token_id": token_id,
            "text": token_str
        })
    
    return token_info


def tokenize_prompt(tokenizer, prompt: str, model: str) -> int:
    """Tokenize prompt and return number of tokens.
    For chat models, we need to format the prompt properly."""
    try:
        # Try to use chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            # Format as a chat message
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = tokenizer.encode(formatted, add_special_tokens=False)
        else:
            # Fallback to simple tokenization
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    except Exception as e:
        # If chat template fails, use simple tokenization
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    return len(prompt_tokens)


def align_probes_with_tokens(
    probe_outputs: List[Dict[str, Any]], 
    tokens: List[Dict[str, Any]],
    prompt: str,
    tokenizer,
    model: str
) -> List[Dict[str, Any]]:
    """Align probe outputs with tokenized response tokens.
    
    Structure: 
    - First batch contains all prompt tokens (start_token=0, end_token=prompt_length)
    - Subsequent batches each contain one generated token (start_token=0, end_token=1)
    """
    # Tokenize prompt to know how many prompt tokens there are
    num_prompt_tokens = tokenize_prompt(tokenizer, prompt, model)
    
    aligned_outputs = []
    generated_token_idx = 0  # Track which generated token we're on
    
    for batch_idx, probe_output in enumerate(probe_outputs):
        start_token = probe_output["start_token"]
        end_token = probe_output["end_token"]
        probe_scores = probe_output["probe_scores"]
        num_tokens_in_batch = end_token - start_token
        
        aligned_tokens = []
        
        # First batch contains the prompt tokens - skip it
        if batch_idx == 0:
            # This is the prompt batch, skip aligning these tokens
            aligned_outputs.append({
                **probe_output,
                "aligned_tokens": [],
                "num_aligned_tokens": 0,
                "prompt_token_offset": num_prompt_tokens,
                "is_prompt_batch": True
            })
            continue
        
        # Subsequent batches: each batch with 0-1 range represents one generated token
        # The probe_scores array has shape [num_tokens_in_batch, num_probes]
        # For a 0-1 batch, we have one row of probe scores
        
        if num_tokens_in_batch == 1 and generated_token_idx < len(tokens):
            # This is a single generated token
            if len(probe_scores) > 0:
                token_info = {
                    "token_index": generated_token_idx,
                    "token_text": tokens[generated_token_idx]["text"],
                    "token_id": tokens[generated_token_idx]["token_id"],
                    "probe_scores": probe_scores[0],  # First (and only) row
                    "avg_probe_score": float(np.mean(probe_scores[0])) if len(probe_scores[0]) > 0 else 0.0
                }
                aligned_tokens.append(token_info)
                generated_token_idx += 1
        else:
            # Handle case where batch might have multiple tokens
            for i in range(num_tokens_in_batch):
                if generated_token_idx < len(tokens) and i < len(probe_scores):
                    token_info = {
                        "token_index": generated_token_idx,
                        "token_text": tokens[generated_token_idx]["text"],
                        "token_id": tokens[generated_token_idx]["token_id"],
                        "probe_scores": probe_scores[i],
                        "avg_probe_score": float(np.mean(probe_scores[i])) if len(probe_scores[i]) > 0 else 0.0
                    }
                    aligned_tokens.append(token_info)
                    generated_token_idx += 1
        
        aligned_outputs.append({
            **probe_output,
            "aligned_tokens": aligned_tokens,
            "num_aligned_tokens": len(aligned_tokens),
            "prompt_token_offset": num_prompt_tokens,
            "is_prompt_batch": False
        })
    
    return aligned_outputs


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hit vLLM API and monitor probe outputs")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name (default: from PROBE_MODEL env)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    args = parser.parse_args()
    
    model = args.model or os.environ.get("PROBE_MODEL", "default")
    
    # Load tokenizer
    print(f"Loading tokenizer for {model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        print("  Continuing without tokenization...")
        tokenizer = None
    
    # Connect to IPC
    print("Connecting to IPC...")
    ipc_reader = IPCReader()
    try:
        ipc_reader.connect()
        print("✓ Connected to IPC")
    except Exception as e:
        print(f"✗ Failed to connect to IPC: {e}")
        return
    
    # Make API request and monitor probes concurrently
    print(f"Making request to {args.url}...")
    print(f"Prompt: {args.prompt}")
    
    start_time = time.time()
    stop_event = asyncio.Event()
    
    # Start monitoring task
    monitor_task = asyncio.create_task(monitor_probes(ipc_reader, stop_event))
    
    # Make API request
    response = await make_vllm_request(args.url, model, args.prompt, args.max_tokens)
    
    # Stop monitoring and get all probe outputs
    stop_event.set()
    probe_outputs = await monitor_task
    
    total_time = time.time() - start_time
    
    # Tokenize response and align with probes
    aligned_outputs = []
    tokens = []
    
    if tokenizer and "error" not in response:
        response_content = response.get("content", "")
        if response_content:
            tokens = tokenize_response(tokenizer, response_content)
            aligned_outputs = align_probes_with_tokens(probe_outputs, tokens, args.prompt, tokenizer, model)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nAPI Response:")
    if "error" in response:
        print(f"  Error: {response['error']}")
    else:
        print(f"  Status: {response['status']}")
        print(f"  Content: {response.get('content', '')}")
        if tokens:
            print(f"  Tokens: {len(tokens)} tokens")
    
    print(f"\nProbe Outputs: {len(probe_outputs)} batches")
    for i, output in enumerate(probe_outputs):
        print(f"  [{i+1}] Seq: {output['sequence']}, Batch: {output['batch_id']}, "
              f"Req: {output['request_idx']}, Tokens: {output['start_token']}-{output['end_token']}, "
              f"Avg Score: {output['avg_score']:.4f}")
    
    # Print aligned tokens with probe values
    if aligned_outputs:
        print(f"\n" + "="*80)
        print("TOKEN-PROBE ALIGNMENT")
        print("="*80)
        for i, aligned in enumerate(aligned_outputs):
            print(f"\nBatch {i+1} (Seq: {aligned['sequence']}, Batch ID: {aligned['batch_id']}):")
            print(f"  Prompt token offset: {aligned['prompt_token_offset']}")
            print(f"  Aligned tokens: {aligned['num_aligned_tokens']}")
            
            if aligned['aligned_tokens']:
                print(f"  Token details:")
                for token_info in aligned['aligned_tokens'][:20]:  # Show first 20 tokens
                    token_text = repr(token_info['token_text'])
                    avg_score = token_info['avg_probe_score']
                    print(f"    [{token_info['token_index']:3d}] {token_text:20s} | Avg Probe: {avg_score:.4f}")
                if len(aligned['aligned_tokens']) > 20:
                    print(f"    ... ({len(aligned['aligned_tokens']) - 20} more tokens)")
    
    print(f"\nTotal Time: {total_time:.2f}s")
    print("="*80)
    
    # Return structured result
    result = {
        "response": response,
        "probe_outputs": probe_outputs,
        "tokens": tokens,
        "aligned_outputs": aligned_outputs if aligned_outputs else None,
        "total_time": total_time,
        "num_probe_batches": len(probe_outputs)
    }
    
    return result


if __name__ == "__main__":
    asyncio.run(main())

