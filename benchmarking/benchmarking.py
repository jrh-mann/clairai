#!/usr/bin/env python3
"""
Benchmarking script for vLLM endpoints.
Measures throughput, TTFT (Time To First Token), and other performance metrics.
"""

import asyncio
import time
import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import aiohttp
import os


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft: float  # Time to first token (seconds)
    total_time: float  # Total request time (seconds)
    time_per_token: float  # Average time per token (seconds)
    throughput: float  # Tokens per second
    first_token_timestamp: Optional[float] = None
    last_token_timestamp: Optional[float] = None
    error: Optional[str] = None
    has_probe_values: bool = False  # Whether probe values were present in response


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    total_time: float
    
    # Throughput metrics
    avg_throughput: float  # Average tokens per second
    p50_throughput: float
    p95_throughput: float
    p99_throughput: float
    
    # TTFT metrics
    avg_ttft: float  # Average time to first token
    p50_ttft: float
    p95_ttft: float
    p99_ttft: float
    
    # Latency metrics
    avg_total_time: float
    p50_total_time: float
    p95_total_time: float
    p99_total_time: float
    
    # Token metrics
    avg_tokens_per_request: float
    avg_time_per_token: float
    
    # Request rate
    requests_per_second: float
    
    # Probe metrics
    requests_with_probes: int = 0
    probe_coverage: float = 0.0  # Percentage of requests with probe values


class VLLMBenchmarker:
    """Benchmark vLLM endpoints with concurrent requests."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name or os.environ.get("PROBE_MODEL") or "default"
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_available_models(self) -> List[str]:
        """Fetch available models from the endpoint."""
        if not self.session:
            raise RuntimeError("Benchmarker must be used as async context manager")
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with self.session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["id"] for model in data.get("data", [])]
                    return models
                else:
                    return []
        except Exception:
            return []
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        request_id: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True
    ) -> RequestMetrics:
        """Make a single streaming request and measure metrics."""
        start_time = time.time()
        first_token_time = None
        last_token_time = None
        completion_tokens = 0
        prompt_tokens = 0
        error = None
        usage_info = None
        has_probe_values = False
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    # Try to parse JSON error response for better error messages
                    try:
                        error_json = json.loads(error_text)
                        if isinstance(error_json.get("error"), dict):
                            error_msg = error_json["error"].get("message", error_text)
                        elif isinstance(error_json.get("error"), str):
                            error_msg = error_json["error"]
                        else:
                            error_msg = error_text
                    except:
                        error_msg = error_text
                    error = f"HTTP {response.status}: {error_msg}"
                    total_time = time.time() - start_time
                    return RequestMetrics(
                        request_id=request_id,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        ttft=0.0,
                        total_time=total_time,
                        time_per_token=0.0,
                        throughput=0.0,
                        error=error
                    )
                
                if stream:
                    # For streaming, we need to parse Server-Sent Events
                    content_buffer = ""
                    accumulated_content = ""
                    has_probe_values = False
                    usage_info = None
                    stream_done = False
                    
                    try:
                        # Read streaming response chunk by chunk
                        content_buffer = ""
                        async for chunk in response.content.iter_any():
                            if stream_done:
                                break
                                
                            if chunk:
                                content_buffer += chunk.decode('utf-8', errors='ignore')
                            
                            # Process complete lines from buffer
                            while '\n' in content_buffer:
                                line, content_buffer = content_buffer.split('\n', 1)
                                line = line.strip()
                                
                                if not line:
                                    continue
                                
                                if not line.startswith('data: '):
                                    continue
                                
                                data_str = line[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    stream_done = True
                                    # Process any remaining buffer before breaking
                                    break
                                
                                try:
                                    chunk_data = json.loads(data_str)
                                    
                                    # Check for proxy.py custom probe format
                                    chunk_type = chunk_data.get('type', '')
                                    if chunk_type in ['probe_update', 'probe_final']:
                                        if 'probe_scores' in chunk_data and chunk_data['probe_scores']:
                                            has_probe_values = True
                                    elif chunk_type == 'done':
                                        # Final payload with probe_scores
                                        if 'probe_scores' in chunk_data and chunk_data['probe_scores']:
                                            has_probe_values = True
                                    elif chunk_type == 'token':
                                        # Token chunk, check if it has probe data
                                        if 'probe_scores' in chunk_data and chunk_data['probe_scores']:
                                            has_probe_values = True
                                    
                                    # Check for standard OpenAI format
                                    choices = chunk_data.get('choices', [])
                                    if choices:
                                        delta = choices[0].get('delta', {})
                                        content = delta.get('content', '')
                                        
                                        # Check for probe values in standard format
                                        if 'probe_values' in delta:
                                            has_probe_values = True
                                        
                                        if content:
                                            # Check if this is the first token
                                            if first_token_time is None:
                                                first_token_time = time.time()
                                            
                                            # Accumulate content for token counting
                                            accumulated_content += content
                                            last_token_time = time.time()
                                    
                                    # Check for probe_scores at top level (proxy.py format)
                                    if 'probe_scores' in chunk_data and chunk_data['probe_scores']:
                                        has_probe_values = True
                                    
                                    # Check for usage information (usually in final chunk)
                                    if 'usage' in chunk_data:
                                        usage_info = chunk_data['usage']
                                except json.JSONDecodeError as e:
                                    # Skip malformed JSON
                                    continue
                            
                            # If stream is done, break from outer loop
                            if stream_done:
                                break
                        
                        # Process any remaining content in buffer after stream ends
                        if content_buffer and not stream_done:
                            for line in content_buffer.split('\n'):
                                line = line.strip()
                                if line and line.startswith('data: '):
                                    data_str = line[6:]
                                    if data_str != '[DONE]':
                                        try:
                                            chunk_data = json.loads(data_str)
                                            
                                            # Check for proxy.py custom probe format
                                            chunk_type = chunk_data.get('type', '')
                                            if chunk_type in ['probe_update', 'probe_final']:
                                                if 'probe_scores' in chunk_data and chunk_data['probe_scores']:
                                                    has_probe_values = True
                                            elif chunk_type == 'done':
                                                if 'probe_scores' in chunk_data and chunk_data['probe_scores']:
                                                    has_probe_values = True
                                            
                                            # Check for standard OpenAI format
                                            choices = chunk_data.get('choices', [])
                                            if choices:
                                                delta = choices[0].get('delta', {})
                                                content = delta.get('content', '')
                                                if content:
                                                    if first_token_time is None:
                                                        first_token_time = time.time()
                                                    accumulated_content += content
                                                    last_token_time = time.time()
                                                if 'probe_values' in delta:
                                                    has_probe_values = True
                                            
                                            # Check for probe_scores at top level
                                            if 'probe_scores' in chunk_data and chunk_data['probe_scores']:
                                                has_probe_values = True
                                            
                                            if 'usage' in chunk_data:
                                                usage_info = chunk_data['usage']
                                        except json.JSONDecodeError:
                                            pass
                                
                    except Exception as e:
                        error = f"Stream parsing error: {str(e)}"
                    
                    # Use usage info if available, otherwise estimate from content
                    if usage_info and usage_info.get('completion_tokens', 0) > 0:
                        prompt_tokens = usage_info.get('prompt_tokens', 0)
                        completion_tokens = usage_info.get('completion_tokens', 0)
                    elif accumulated_content:
                        # Estimate tokens: more accurate method
                        # Average English token is ~4 characters
                        completion_tokens = max(1, int(len(accumulated_content) / 4))
                        # For prompt, use word count approximation
                        prompt_tokens = max(1, int(len(prompt.split()) * 1.3))
                    else:
                        # No content received - this is an error case
                        completion_tokens = 0
                        prompt_tokens = int(len(prompt.split()) * 1.3)
                        
                else:
                    # Non-streaming response
                    data = await response.json()
                    choices = data.get('choices', [])
                    if choices:
                        message = choices[0].get('message', {})
                        content = message.get('content', '')
                        # Check for probe values in message or at top level (proxy.py returns probe_scores at top level)
                        if 'probe_values' in message or 'all_probe_values' in message:
                            has_probe_values = True
                        # Also check for probe_scores at top level (proxy.py format)
                        if 'probe_scores' in data and data['probe_scores'] is not None:
                            has_probe_values = True
                        # Check for proxy.py custom format with type field
                        if data.get('type') == 'done' and 'probe_scores' in data and data['probe_scores']:
                            has_probe_values = True
                        completion_tokens = len(content.split())
                        prompt_tokens = len(prompt.split())
                        
                        # For non-streaming, we can't measure TTFT accurately
                        # Use a rough estimate based on response time
                        first_token_time = start_time + (time.time() - start_time) * 0.1
                        last_token_time = time.time()
                    
                    if 'usage' in data:
                        usage_info = data['usage']
                        prompt_tokens = usage_info.get('prompt_tokens', prompt_tokens)
                        completion_tokens = usage_info.get('completion_tokens', completion_tokens)
            
            total_time = time.time() - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_time
            
            # Convert to integers for token counts
            prompt_tokens = int(prompt_tokens)
            completion_tokens = int(completion_tokens)
            
            time_per_token = total_time / completion_tokens if completion_tokens > 0 else 0.0
            throughput = completion_tokens / total_time if total_time > 0 else 0.0
            
            return RequestMetrics(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                ttft=ttft,
                total_time=total_time,
                time_per_token=time_per_token,
                throughput=throughput,
                first_token_timestamp=first_token_time,
                last_token_timestamp=last_token_time,
                has_probe_values=has_probe_values
            )
            
        except Exception as e:
            error = str(e)
            total_time = time.time() - start_time
            return RequestMetrics(
                request_id=request_id,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                ttft=0.0,
                total_time=total_time,
                time_per_token=0.0,
                throughput=0.0,
                error=error
            )
    
    async def benchmark(
        self,
        prompts: List[str],
        num_concurrent: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True
    ) -> Tuple[BenchmarkResults, List[RequestMetrics]]:
        """Run benchmark with concurrent requests. Returns results and individual metrics."""
        if not self.session:
            raise RuntimeError("Benchmarker must be used as async context manager")
        
        start_time = time.time()
        metrics: List[RequestMetrics] = []
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(num_concurrent)
        
        async def make_request_with_semaphore(prompt: str, req_id: int):
            async with semaphore:
                return await self._make_request(
                    self.session,
                    prompt,
                    req_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )
        
        # Create tasks
        tasks = [
            make_request_with_semaphore(prompt, i)
            for i, prompt in enumerate(prompts)
        ]
        
        # Run all requests
        metrics = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate aggregate statistics
        results = self._calculate_results(metrics, total_time)
        return results, metrics
    
    def _calculate_results(
        self,
        metrics: List[RequestMetrics],
        total_time: float
    ) -> BenchmarkResults:
        """Calculate aggregated statistics from metrics."""
        successful = [m for m in metrics if m.error is None]
        failed = [m for m in metrics if m.error is not None]
        
        if not successful:
            # Show error details to help debug
            error_counts = {}
            for m in failed:
                error_msg = m.error or "Unknown error"
                error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
            
            error_summary = "\n".join([f"  - {msg}: {count} times" for msg, count in error_counts.items()])
            raise ValueError(
                f"All {len(failed)} requests failed! Check endpoint and configuration.\n"
                f"Errors encountered:\n{error_summary}"
            )
        
        # Extract arrays for statistics
        throughputs = [m.throughput for m in successful]
        ttfts = [m.ttft for m in successful]
        total_times = [m.total_time for m in successful]
        tokens_per_request = [m.completion_tokens for m in successful]
        time_per_token = [m.time_per_token for m in successful if m.time_per_token > 0]
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * p / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        total_tokens = sum(m.completion_tokens for m in successful)
        requests_with_probes = sum(1 for m in successful if m.has_probe_values)
        probe_coverage = (requests_with_probes / len(successful) * 100) if successful else 0.0
        
        return BenchmarkResults(
            total_requests=len(metrics),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_tokens_generated=total_tokens,
            total_time=total_time,
            
            # Throughput
            avg_throughput=statistics.mean(throughputs) if throughputs else 0.0,
            p50_throughput=percentile(throughputs, 50),
            p95_throughput=percentile(throughputs, 95),
            p99_throughput=percentile(throughputs, 99),
            
            # TTFT
            avg_ttft=statistics.mean(ttfts) if ttfts else 0.0,
            p50_ttft=percentile(ttfts, 50),
            p95_ttft=percentile(ttfts, 95),
            p99_ttft=percentile(ttfts, 99),
            
            # Total time
            avg_total_time=statistics.mean(total_times) if total_times else 0.0,
            p50_total_time=percentile(total_times, 50),
            p95_total_time=percentile(total_times, 95),
            p99_total_time=percentile(total_times, 99),
            
            # Token metrics
            avg_tokens_per_request=statistics.mean(tokens_per_request) if tokens_per_request else 0.0,
            avg_time_per_token=statistics.mean(time_per_token) if time_per_token else 0.0,
            
            # Request rate
            requests_per_second=len(metrics) / total_time if total_time > 0 else 0.0,
            
            # Probe metrics
            requests_with_probes=requests_with_probes,
            probe_coverage=probe_coverage,
        )
    
    def print_results(self, results: BenchmarkResults, detailed: bool = False):
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"\nTotal Requests: {results.total_requests}")
        print(f"Successful: {results.successful_requests}")
        print(f"Failed: {results.failed_requests}")
        print(f"\nTotal Time: {results.total_time:.2f}s")
        print(f"Total Tokens Generated: {results.total_tokens_generated:,}")
        
        print("\n" + "-" * 80)
        print("THROUGHPUT (tokens/second)")
        print("-" * 80)
        print(f"Average:  {results.avg_throughput:.2f} tok/s")
        print(f"P50:      {results.p50_throughput:.2f} tok/s")
        print(f"P95:      {results.p95_throughput:.2f} tok/s")
        print(f"P99:      {results.p99_throughput:.2f} tok/s")
        
        print("\n" + "-" * 80)
        print("TIME TO FIRST TOKEN (TTFT)")
        print("-" * 80)
        print(f"Average:  {results.avg_ttft*1000:.2f} ms")
        print(f"P50:      {results.p50_ttft*1000:.2f} ms")
        print(f"P95:      {results.p95_ttft*1000:.2f} ms")
        print(f"P99:      {results.p99_ttft*1000:.2f} ms")
        
        print("\n" + "-" * 80)
        print("TOTAL REQUEST LATENCY")
        print("-" * 80)
        print(f"Average:  {results.avg_total_time:.3f}s")
        print(f"P50:      {results.p50_total_time:.3f}s")
        print(f"P95:      {results.p95_total_time:.3f}s")
        print(f"P99:      {results.p99_total_time:.3f}s")
        
        print("\n" + "-" * 80)
        print("OTHER METRICS")
        print("-" * 80)
        print(f"Avg Tokens per Request: {results.avg_tokens_per_request:.1f}")
        print(f"Avg Time per Token: {results.avg_time_per_token*1000:.3f} ms")
        print(f"Requests per Second: {results.requests_per_second:.2f}")
        print(f"Requests with Probe Values: {results.requests_with_probes}/{results.successful_requests} ({results.probe_coverage:.1f}%)")
        
        print("\n" + "=" * 80 + "\n")
    
    def save_results(self, results: BenchmarkResults, metrics: List[RequestMetrics], output_file: str):
        """Save detailed results to JSON file."""
        output = {
            "summary": asdict(results),
            "individual_requests": [asdict(m) for m in metrics]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_file}")


def print_comparison(direct_results: BenchmarkResults, proxy_results: BenchmarkResults, direct_metrics: List[RequestMetrics], proxy_metrics: List[RequestMetrics]):
    """Print side-by-side comparison of direct vLLM vs proxy results."""
    print("\n" + "=" * 100)
    print("COMPARISON: Direct vLLM vs Proxy Server")
    print("=" * 100)
    
    def format_diff(direct_val: float, proxy_val: float, unit: str = "", is_lower_better: bool = False) -> str:
        """Format a comparison value with difference percentage."""
        if direct_val == 0:
            return f"{proxy_val:.2f}{unit} (N/A)"
        diff = ((proxy_val - direct_val) / direct_val) * 100
        arrow = "↓" if (is_lower_better and diff < 0) or (not is_lower_better and diff > 0) else "↑"
        color = "better" if arrow == "↓" and is_lower_better or arrow == "↑" and not is_lower_better else "worse"
        return f"{proxy_val:.2f}{unit} ({diff:+.1f}% {arrow})"
    
    print(f"\nTotal Requests: {direct_results.total_requests} (both)")
    print(f"Successful: Direct={direct_results.successful_requests}, Proxy={proxy_results.successful_requests}")
    print(f"Failed: Direct={direct_results.failed_requests}, Proxy={proxy_results.failed_requests}")
    print(f"Total Tokens: Direct={direct_results.total_tokens_generated:,}, Proxy={proxy_results.total_tokens_generated:,}")
    print(f"Total Time: Direct={direct_results.total_time:.2f}s, Proxy={proxy_results.total_time:.2f}s")
    
    print("\n" + "-" * 100)
    print("THROUGHPUT (tokens/second)")
    print("-" * 100)
    print(f"{'Metric':<20} {'Direct vLLM':<25} {'Proxy':<25} {'Difference':<25}")
    print("-" * 100)
    print(f"{'Average':<20} {direct_results.avg_throughput:<25.2f} {format_diff(direct_results.avg_throughput, proxy_results.avg_throughput, ' tok/s'):<25}")
    print(f"{'P50':<20} {direct_results.p50_throughput:<25.2f} {format_diff(direct_results.p50_throughput, proxy_results.p50_throughput, ' tok/s'):<25}")
    print(f"{'P95':<20} {direct_results.p95_throughput:<25.2f} {format_diff(direct_results.p95_throughput, proxy_results.p95_throughput, ' tok/s'):<25}")
    print(f"{'P99':<20} {direct_results.p99_throughput:<25.2f} {format_diff(direct_results.p99_throughput, proxy_results.p99_throughput, ' tok/s'):<25}")
    
    print("\n" + "-" * 100)
    print("TIME TO FIRST TOKEN (TTFT) - Lower is Better")
    print("-" * 100)
    print(f"{'Metric':<20} {'Direct vLLM':<25} {'Proxy':<25} {'Difference':<25}")
    print("-" * 100)
    print(f"{'Average':<20} {direct_results.avg_ttft*1000:<25.2f} ms {format_diff(direct_results.avg_ttft*1000, proxy_results.avg_ttft*1000, ' ms', True):<25}")
    print(f"{'P50':<20} {direct_results.p50_ttft*1000:<25.2f} ms {format_diff(direct_results.p50_ttft*1000, proxy_results.p50_ttft*1000, ' ms', True):<25}")
    print(f"{'P95':<20} {direct_results.p95_ttft*1000:<25.2f} ms {format_diff(direct_results.p95_ttft*1000, proxy_results.p95_ttft*1000, ' ms', True):<25}")
    print(f"{'P99':<20} {direct_results.p99_ttft*1000:<25.2f} ms {format_diff(direct_results.p99_ttft*1000, proxy_results.p99_ttft*1000, ' ms', True):<25}")
    
    print("\n" + "-" * 100)
    print("TOTAL REQUEST LATENCY - Lower is Better")
    print("-" * 100)
    print(f"{'Metric':<20} {'Direct vLLM':<25} {'Proxy':<25} {'Difference':<25}")
    print("-" * 100)
    print(f"{'Average':<20} {direct_results.avg_total_time:<25.3f}s {format_diff(direct_results.avg_total_time, proxy_results.avg_total_time, 's', True):<25}")
    print(f"{'P50':<20} {direct_results.p50_total_time:<25.3f}s {format_diff(direct_results.p50_total_time, proxy_results.p50_total_time, 's', True):<25}")
    print(f"{'P95':<20} {direct_results.p95_total_time:<25.3f}s {format_diff(direct_results.p95_total_time, proxy_results.p95_total_time, 's', True):<25}")
    print(f"{'P99':<20} {direct_results.p99_total_time:<25.3f}s {format_diff(direct_results.p99_total_time, proxy_results.p99_total_time, 's', True):<25}")
    
    print("\n" + "-" * 100)
    print("OTHER METRICS")
    print("-" * 100)
    print(f"{'Metric':<30} {'Direct vLLM':<30} {'Proxy':<30}")
    print("-" * 100)
    print(f"{'Avg Tokens per Request':<30} {direct_results.avg_tokens_per_request:<30.1f} {proxy_results.avg_tokens_per_request:<30.1f}")
    print(f"{'Avg Time per Token':<30} {direct_results.avg_time_per_token*1000:<30.3f} ms {proxy_results.avg_time_per_token*1000:<30.3f} ms")
    print(f"{'Requests per Second':<30} {direct_results.requests_per_second:<30.2f} {proxy_results.requests_per_second:<30.2f}")
    print(f"{'Requests with Probes':<30} {direct_results.requests_with_probes}/{direct_results.successful_requests} ({direct_results.probe_coverage:.1f}%) {proxy_results.requests_with_probes}/{proxy_results.successful_requests} ({proxy_results.probe_coverage:.1f}%)")
    
    # Calculate overhead
    print("\n" + "-" * 100)
    print("PROXY OVERHEAD ANALYSIS")
    print("-" * 100)
    ttft_overhead = ((proxy_results.avg_ttft - direct_results.avg_ttft) / direct_results.avg_ttft * 100) if direct_results.avg_ttft > 0 else 0
    latency_overhead = ((proxy_results.avg_total_time - direct_results.avg_total_time) / direct_results.avg_total_time * 100) if direct_results.avg_total_time > 0 else 0
    throughput_overhead = ((direct_results.avg_throughput - proxy_results.avg_throughput) / direct_results.avg_throughput * 100) if direct_results.avg_throughput > 0 else 0
    
    print(f"TTFT Overhead: {ttft_overhead:+.2f}% ({proxy_results.avg_ttft*1000 - direct_results.avg_ttft*1000:+.2f} ms)")
    print(f"Latency Overhead: {latency_overhead:+.2f}% ({proxy_results.avg_total_time - direct_results.avg_total_time:+.3f} s)")
    print(f"Throughput Reduction: {throughput_overhead:+.2f}% ({proxy_results.avg_throughput - direct_results.avg_throughput:+.2f} tok/s)")
    
    print("\n" + "=" * 100 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM endpoint performance")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:6969/v1",
        help="Base URL of the vLLM endpoint (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name (default: default)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (optional)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["What is the meaning of life?"],
        help="Prompts to use for benchmarking"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to make (default: 10)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 1)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per request (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (less accurate TTFT measurement)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--repeat-prompts",
        action="store_true",
        help="Repeat prompts to reach --num-requests count"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare direct vLLM vs proxy server (requires --direct-url and --proxy-url)"
    )
    parser.add_argument(
        "--direct-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Direct vLLM URL for comparison (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--proxy-url",
        type=str,
        default="http://localhost:8888/v1",
        help="Proxy server URL for comparison (default: http://localhost:8888/v1)"
    )
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = list(args.prompts)
    
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            file_prompts = [line.strip() for line in f if line.strip()]
            prompts.extend(file_prompts)
    
    # Repeat prompts if needed
    if args.repeat_prompts and len(prompts) < args.num_requests:
        prompts = (prompts * ((args.num_requests // len(prompts)) + 1))[:args.num_requests]
    elif len(prompts) > args.num_requests:
        prompts = prompts[:args.num_requests]
    elif len(prompts) < args.num_requests:
        # Repeat last prompt
        prompts.extend([prompts[-1]] * (args.num_requests - len(prompts)))
    
    # Auto-detect model if not provided or is placeholder
    if not args.model or args.model == "default" or args.model == "your-model-name":
        # Try to detect from the endpoint
        test_url = args.proxy_url if args.compare else args.url
        print(f"Attempting to auto-detect model from {test_url}...")
        try:
            async with VLLMBenchmarker(base_url=test_url, api_key=args.api_key) as temp_benchmarker:
                available_models = await temp_benchmarker.get_available_models()
                if available_models:
                    args.model = available_models[0]
                    print(f"✓ Auto-detected model: {args.model}")
                    if len(available_models) > 1:
                        print(f"  Available models: {', '.join(available_models)}")
                else:
                    print(f"⚠ WARNING: Could not auto-detect model from {test_url}")
                    print("  Please specify --model explicitly or ensure the endpoint is accessible.")
                    if not args.model or args.model == "your-model-name":
                        print("  Attempting to continue with model name from endpoint...")
        except Exception as e:
            print(f"⚠ WARNING: Failed to auto-detect model: {e}")
            print("  Please specify --model explicitly.")
        print()
    
    if args.compare:
        # Comparison mode: run both direct and proxy benchmarks
        print("=" * 100)
        print("COMPARISON MODE: Direct vLLM vs Proxy Server")
        print("=" * 100)
        print(f"Model: {args.model}")
        print(f"Number of requests: {len(prompts)}")
        print(f"Concurrent requests: {args.concurrent}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Streaming: {not args.no_stream}")
        print()
        
        # Run direct vLLM benchmark
        print("\n" + "=" * 100)
        print("BENCHMARKING DIRECT vLLM")
        print("=" * 100)
        async with VLLMBenchmarker(
            base_url=args.direct_url,
            api_key=args.api_key,
            model_name=args.model
        ) as direct_benchmarker:
            direct_results, direct_metrics = await direct_benchmarker.benchmark(
                prompts=prompts,
                num_concurrent=args.concurrent,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=not args.no_stream
            )
            direct_benchmarker.print_results(direct_results, detailed=False)
        
        # Run proxy benchmark
        print("\n" + "=" * 100)
        print("BENCHMARKING PROXY SERVER")
        print("=" * 100)
        async with VLLMBenchmarker(
            base_url=args.proxy_url,
            api_key=args.api_key,
            model_name=args.model
        ) as proxy_benchmarker:
            proxy_results, proxy_metrics = await proxy_benchmarker.benchmark(
                prompts=prompts,
                num_concurrent=args.concurrent,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=not args.no_stream
            )
            proxy_benchmarker.print_results(proxy_results, detailed=False)
        
        # Print comparison
        print_comparison(direct_results, proxy_results, direct_metrics, proxy_metrics)
        
        # Save results if requested
        if args.output:
            output_data = {
                "direct": {
                    "summary": asdict(direct_results),
                    "individual_requests": [asdict(m) for m in direct_metrics]
                },
                "proxy": {
                    "summary": asdict(proxy_results),
                    "individual_requests": [asdict(m) for m in proxy_metrics]
                }
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Comparison results saved to {args.output}")
    else:
        # Single endpoint benchmark mode
        print(f"Benchmarking vLLM endpoint: {args.url}")
        print(f"Model: {args.model}")
        print(f"Number of requests: {len(prompts)}")
        print(f"Concurrent requests: {args.concurrent}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Streaming: {not args.no_stream}")
        print()
        
        async with VLLMBenchmarker(
            base_url=args.url,
            api_key=args.api_key,
            model_name=args.model
        ) as benchmarker:
            results, metrics = await benchmarker.benchmark(
                prompts=prompts,
                num_concurrent=args.concurrent,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=not args.no_stream
            )
            
            benchmarker.print_results(results, detailed=True)
            
            if args.output:
                benchmarker.save_results(results, metrics, args.output)


if __name__ == "__main__":
    asyncio.run(main())
