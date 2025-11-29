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


class VLLMBenchmarker:
    """Benchmark vLLM endpoints with concurrent requests."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = os.environ.get("PROBE_MODEL")
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
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
                    error = f"HTTP {response.status}: {error_text}"
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
                    async for chunk in response.content.iter_any():
                        content_buffer += chunk.decode('utf-8', errors='ignore')
                        
                        # Process complete lines
                        while '\n' in content_buffer:
                            line, content_buffer = content_buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line or not line.startswith('data: '):
                                continue
                            
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str == '[DONE]':
                                continue
                            
                            try:
                                chunk_data = json.loads(data_str)
                                choices = chunk_data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    
                                    if content:
                                        # Check if this is the first token
                                        if first_token_time is None:
                                            first_token_time = time.time()
                                        
                                        # Accumulate content for token counting
                                        accumulated_content += content
                                        last_token_time = time.time()
                                
                                # Check for usage information (usually in final chunk)
                                if 'usage' in chunk_data:
                                    usage_info = chunk_data['usage']
                            except json.JSONDecodeError:
                                continue
                    
                    # Use usage info if available, otherwise estimate from content
                    if usage_info:
                        prompt_tokens = usage_info.get('prompt_tokens', 0)
                        completion_tokens = usage_info.get('completion_tokens', 0)
                    else:
                        # Estimate tokens from accumulated content
                        # Rough approximation: ~1.3 tokens per word for English
                        prompt_tokens = int(len(prompt.split()) * 1.3)
                        completion_tokens = int(len(accumulated_content.split()) * 1.3)
                        
                else:
                    # Non-streaming response
                    data = await response.json()
                    choices = data.get('choices', [])
                    if choices:
                        message = choices[0].get('message', {})
                        content = message.get('content', '')
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
                last_token_timestamp=last_token_time
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
            raise ValueError("All requests failed! Check endpoint and configuration.")
        
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


async def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM endpoint performance")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
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
    
    print(f"Benchmarking vLLM endpoint: {args.url}")
    print(f"Model: {args.model}")
    print(f"Number of requests: {len(prompts)}")
    print(f"Concurrent requests: {args.concurrent}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Streaming: {not args.no_stream}")
    print()
    
    async with VLLMBenchmarker(
        base_url=args.url,
        api_key=args.api_key
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
