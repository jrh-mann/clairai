#!/usr/bin/env python3
"""
Find and download SAE encoder features from Neuronpedia.

Usage:
    python find_feature.py "deception"           # Search by description
    python find_feature.py "lying" --layer 19    # Filter to layer 19
    python find_feature.py --feature-id 56479    # Direct download by ID
    python find_feature.py --browse              # Show known useful features

The script searches Neuronpedia's explanation database, lets you select
features interactively, downloads the encoder direction from the SAE,
and saves them to probes/llama-3.1-8b/ for use with probefast.
"""

import argparse
import json
import os
import sys
import requests
import torch
from pathlib import Path

# Config
MODEL_ID = "llama3.1-8b-it"
SOURCE_ID = "19-resid-post-aa"  # Layer 19 resid post
SAE_REPO = "andyrdt/saes-llama-3.1-8b-instruct"
SAE_PATH = "resid_post_layer_19/trainer_1/ae.pt"
PROBES_DIR = Path(__file__).parent / "probes" / "llama-3.1-8b"
CACHE_DIR = Path("/tmp/sae_cache")

# Neuronpedia API
NP_API = "https://www.neuronpedia.org/api"

# Some known useful features (you can add to this list)
KNOWN_FEATURES = {
    56479: "deception/misleading content",
    # Add more as you discover them
}


def search_features(query: str, top_k: int = 10, layer_filter: str = None) -> list:
    """Search Neuronpedia for features matching a query using explanation search."""
    print(f"\n🔍 Searching for features related to: '{query}'")
    
    url = f"{NP_API}/explanation/search-model"
    payload = {
        "query": query,
        "modelId": MODEL_ID
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            
            # Filter by layer if specified
            if layer_filter:
                results = [r for r in results if r.get("layer", "").startswith(layer_filter)]
            
            # Sort by similarity
            results.sort(key=lambda x: x.get("cosine_similarity", 0), reverse=True)
            
            return results[:top_k]
    except Exception as e:
        print(f"Search failed: {e}")
    
    print("❌ Search API unavailable. Try --feature-id for direct download.")
    return []


def get_feature_info(feature_idx: int) -> dict:
    """Get detailed info about a specific feature."""
    url = f"{NP_API}/feature/{MODEL_ID}/{SOURCE_ID}/{feature_idx}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Failed to get feature info: {e}")
    return {}


def display_feature(info: dict, idx: int = None):
    """Display feature information nicely."""
    feature_id = info.get("index", "?")
    prefix = f"[{idx}] " if idx is not None else ""
    
    print(f"\n{prefix}Feature {feature_id}")
    print("-" * 50)
    
    # Top activating tokens
    pos_tokens = info.get("pos_str", [])[:8]
    pos_values = info.get("pos_values", [])[:8]
    if pos_tokens:
        print("Top activating tokens:")
        for tok, val in zip(pos_tokens, pos_values):
            print(f"  {val:+.4f}  {repr(tok)}")
    
    # Stats
    max_act = info.get("maxActApprox", 0)
    frac = info.get("frac_nonzero", 0)
    print(f"\nMax activation: {max_act:.2f}")
    print(f"Sparsity: {frac*100:.4f}% of tokens activate")
    
    # Explanation if available
    explanations = info.get("explanations", [])
    if explanations:
        print(f"\nExplanation: {explanations[0].get('description', 'N/A')}")


def load_sae() -> dict:
    """Load or download the SAE weights."""
    cache_path = CACHE_DIR / SAE_PATH
    
    if not cache_path.exists():
        print(f"\n📥 Downloading SAE weights (~4GB)...")
        print("This only happens once, weights will be cached.")
        
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=SAE_REPO,
            filename=SAE_PATH,
            local_dir=str(CACHE_DIR)
        )
    
    print("Loading SAE...")
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def extract_encoder(sae: dict, feature_idx: int) -> tuple:
    """Extract encoder direction and bias for a feature."""
    encoder_weight = sae["encoder.weight"]  # [131072, 4096]
    encoder_bias = sae["encoder.bias"]       # [131072]
    
    direction = encoder_weight[feature_idx]  # [4096]
    bias = encoder_bias[feature_idx].item()
    
    return direction, bias


def save_feature(feature_idx: int, direction: torch.Tensor, bias: float, info: dict):
    """Save the encoder feature to probes directory."""
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = f"enc_19_{feature_idx}.json"
    filepath = PROBES_DIR / filename
    
    # Build the probe file
    data = {
        "modelId": MODEL_ID,
        "layer": "19-resid-post",
        "index": str(feature_idx),
        "type": "encoder",
        "encoder_bias": bias,
        "vector": direction.tolist(),
        # Include some metadata
        "max_activation": info.get("maxActApprox"),
        "sparsity": info.get("frac_nonzero"),
        "top_tokens": info.get("pos_str", [])[:5]
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f)
    
    print(f"\n✅ Saved to: {filepath}")
    return filepath


def interactive_search(query: str, top_k: int = 10, layer: str = None):
    """Search and interactively select features to download."""
    results = search_features(query, top_k, layer_filter=layer)
    
    if not results:
        print("No features found. Try a different query or use --feature-id for direct download.")
        return
    
    # Display search results
    print(f"\nFound {len(results)} features:\n")
    print(f"{'#':<4} {'Layer':<20} {'Index':<10} {'Similarity':<12} Description")
    print("-" * 80)
    
    for i, result in enumerate(results):
        layer_name = result.get("layer", "?")
        idx = result.get("index", "?")
        desc = result.get("description", "N/A")[:30]
        sim = result.get("cosine_similarity", 0)
        print(f"{i+1:<4} {layer_name:<20} {idx:<10} {sim:<12.3f} {desc}")
    
    # Ask user to select
    print("\n" + "=" * 80)
    print("Note: Only layer-19 features can be downloaded with the current SAE.")
    selection = input("\nEnter number(s) to download (e.g., '1' or '1,2,3') or 'q' to quit: ").strip()
    
    if selection.lower() == 'q':
        return
    
    # Parse selection
    try:
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
    except ValueError:
        print("Invalid selection")
        return
    
    # Load SAE once
    sae = None
    
    # Download selected features
    for idx in indices:
        if 0 <= idx < len(results):
            result = results[idx]
            layer_name = result.get("layer", "")
            feature_idx = int(result.get("index", 0))
            
            # Check if it's layer 19
            if not layer_name.startswith("19"):
                print(f"\n⚠️  Feature {feature_idx} is from {layer_name}, not layer 19.")
                print("   Current SAE only supports layer 19. Skipping.")
                continue
            
            # Load SAE lazily
            if sae is None:
                sae = load_sae()
            
            # Get full feature info
            info = get_feature_info(feature_idx)
            if info:
                display_feature(info)
            
            print(f"\n📦 Extracting encoder for feature {feature_idx}...")
            direction, bias = extract_encoder(sae, feature_idx)
            
            print(f"Encoder direction norm: {direction.norm():.4f}")
            print(f"Encoder bias: {bias:.4f}")
            
            save_feature(feature_idx, direction, bias, info or {})


def browse_features():
    """Browse known/recommended features."""
    print("\n📚 Known useful features for layer 19:")
    print("=" * 60)
    
    for idx, desc in KNOWN_FEATURES.items():
        print(f"  {idx:>6}  {desc}")
        info = get_feature_info(idx)
        if info:
            pos_tokens = info.get("pos_str", [])[:5]
            print(f"           Top tokens: {', '.join(repr(t) for t in pos_tokens)}")
    
    print("\n" + "=" * 60)
    print(f"\n🔗 Browse more: https://www.neuronpedia.org/{MODEL_ID}/{SOURCE_ID}")
    print("\nTo download a feature:")
    print("  python find_feature.py --feature-id <ID>")


def direct_download(feature_idx: int, auto_confirm: bool = False):
    """Directly download a feature by ID."""
    print(f"\n📦 Feature {feature_idx}")
    print("=" * 50)
    
    # Get feature info
    info = get_feature_info(feature_idx)
    if info:
        display_feature(info)
    else:
        print("Could not fetch feature info from Neuronpedia.")
        print("Will still attempt to download from SAE weights.")
    
    # Confirm
    if not auto_confirm:
        confirm = input("\nDownload this feature? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    # Load SAE and extract
    sae = load_sae()
    direction, bias = extract_encoder(sae, feature_idx)
    
    print(f"\n📐 Encoder stats:")
    print(f"   Direction norm: {direction.norm():.4f}")
    print(f"   Bias: {bias:.4f}")
    
    filepath = save_feature(feature_idx, direction, bias, info or {})
    
    # Update known features
    if feature_idx not in KNOWN_FEATURES and info:
        tokens = info.get("pos_str", [])[:3]
        desc = ", ".join(repr(t) for t in tokens) if tokens else "unknown"
        print(f"\n💡 Tip: Add this to KNOWN_FEATURES in the script:")
        print(f'    {feature_idx}: "{desc}",')
    
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Find and download SAE encoder features from Neuronpedia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_feature.py "deception"               # Search for deception features
  python find_feature.py "lying" --layer 19        # Search layer 19 only
  python find_feature.py --feature-id 56479        # Download feature 56479
  python find_feature.py --feature-id 56479 -y     # Download without confirmation
  python find_feature.py --browse                  # Show known useful features

Workflow:
  1. Search: python find_feature.py "your query"
  2. Select features to download from results
  3. Features are saved to probes/llama-3.1-8b/
"""
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (e.g., 'deception', 'lying', 'code')"
    )
    parser.add_argument(
        "--feature-id", "-f",
        type=int,
        help="Download a specific feature by ID"
    )
    parser.add_argument(
        "--layer", "-l",
        type=str,
        default=None,
        help="Filter results by layer (e.g., '19' for layer 19 only)"
    )
    parser.add_argument(
        "--browse", "-b",
        action="store_true",
        help="Browse known useful features"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Auto-confirm downloads"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of search results to show (default: 10)"
    )
    
    args = parser.parse_args()
    
    if args.browse:
        browse_features()
    elif args.feature_id:
        direct_download(args.feature_id, auto_confirm=args.yes)
    elif args.query:
        interactive_search(args.query, args.top, layer=args.layer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()