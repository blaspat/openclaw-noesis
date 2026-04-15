#!/usr/bin/env python3
"""
Noesis — Standalone Memory Import CLI

Imports markdown memory files from OpenClaw agents into LanceDB.
Scans MEMORY.md and memory/*.md files, chunks, embeds via Ollama, and upserts.
Fully idempotent — duplicate content is skipped via SHA-256 checksum.

Install deps:
  pip install lancedb pyarrow numpy requests

Usage:
  # Import a specific agent
  python3 scripts/import_memory.py --agent claire

  # Import all agents
  python3 scripts/import_memory.py --all

  # Import with custom settings
  python3 scripts/import_memory.py --agent claire \\
    --db ~/.openclaw/noesis/db \\
    --ollama http://localhost:11434 \\
    --model nomic-embed-text \\
    --chunk-size 512 \\
    --chunk-overlap 64
"""

import sys
import os
import json
import hashlib
import argparse
import uuid
import re
from pathlib import Path
from typing import Optional

import pyarrow as pa  # Safety net — used at module level even if try block fails

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import lancedb
    import numpy as np
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


# ─── embedding ────────────────────────────────────────────────────────────────

def get_embedding(text: str, endpoint: str, model: str) -> list[float]:
    resp = requests.post(
        f"{endpoint}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("embedding")
    if not embedding:
        raise ValueError("Ollama returned no embedding")
    return embedding


def check_ollama(endpoint: str, model: str) -> str:
    """Check Ollama connectivity and ensure model is available. Returns resolved model name."""
    resp = requests.get(f"{endpoint}/api/tags", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    available = {m["name"].split(":")[0] for m in data.get("models", [])}

    if model.split(":")[0] not in available:
        print(f"Model '{model}' not found. Pulling...")
        pull_resp = requests.post(
            f"{endpoint}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        pull_resp.raise_for_status()
        print(f"Model '{model}' ready.")

    return model


# ─── text processing ──────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c]


def split_by_headings(text: str) -> list[str]:
    """Split markdown by ## headings."""
    sections = []
    current = []
    for line in text.split("\n"):
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def content_checksum(content: str, agent_id: str) -> str:
    return hashlib.sha256((content + agent_id).encode()).hexdigest()


def infer_memory_type(content: str) -> str:
    lower = content.lower()
    if re.search(r'\bdecid|decision|chose|choice\b', lower):
        return "decision"
    if re.search(r'\bprefer|always|never|style|like to\b', lower):
        return "preference"
    if re.search(r'\bsession|today|yesterday|this morning\b', lower):
        return "context"
    return "fact"


# ─── lancedb ─────────────────────────────────────────────────────────────────

def build_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("agentId", pa.string(), nullable=False),
        pa.field("sessionId", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("chunk", pa.string(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False),
        pa.field("memoryType", pa.string(), nullable=False),
        pa.field("createdAt", pa.int64(), nullable=False),
        pa.field("sourcePath", pa.string(), nullable=False),
        pa.field("checksum", pa.string(), nullable=False),
        pa.field("tags", pa.list_(pa.string()), nullable=False),
        pa.field("expiresAt", pa.int64(), nullable=True),
        pa.field("priority", pa.string(), nullable=True),
    ])


def open_or_create_table(conn, dim: int):
    tables = conn.list_tables()
    if "memories" in tables:
        return conn.open_table("memories")
    schema = build_schema(dim)
    return conn.create_table("memories", schema=schema)


def upsert_entries(table, entries: list[dict], schema: pa.Schema) -> int:
    if not entries:
        return 0
    checksums = [e["checksum"] for e in entries]
    try:
        existing_df = (
            table.search()
            .where(f"checksum IN ({', '.join(repr(c) for c in checksums)})")
            .select(["checksum"])
            .limit(len(checksums) + 1)
            .to_pandas()
        )
        existing = set(existing_df["checksum"].tolist())
    except Exception:
        existing = set()

    new_entries = [e for e in entries if e["checksum"] not in existing]
    if not new_entries:
        return 0

    data = {
        "id": [e["id"] for e in new_entries],
        "agentId": [e["agentId"] for e in new_entries],
        "sessionId": ["migration"] * len(new_entries),
        "content": [e["content"] for e in new_entries],
        "chunk": [e["chunk"] for e in new_entries],
        "embedding": [np.array(e["embedding"], dtype=np.float32) for e in new_entries],
        "memoryType": [e["memoryType"] for e in new_entries],
        "createdAt": [int(e["createdAt"]) for e in new_entries],
        "expiresAt": [int(e.get("expiresAt", 0)) for e in new_entries],
        "priority": [e.get("priority", "") for e in new_entries],
        "sourcePath": [e["sourcePath"] for e in new_entries],
        "checksum": [e["checksum"] for e in new_entries],
        "tags": [[] for _ in new_entries],
    }
    table.add(pa.table(data, schema=schema))
    return len(new_entries)


# ─── import ───────────────────────────────────────────────────────────────────

def import_agent(
    agent_id: str,
    conn,
    endpoint: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    dim: int,
    verbose: bool = True,
) -> dict:
    agent_base = Path.home() / ".openclaw" / "agents" / agent_id / "workspace"
    files = []

    memory_md = agent_base / "MEMORY.md"
    if memory_md.exists():
        files.append(memory_md)

    memory_dir = agent_base / "memory"
    if memory_dir.exists():
        files.extend(sorted(memory_dir.glob("*.md")))

    if not files:
        print(f"  No files found for agent '{agent_id}'")
        return {"agent": agent_id, "indexed": 0, "skipped": 0, "errors": 0}

    print(f"  Found {len(files)} file(s) for agent '{agent_id}'")

    # Collect all chunks
    all_chunks = []
    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8")
            sections = split_by_headings(raw)
            for section in sections:
                if not section.strip():
                    continue
                chunks = chunk_text(section, chunk_size, chunk_overlap)
                for chunk in chunks:
                    all_chunks.append({
                        "content": section,
                        "chunk": chunk,
                        "sourcePath": str(file_path),
                        "memoryType": infer_memory_type(section),
                    })
        except Exception as e:
            print(f"  Error reading {file_path}: {e}", file=sys.stderr)

    if not all_chunks:
        return {"agent": agent_id, "indexed": 0, "skipped": 0, "errors": 0}

    # Get schema/dim from first embedding
    first_embed = get_embedding(all_chunks[0]["chunk"], endpoint, model)
    actual_dim = len(first_embed)
    schema = build_schema(actual_dim)

    # Open/create table with correct dim
    table = open_or_create_table(conn, actual_dim)

    BATCH = 10
    indexed = 0
    skipped = 0
    errors = 0
    import time

    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        try:
            # Embed first chunk's text (already have first), rest via API
            embeddings = []
            for j, item in enumerate(batch):
                if i == 0 and j == 0:
                    embeddings.append(first_embed)
                else:
                    embeddings.append(get_embedding(item["chunk"], endpoint, model))

            entries = []
            for j, item in enumerate(batch):
                checksum = content_checksum(item["content"] + item["chunk"], agent_id)
                entries.append({
                    "id": str(uuid.uuid4()),
                    "agentId": agent_id,
                    "content": item["content"],
                    "chunk": item["chunk"],
                    "embedding": embeddings[j],
                    "memoryType": item["memoryType"],
                    "createdAt": int(time.time() * 1000),
                    "sourcePath": item["sourcePath"],
                    "checksum": checksum,
                })

            n = upsert_entries(table, entries, schema)
            indexed += n
            skipped += len(entries) - n
            if verbose:
                print(f"  Batch {i // BATCH + 1}/{(len(all_chunks) + BATCH - 1) // BATCH}: +{n} indexed, {len(entries) - n} skipped")
        except Exception as e:
            print(f"  Batch error: {e}", file=sys.stderr)
            errors += len(batch)

    print(f"  Done: {indexed} indexed, {skipped} skipped, {errors} errors")
    return {"agent": agent_id, "indexed": indexed, "skipped": skipped, "errors": errors}


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if not REQUESTS_AVAILABLE:
        print("Error: 'requests' not installed. Run: pip install requests", file=sys.stderr)
        sys.exit(1)
    if not LANCEDB_AVAILABLE:
        print("Error: 'lancedb' not installed. Run: pip install lancedb pyarrow numpy", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Noesis — Import markdown memory files into LanceDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--agent", help="Agent ID to import (e.g. claire)")
    parser.add_argument("--all", action="store_true", help="Import all agents")
    parser.add_argument("--db", default=os.path.expanduser("~/.openclaw/noesis/db"),
                        help="LanceDB path (default: ~/.openclaw/noesis/db)")
    parser.add_argument("--ollama", default="http://localhost:11434",
                        help="Ollama endpoint (default: http://localhost:11434)")
    parser.add_argument("--model", default="nomic-embed-text",
                        help="Embedding model (default: nomic-embed-text)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Chunk size in words (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64,
                        help="Chunk overlap in words (default: 64)")
    parser.add_argument("--quiet", action="store_true", help="Suppress batch-level output")
    args = parser.parse_args()

    if not args.agent and not args.all:
        parser.error("Specify --agent <id> or --all")

    print(f"Connecting to Ollama at {args.ollama}...")
    try:
        model = check_ollama(args.ollama, args.model)
        print(f"Using embedding model: {model}")
    except Exception as e:
        print(f"Error: Cannot connect to Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.db, exist_ok=True)
    conn = lancedb.connect(args.db)
    print(f"LanceDB connected: {args.db}")

    results = []
    if args.agent:
        print(f"\nImporting agent: {args.agent}")
        result = import_agent(
            args.agent, conn, args.ollama, model,
            args.chunk_size, args.chunk_overlap, 768,
            verbose=not args.quiet,
        )
        results.append(result)
    else:
        agents_dir = Path.home() / ".openclaw" / "agents"
        if not agents_dir.exists():
            print(f"No agents directory found at {agents_dir}", file=sys.stderr)
            sys.exit(1)
        agents = sorted([d.name for d in agents_dir.iterdir() if d.is_dir()])
        print(f"\nImporting {len(agents)} agent(s): {', '.join(agents)}")
        for agent_id in agents:
            print(f"\nAgent: {agent_id}")
            result = import_agent(
                agent_id, conn, args.ollama, model,
                args.chunk_size, args.chunk_overlap, 768,
                verbose=not args.quiet,
            )
            results.append(result)

    # Summary
    total_indexed = sum(r["indexed"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    print(f"\n{'─' * 40}")
    print(f"Summary: {total_indexed} indexed, {total_skipped} skipped, {total_errors} errors")
    print(f"Database: {args.db}")

    sys.exit(0 if total_errors == 0 else 1)


if __name__ == "__main__":
    main()
