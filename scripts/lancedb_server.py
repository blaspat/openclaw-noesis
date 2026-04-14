#!/usr/bin/env python3
"""
Noesis — LanceDB Python Bridge Server

A JSON-RPC 2.0 server over stdin/stdout that provides LanceDB operations.
The main Noesis plugin uses the @lancedb/lancedb Node.js SDK directly (v0.27+).
This Python bridge is provided as an alternative entry point for environments
where the native Node.js SDK is unavailable or for standalone scripting.

Protocol:
  - Read newline-delimited JSON requests from stdin
  - Write newline-delimited JSON responses to stdout
  - Errors written to stderr

Usage:
  python3 scripts/lancedb_server.py --db ~/.openclaw/noesis/db [--dim 768]

Install deps:
  pip install lancedb pyarrow numpy rank_bm25

JSON-RPC methods:
  upsert        { id, agentId, sessionId, content, chunk, embedding, memoryType, createdAt, sourcePath, checksum, tags }
  vector_search { embedding, topK, agentId?, memoryType?, crossAgent?, nprobe? }
  fts_search    { query, topK, agentId?, memoryType? }
  get_by_id     { id }
  recall        { agentId?, sessionId?, limit? }
  delete        { id }
  count         { agentId? }
  stats         {}
  create_index  {}
  shutdown      {}
"""

import sys
import json
import os
import hashlib
import argparse
import traceback
from pathlib import Path
from typing import Optional

try:
    import lancedb
    import pyarrow as pa
    import numpy as np
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

TABLE_NAME = "memories"


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
    ])


class NoesisServer:
    def __init__(self, db_path: str, dim: int = 768):
        if not LANCEDB_AVAILABLE:
            raise RuntimeError(
                "lancedb not installed. Run: pip install lancedb pyarrow numpy"
            )
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.dim = dim
        self.table = self._open_or_create_table()

    def _open_or_create_table(self):
        existing = self.db.table_names()
        if TABLE_NAME in existing:
            return self.db.open_table(TABLE_NAME)
        schema = build_schema(self.dim)
        return self.db.create_table(TABLE_NAME, schema=schema)

    def upsert(self, entries: list[dict]) -> int:
        if not entries:
            return 0
        checksums = [e["checksum"] for e in entries]
        try:
            existing_df = self.table.search().where(
                f"checksum IN ({', '.join(repr(c) for c in checksums)})"
            ).select(["checksum"]).limit(len(checksums) + 1).to_pandas()
            existing_set = set(existing_df["checksum"].tolist())
        except Exception:
            existing_set = set()

        new_entries = [e for e in entries if e["checksum"] not in existing_set]
        if not new_entries:
            return 0

        data = {
            "id": [e["id"] for e in new_entries],
            "agentId": [e["agentId"] for e in new_entries],
            "sessionId": [e["sessionId"] for e in new_entries],
            "content": [e["content"] for e in new_entries],
            "chunk": [e["chunk"] for e in new_entries],
            "embedding": [
                np.array(e["embedding"], dtype=np.float32) for e in new_entries
            ],
            "memoryType": [e["memoryType"] for e in new_entries],
            "createdAt": [int(e["createdAt"]) for e in new_entries],
            "sourcePath": [e.get("sourcePath", "") for e in new_entries],
            "checksum": [e["checksum"] for e in new_entries],
            "tags": [e.get("tags", []) for e in new_entries],
        }
        self.table.add(pa.table(data, schema=build_schema(self.dim)))
        return len(new_entries)

    def vector_search(
        self,
        embedding: list[float],
        top_k: int = 6,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        cross_agent: bool = False,
        nprobe: int = 16,
    ) -> list[dict]:
        q = (
            self.table.search(np.array(embedding, dtype=np.float32))
            .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags", "_distance"])
            .limit(top_k * 2)
            .nprobes(nprobe)
        )
        filters = []
        if agent_id and not cross_agent:
            filters.append(f"agentId = '{agent_id}'")
        if memory_type:
            filters.append(f"memoryType = '{memory_type}'")
        if filters:
            q = q.where(" AND ".join(filters))

        df = q.to_pandas()
        results = []
        for _, row in df.iterrows():
            dist = float(row.get("_distance", 0))
            results.append({
                "id": row["id"],
                "agentId": row["agentId"],
                "sessionId": row["sessionId"],
                "content": row["content"],
                "memoryType": row["memoryType"],
                "createdAt": int(row["createdAt"]),
                "sourcePath": row["sourcePath"],
                "tags": list(row["tags"]) if row["tags"] is not None else [],
                "score": 1 / (1 + dist),
            })
        return results

    def fts_search(
        self,
        query: str,
        top_k: int = 6,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        try:
            q = (
                self.table.search(query, query_type="fts")
                .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags"])
                .limit(top_k)
            )
            filters = []
            if agent_id:
                filters.append(f"agentId = '{agent_id}'")
            if memory_type:
                filters.append(f"memoryType = '{memory_type}'")
            if filters:
                q = q.where(" AND ".join(filters))

            df = q.to_pandas()
            results = []
            for i, (_, row) in enumerate(df.iterrows()):
                results.append({
                    "id": row["id"],
                    "agentId": row["agentId"],
                    "sessionId": row["sessionId"],
                    "content": row["content"],
                    "memoryType": row["memoryType"],
                    "createdAt": int(row["createdAt"]),
                    "sourcePath": row["sourcePath"],
                    "tags": list(row["tags"]) if row["tags"] is not None else [],
                    "score": 1 / (1 + i),
                })
            return results
        except Exception:
            return []

    def get_by_id(self, entry_id: str) -> Optional[dict]:
        try:
            df = (
                self.table.search()
                .where(f"id = '{entry_id}'")
                .select(["id", "agentId", "sessionId", "content", "chunk", "memoryType",
                         "createdAt", "sourcePath", "checksum", "tags"])
                .limit(1)
                .to_pandas()
            )
            if df.empty:
                return None
            row = df.iloc[0]
            return {
                "id": row["id"],
                "agentId": row["agentId"],
                "sessionId": row["sessionId"],
                "content": row["content"],
                "chunk": row["chunk"],
                "memoryType": row["memoryType"],
                "createdAt": int(row["createdAt"]),
                "sourcePath": row["sourcePath"],
                "checksum": row["checksum"],
                "tags": list(row["tags"]) if row["tags"] is not None else [],
            }
        except Exception:
            return None

    def recall(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        q = (
            self.table.search()
            .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags"])
            .limit(limit)
        )
        filters = []
        if agent_id:
            filters.append(f"agentId = '{agent_id}'")
        if session_id:
            filters.append(f"sessionId = '{session_id}'")
        if filters:
            q = q.where(" AND ".join(filters))

        df = q.to_pandas()
        df = df.sort_values("createdAt", ascending=False)
        results = []
        for _, row in df.iterrows():
            results.append({
                "id": row["id"],
                "agentId": row["agentId"],
                "sessionId": row["sessionId"],
                "content": row["content"],
                "memoryType": row["memoryType"],
                "createdAt": int(row["createdAt"]),
                "sourcePath": row["sourcePath"],
                "tags": list(row["tags"]) if row["tags"] is not None else [],
            })
        return results

    def delete(self, entry_id: str) -> None:
        self.table.delete(f"id = '{entry_id}'")

    def count(self, agent_id: Optional[str] = None) -> int:
        if agent_id:
            return self.table.count_rows(f"agentId = '{agent_id}'")
        return self.table.count_rows()

    def stats(self) -> dict:
        total = self.table.count_rows()
        try:
            df = self.table.search().select(["agentId", "memoryType"]).limit(1_000_000).to_pandas()
            by_agent = df["agentId"].value_counts().to_dict()
            by_type = df["memoryType"].value_counts().to_dict()
        except Exception:
            by_agent = {}
            by_type = {}
        return {
            "totalEntries": total,
            "byAgent": by_agent,
            "byMemoryType": by_type,
        }

    def create_index(self) -> bool:
        try:
            count = self.table.count_rows()
            if count < 256:
                return False
            self.table.create_index(
                "embedding",
                config=lancedb.index.IvfPq(num_sub_vectors=96),
                replace=True,
            )
            return True
        except Exception as e:
            print(f"Index creation warning: {e}", file=sys.stderr)
            return False


def handle_request(server: NoesisServer, request: dict) -> dict:
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id", None)

    try:
        if method == "upsert":
            entries = params if isinstance(params, list) else [params]
            n = server.upsert(entries)
            result = {"inserted": n}

        elif method == "vector_search":
            result = server.vector_search(
                embedding=params["embedding"],
                top_k=params.get("topK", 6),
                agent_id=params.get("agentId"),
                memory_type=params.get("memoryType"),
                cross_agent=params.get("crossAgent", False),
                nprobe=params.get("nprobe", 16),
            )

        elif method == "fts_search":
            result = server.fts_search(
                query=params["query"],
                top_k=params.get("topK", 6),
                agent_id=params.get("agentId"),
                memory_type=params.get("memoryType"),
            )

        elif method == "get_by_id":
            result = server.get_by_id(params["id"])

        elif method == "recall":
            result = server.recall(
                agent_id=params.get("agentId"),
                session_id=params.get("sessionId"),
                limit=params.get("limit", 50),
            )

        elif method == "delete":
            server.delete(params["id"])
            result = {"ok": True}

        elif method == "count":
            result = {"count": server.count(params.get("agentId"))}

        elif method == "stats":
            result = server.stats()

        elif method == "create_index":
            ok = server.create_index()
            result = {"created": ok}

        elif method == "shutdown":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}, "_shutdown": True}

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(e)},
        }


def main():
    parser = argparse.ArgumentParser(description="Noesis LanceDB Python bridge server")
    parser.add_argument("--db", default=os.path.expanduser("~/.openclaw/noesis/db"),
                        help="LanceDB path (default: ~/.openclaw/noesis/db)")
    parser.add_argument("--dim", type=int, default=768,
                        help="Embedding dimension (default: 768 for nomic-embed-text)")
    args = parser.parse_args()

    print(json.dumps({"status": "ready", "db": args.db, "dim": args.dim}), flush=True)

    server = NoesisServer(args.db, dim=args.dim)
    print(json.dumps({"status": "connected"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"jsonrpc": "2.0", "id": None,
                              "error": {"code": -32700, "message": f"Parse error: {e}"}}), flush=True)
            continue

        response = handle_request(server, request)
        shutdown = response.pop("_shutdown", False)
        print(json.dumps(response), flush=True)
        if shutdown:
            break


if __name__ == "__main__":
    main()
