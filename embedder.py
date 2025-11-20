# app/utils/embedder.py

import uuid
import hashlib
from typing import List, Dict, Any
import asyncio
import tiktoken
import re
import logging
from tqdm import tqdm

from app.config import (
    QDRANT_ENDPOINT,
    QDRANT_API_KEY,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_EMBEDDINGS_MODEL,
    AZURE_OPENAI_EMBEDDINGS_API_VERSION
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 🔐 Hash API key to store creator reference
def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


# 🧠 Structure-aware Markdown chunker (same logic, logging added)
def smart_chunk_markdown(
    markdown: str,
    max_tokens: int = 400,
    overlap_tokens: int = 25
) -> List[str]:

    logger.info("🔪 Chunking started...")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenization progress bar
    tokens = tokenizer.encode(markdown)
    total_tokens = len(tokens)
    logger.info(f"🧮 Total tokens detected: {total_tokens}")

    token_bar = tqdm(total=total_tokens, desc="Tokenizing", unit="tok")

    # PAGE SPLITTING
    page_splits = re.split(r"(?:\f|---PAGE BREAK---|==== PAGE \d+ ====)", markdown)
    pages = [p.strip() for p in page_splits if p.strip()]
    logger.info(f"📄 Pages detected: {len(pages)}")

    final_chunks = []

    # Initialize chunk progress bar
    chunk_bar = tqdm(desc="Chunking", unit="chunk", total=0)

    for page in pages:
        header_blocks = re.split(r"(?=^#{1,3}\s)", page, flags=re.MULTILINE)
        header_blocks = [hb.strip() for hb in header_blocks if hb.strip()]
        logger.info(f"📌 Headers detected: {len(header_blocks)}")

        for block in header_blocks:
            block_tokens = tokenizer.encode(block)
            token_bar.update(len(block_tokens))

            if len(block_tokens) <= max_tokens:
                final_chunks.append(block)
                chunk_bar.total += 1
                chunk_bar.update(1)
                chunk_bar.refresh()
                continue

            paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
            paragraph_buffer = ""

            for para in paragraphs:
                p_tokens = tokenizer.encode(para)
                token_bar.update(len(p_tokens))

                if len(p_tokens) <= max_tokens:
                    if tokenizer.encode(paragraph_buffer + "\n\n" + para) <= max_tokens:
                        paragraph_buffer += ("\n\n" + para if paragraph_buffer else para)
                    else:
                        final_chunks.append(paragraph_buffer)
                        chunk_bar.total += 1
                        chunk_bar.update(1)
                        chunk_bar.refresh()
                        paragraph_buffer = para
                    continue

                sentences = re.split(r"(?<=[.!?])\s+", para)
                sentence_buffer = ""

                for sent in sentences:
                    s_tokens = tokenizer.encode(sent)
                    token_bar.update(len(s_tokens))

                    if len(s_tokens) > max_tokens:
                        logger.warning("⚠️ Hard token split required")

                        start = 0
                        while start < len(s_tokens):
                            end = start + max_tokens
                            chunk_tokens = s_tokens[start:end]
                            chunk_text = tokenizer.decode(chunk_tokens)

                            final_chunks.append(chunk_text)
                            chunk_bar.total += 1
                            chunk_bar.update(1)
                            chunk_bar.refresh()

                            start += max_tokens - overlap_tokens
                        continue

                    if tokenizer.encode(sentence_buffer + " " + sent) <= max_tokens:
                        sentence_buffer += (" " + sent if sentence_buffer else sent)
                    else:
                        final_chunks.append(sentence_buffer)
                        chunk_bar.total += 1
                        chunk_bar.update(1)
                        chunk_bar.refresh()
                        sentence_buffer = sent

                if sentence_buffer:
                    final_chunks.append(sentence_buffer)
                    chunk_bar.total += 1
                    chunk_bar.update(1)
                    chunk_bar.refresh()

            if paragraph_buffer:
                final_chunks.append(paragraph_buffer)
                chunk_bar.total += 1
                chunk_bar.update(1)
                chunk_bar.refresh()

    token_bar.close()
    chunk_bar.close()

    logger.info(f"✅ Total final chunks created: {len(final_chunks)}")

    return final_chunks


# 🧬 Deterministic ID using hash of content and metadata
def deterministic_id(chunk: str, metadata: Dict[str, Any]) -> str:
    base_string = chunk + "|" + str(sorted(metadata.items()))
    hash_value = hashlib.sha256(base_string.encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=hash_value))


# ⚡ Async OpenAI embedding function with progress logging
async def embed_chunk(text: str) -> Dict[str, Any]:
    logger.info(f"🧠 Embedding chunk ({len(text)} chars)...")

    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_EMBEDDINGS_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    try:
        response = await client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDINGS_MODEL,
            input=text
        )

        tokens = response.usage.total_tokens
        logger.info(f"🟢 Embedding OK — tokens used: {tokens}")

        return {
            "embedding": response.data[0].embedding,
            "tokens_used": tokens
        }

    except Exception as e:
        logger.error(f"🔴 Embedding failed: {e}", exc_info=True)
        return {
            "embedding": [0.0] * 1536,
            "tokens_used": 0
        }


# 🚀 Main async upsert to Qdrant with tqdm progress bars
async def upsert_to_qdrant(
    markdown: str,
    metadata: Dict[str, Any],
    collection_name: str,
    api_key: str
) -> Dict[str, Any]:

    logger.info("🚀 Upsert to Qdrant started...")

    if not isinstance(metadata, dict):
        raise ValueError("❌ Metadata must be a dictionary.")

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct, VectorParams, Distance

    qdrant = QdrantClient(
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        https=True,
        timeout=60,
        verify=False
    )

    collections = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in collections:
        logger.info("📂 Creating missing Qdrant collection...")
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    metadata['creator_api_key_hash'] = hash_api_key(api_key)

    logger.info("📖 Running chunker...")
    chunks = smart_chunk_markdown(markdown)

    if not chunks:
        return {
            "message": "✅ Empty document processed",
            "collection": collection_name,
            "chunks": 0,
            "metadata": metadata,
            "tokens_used": 0
        }

    # Embedding with progress bar
    embed_results = []
    embed_bar = tqdm(total=len(chunks), desc="Embedding", unit="chunk")

    for chunk in chunks:
        result = await embed_chunk(chunk)
        embed_results.append(result)
        embed_bar.update(1)

    embed_bar.close()

    # Prepare points
    points = []
    total_tokens = 0

    for i, (chunk, embed_result) in enumerate(zip(chunks, embed_results)):
        embedding = embed_result["embedding"]
        tokens_used = embed_result["tokens_used"]
        total_tokens += tokens_used

        payload = {
            **{k: v for k, v in metadata.items() if v is not None},
            "chunkIndex": i,
            "text": chunk
        }

        point_id = deterministic_id(chunk, payload)
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    # Qdrant upsert with tqdm
    logger.info("⬆️ Upserting vectors to Qdrant...")

    upsert_bar = tqdm(total=len(points), desc="Upserting", unit="vec")
    BATCH = 25

    try:
        for i in range(0, len(points), BATCH):
            batch = points[i:i+BATCH]
            qdrant.upsert(collection_name=collection_name, points=batch)
            upsert_bar.update(len(batch))

        upsert_bar.close()
        message = "✅ Document processed and embedded successfully"

    except Exception as e:
        logger.error(f"🔴 Qdrant upsert failed: {e}", exc_info=True)
        message = f"⚠️ Partial success - embeddings generated but upsert failed: {str(e)}"
        upsert_bar.close()

    return {
        "message": message,
        "collection": collection_name,
        "chunks": len(points),
        "metadata": metadata,
        "tokens_used": total_tokens
    }


# 🔄 Batch processing helper
async def batch_upsert_to_qdrant(
    documents: List[Dict[str, Any]],
    collection_name: str,
    api_key: str,
    global_metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:

    logger.info(f"📦 Batch upsert started — documents: {len(documents)}")

    tasks = []

    for doc in documents:
        merged_metadata = {
            **(global_metadata or {}),
            **(doc.get("metadata", {}) or {})
        }

        tasks.append(
            upsert_to_qdrant(
                markdown=doc["markdown"],
                metadata=merged_metadata,
                collection_name=collection_name,
                api_key=api_key
            )
        )

    results = await asyncio.gather(*tasks)
    logger.info("📨 Batch upsert complete")
    return results
