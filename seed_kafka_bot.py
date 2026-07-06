"""
Seed script: provision kafka-performance-bot for the platform tenant.

Steps:
  1. Generate a stable KB UUID for "kafka-performance-bot" / "platform"
  2. Embed all knowledge chunks via Gemini embedding API
  3. Upsert into Supabase pgvector
  4. Upsert the bot + KB entry into MongoDB customerService

Run from the mcp-server directory:
    python seed_kafka_bot.py
"""

import asyncio
import uuid
import sys
import os

# Make sure project modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(".env", override=True)

# ── Knowledge content ─────────────────────────────────────────────────────────

KNOWLEDGE_CHUNKS = [

    # ── Shielva system overview ───────────────────────────────────────────────
    {
        "title": "Shielva Logstream Cluster Overview",
        "content": (
            "Shielva runs a 3-node Logstream (Kafka-compatible) cluster. "
            "Admin REST API ports: 9644 (primary), 19645, 19646. "
            "Kafka wire protocol port: 9092. "
            "Authentication: SASL_SSL with SCRAM-SHA-256. "
            "TLS minimum version: TLS 1.3 (rustls). "
            "Replication factor default: 3 (one copy per broker). "
            "The cluster uses Raft consensus for metadata; partition leadership "
            "is stable under normal conditions but may shift on broker restart."
        ),
    },
    {
        "title": "Shielva Topic Naming and Defaults",
        "content": (
            "Topic naming convention: <service>.<domain>.<event-type>, e.g. "
            "shielva.notifications, shielva.devops.build.completed, "
            "shielva.fix.intents. DLQ suffix is .dlq (e.g. shielva.notifications.dlq). "
            "Default topic config: retention.ms=604800000 (7 days), "
            "retention.bytes=-1 (unlimited), segment.bytes=1073741824 (1 GB), "
            "compression.type=producer, cleanup.policy=delete. "
            "Most topics have 6 partitions for parallelism across 3 brokers (2 per broker)."
        ),
    },
    {
        "title": "Shielva Default Producer Configuration",
        "content": (
            "The shielva-event-streaming-service producer uses: "
            "acks=all (strongest durability), compression=lz4, linger_ms=10, "
            "max_batch_size=16384 bytes, request_timeout_ms=30000. "
            "Retry policy: max_attempts=5, backoff_base_ms=100. "
            "Circuit breaker: fail_threshold=5, reset_seconds=30. "
            "The resilience test producer uses AIOKafka with api_version=2.0 "
            "(forces SaslHandshake v1 + Kafka-framed SaslAuthenticate). "
            "Ephemeral SCRAM credentials (temp-SCRAM) are minted per test via "
            "POST /v1/principals/me/issue-temp-scram with a 1800-second TTL."
        ),
    },
    {
        "title": "Shielva ACL and Authorization Model",
        "content": (
            "Logstream uses Kafka-wire ACLs checked on every Produce/Fetch/Describe. "
            "ClusterAdmin role gets: All on Cluster:kafka-cluster, All on Topic:*, "
            "All on Group:*. TenantUser role gets: scoped to <tenant>. prefix topics "
            "and groups, plus Describe on cluster. When ACLs are empty the cluster is "
            "open (deny-by-default activates on first ACL write). "
            "Temp-SCRAM credentials inherit the parent principal's role and get ACLs "
            "auto-granted at mint time."
        ),
    },
    {
        "title": "Shielva Resilience Test Parameters",
        "content": (
            "A resilience test has: topic (str), target_tps (int), duration_s (default 30), "
            "message_size_bytes (default 256), batch_size (default 100), linger_ms (default 0), "
            "compression_type (none/lz4/snappy/gzip/zstd), kafka_batch_bytes (default 16384). "
            "Achievement % = (avg_actual_tps / target_tps) * 100. "
            "A test is healthy when achievement >= 95% and errors == 0. "
            "Common failure modes: SASL auth failure (wrong creds or ACL not yet replicated), "
            "topic not found (404 on produce), broker unreachable (connection refused). "
            "Errors > 10% of sends = degraded. Errors > 50% = failed."
        ),
    },

    # ── Kafka throughput best practices ──────────────────────────────────────
    {
        "title": "Kafka Throughput Optimisation: Producer Batching",
        "content": (
            "To maximise TPS increase batch.size (kafka_batch_bytes) to 65536–262144 bytes. "
            "Set linger_ms=5–20 to allow batches to fill before sending. "
            "Use compression: lz4 is the best balance of CPU cost and ratio for most payloads; "
            "snappy is lower CPU; gzip gives best ratio but highest CPU; zstd is best overall "
            "ratio with moderate CPU. Compression helps most for text/JSON payloads > 1 KB. "
            "For small messages (< 256 bytes) compression overhead may exceed savings — "
            "test both with and without. "
            "Increase max_batch_size to 1 MiB for high-throughput pipelines."
        ),
    },
    {
        "title": "Kafka Throughput Optimisation: Partitions and Parallelism",
        "content": (
            "More partitions = more parallelism for consumers. Recommended: "
            "partitions = target_peak_tps / single_partition_throughput. "
            "A single partition can typically sustain 10–50 MB/s depending on broker hardware. "
            "For a 3-broker cluster, 6–12 partitions distributes load evenly (2–4 per broker). "
            "Increasing partitions is non-reversible downward — only increase when needed. "
            "Replication factor of 3 on a 3-broker cluster ensures durability and allows "
            "one broker to fail without data loss or availability impact."
        ),
    },
    {
        "title": "Kafka Throughput Optimisation: acks and Durability Trade-off",
        "content": (
            "acks=all (or -1): leader waits for all in-sync replicas — strongest durability, "
            "highest latency (adds one replication round-trip ~1–5 ms). "
            "acks=1: leader acks immediately — faster but data loss possible on leader crash. "
            "acks=0: fire-and-forget — maximum throughput, no durability guarantee. "
            "For resilience testing: acks=all is the baseline (matches production). "
            "If achievement is < 80% with acks=all, test with acks=1 to isolate whether "
            "replication is the bottleneck. "
            "min.insync.replicas=2 with acks=all is the recommended production combination."
        ),
    },

    # ── Latency best practices ────────────────────────────────────────────────
    {
        "title": "Kafka Latency Optimisation",
        "content": (
            "p99 latency > 100 ms usually indicates: "
            "(1) linger_ms too high — reduce to 0–5 ms for latency-sensitive workloads; "
            "(2) batch overflowing — increase batch.size so batches send on size not time; "
            "(3) broker GC pauses — check broker heap and GC logs; "
            "(4) network congestion — check producer/broker network saturation. "
            "p50 latency > 20 ms on a local cluster (same host) indicates a configuration issue. "
            "p50 < 5 ms and p99 < 50 ms is healthy for a local 3-broker setup. "
            "For sub-millisecond latency: linger_ms=0, acks=1, no compression, large batch.size."
        ),
    },

    # ── Error handling ────────────────────────────────────────────────────────
    {
        "title": "Kafka Producer Error Handling: Retries and Idempotency",
        "content": (
            "retries=5 (default) with retry.backoff.ms=100 covers transient leader elections. "
            "enable.idempotence=True prevents duplicate messages on retry (requires acks=all). "
            "delivery.timeout.ms should be > request.timeout.ms * (retries + 1). "
            "If all errors cluster at the start of the test: likely auth/ACL issue. "
            "If errors are spread across the run: likely broker overload or network instability. "
            "If errors only appear at peak TPS: likely broker back-pressure — reduce target_tps "
            "or increase batch.size + linger_ms to smooth out the burst."
        ),
    },
    {
        "title": "Kafka Circuit Breaker Pattern",
        "content": (
            "Shielva's circuit breaker: opens after 5 consecutive failures per partition, "
            "stays open for 30 seconds, then half-opens (allows one probe). "
            "When the circuit is open, produces to that partition return an error immediately "
            "without attempting the Kafka call — prevents thundering-herd on a degraded broker. "
            "To reset manually: POST /v1/partitions/{topic}/{partition}/circuit-reset via the "
            "Logstream admin API. "
            "A resilience test that hits the circuit breaker will show error bursts every 30s."
        ),
    },

    # ── Topic configuration ───────────────────────────────────────────────────
    {
        "title": "Kafka Topic Configuration Tuning",
        "content": (
            "retention.ms: time to keep messages. Default 7 days (604800000 ms). "
            "Reduce for high-volume topics where consumers are real-time (e.g. 3600000 = 1 hr). "
            "retention.bytes: max total log size per partition. -1 = unlimited. "
            "Set to control disk usage: 1073741824 (1 GB) per partition is a safe default. "
            "segment.bytes: log segment file size. Smaller segments = more files but faster "
            "retention cleanup. Default 1 GB. Reduce to 256 MB for topics with frequent compaction. "
            "cleanup.policy=delete: time/size based retention (default). "
            "cleanup.policy=compact: keeps only latest value per key (for state topics). "
            "compression.type=producer: broker stores whatever the producer sent (recommended). "
            "compression.type=lz4: broker re-compresses if producer didn't — avoid, adds CPU."
        ),
    },
    {
        "title": "Kafka Consumer Group Tuning",
        "content": (
            "max.poll.records=500 is a good default. Reduce to 100 for slow per-message processing. "
            "fetch.min.bytes=1024: consumer waits for at least 1 KB before returning — reduces "
            "fetch round-trips at the cost of latency. "
            "fetch.max.wait.ms=500: max time consumer waits for fetch.min.bytes to fill. "
            "For resilience tests the consumer side is not involved — focus on producer config. "
            "Consumer group rebalance is triggered on member join/leave — avoid frequent "
            "consumer restarts during load tests as it stalls consumption for up to "
            "max.poll.interval.ms (default 5 min)."
        ),
    },

    # ── Analysis guide ────────────────────────────────────────────────────────
    {
        "title": "Interpreting Resilience Test Results: Healthy",
        "content": (
            "Healthy test: achievement >= 95%, errors == 0, p99 < 100 ms. "
            "No configuration changes recommended. The topic and producer are well-tuned "
            "for the given TPS target and message size. "
            "Consider increasing target_tps by 2–5x to find the ceiling, "
            "or test with larger message sizes (e.g. 1024, 4096 bytes) to stress serialization."
        ),
    },
    {
        "title": "Interpreting Resilience Test Results: Degraded TPS",
        "content": (
            "Achievement 50–95%: throughput is below target. Investigate: "
            "(1) Increase batch_size from 100 to 500–1000 messages per batch; "
            "(2) Increase kafka_batch_bytes from 16384 to 131072 (128 KB); "
            "(3) Set linger_ms=10–20 to allow batches to fill; "
            "(4) Enable lz4 compression to reduce network bytes; "
            "(5) Check if partitions < target_tps / 10000 — may need more partitions. "
            "The recommended fix scope is 'producer_config_hint' — no topic-level change needed."
        ),
    },
    {
        "title": "Interpreting Resilience Test Results: High Error Rate",
        "content": (
            "Errors > 10%: "
            "If errors are at the very start: ACL or SCRAM credential not yet replicated — "
            "increase SCRAM sync delay or retry the test after 5s. "
            "If errors throughout: check broker logs for 'authorization denied' (ACL missing), "
            "'RecordTooLargeException' (message_size_bytes > message.max.bytes), or "
            "'NotLeaderForPartitionException' (leadership election in progress). "
            "If only specific partitions error: partition-level issue, check circuit breaker. "
            "Fix scope for ACL issues: 'topic_config'; for message size: producer_config_hint."
        ),
    },
    {
        "title": "Interpreting Resilience Test Results: High Latency",
        "content": (
            "p99 > 200 ms with low errors: "
            "(1) Broker disk I/O saturation — check broker metrics; "
            "(2) linger_ms too high — reduce to 0 for latency tests; "
            "(3) Large batches waiting to fill — reduce batch_size or kafka_batch_bytes; "
            "(4) Network round-trip — expected if producer and broker are on different hosts; "
            "(5) Compression overhead — try disabling compression for a baseline. "
            "p50 high but p99 low: steady-state latency issue, not burst spikes. "
            "p50 low but p99 high: occasional slow outliers — check for GC pauses or "
            "producer queue backlog."
        ),
    },
]

# ── Stable IDs (deterministic so re-running the script is idempotent) ─────────
# Tenant ID is REQUIRED at runtime — no hardcoded default. Pass via
# --tenant-id flag or SHIELVA_SEED_TENANT_ID env var. The kafka KB is
# only useful for tenants who actually operate a Logstream cluster, so
# the seeder demands an explicit target.
BOT_ID = "kafka-performance-bot"


def _resolve_tenant_id() -> str:
    """Look up the seeding target tenant id.

    Priority:
      1. ``--tenant-id <id>`` CLI argument.
      2. ``SHIELVA_SEED_TENANT_ID`` environment variable.
    Raises ``SystemExit`` with a helpful message otherwise.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Seed kafka-performance-bot")
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("SHIELVA_SEED_TENANT_ID", ""),
        help="Target tenant id (REQUIRED). No hardcoded default — multi-tenant rule.",
    )
    args, _ = parser.parse_known_args()
    if not args.tenant_id:
        raise SystemExit(
            "seed_kafka_bot.py: tenant id required. Pass --tenant-id <id> "
            "or set SHIELVA_SEED_TENANT_ID. Hardcoded 'platform' default removed "
            "for multi-tenant safety."
        )
    return args.tenant_id


def _make_kb_id(tenant_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}.{BOT_ID}.kafka-kb-v1"))


# ── Main seeding logic ────────────────────────────────────────────────────────

async def embed_chunks(chunks: list[dict], gemini_api_key: str) -> list[dict]:
    """Embed each chunk using Gemini embedding API."""
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)

    results = []
    for i, chunk in enumerate(chunks):
        text = f"{chunk['title']}\n\n{chunk['content']}"
        resp = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            output_dimensionality=768,
        )
        embedding = resp["embedding"]
        results.append({**chunk, "embedding": embedding})
        print(f"  [{i+1}/{len(chunks)}] Embedded: {chunk['title'][:60]}")
    return results


async def upsert_to_supabase(
    embedded_chunks: list[dict], db_url: str, tenant_id: str, kb_id: str,
):
    """Create collection and upsert all chunks into Supabase pgvector."""
    import vecs

    client = vecs.create_client(db_url)
    collection_name = f"shielva_kb_{tenant_id}_{kb_id}".replace("-", "_")
    # Truncate if over 63 chars
    if len(collection_name) > 63:
        import hashlib
        h = hashlib.sha256(f"{tenant_id}_{kb_id}".encode()).hexdigest()[:32]
        collection_name = f"shielva_kb_{h}"

    print(f"\n  Supabase collection: {collection_name}")
    col = client.get_or_create_collection(name=collection_name, dimension=768)

    records = []
    for i, chunk in enumerate(embedded_chunks):
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{kb_id}.{chunk['title']}"))
        text   = f"{chunk['title']}\n\n{chunk['content']}"
        meta   = {
            "content":   text,
            "title":     chunk["title"],
            "tenant_id": tenant_id,
            "kb_id":     kb_id,
            "source":    "seed_kafka_bot",
        }
        records.append((doc_id, chunk["embedding"], meta))

    col.upsert(records=records)
    col.create_index()
    client.disconnect()
    print(f"  Upserted {len(records)} chunks into Supabase.")


async def upsert_bot_to_mongo(mongo_url: str, tenant_id: str, kb_id: str):
    """Create/update the given tenant's kafka-performance-bot in MongoDB."""
    import certifi
    from motor.motor_asyncio import AsyncIOMotorClient

    tls = {"tlsCAFile": certifi.where()} if "mongodb+srv" in mongo_url else {}
    client = AsyncIOMotorClient(mongo_url, **tls)
    db     = client["CustomerProfile"]
    col    = db["customerService"]

    bot_doc = {
        "id":          BOT_ID,
        "name":        "Kafka Performance Bot",
        "description": "Analyses resilience test metrics and recommends Kafka tuning fixes.",
        "kbs": [{"id": kb_id, "name": "kafka-best-practices"}],
        "prompt_config": {
            "system_prompt": (
                "You are a Kafka performance analysis expert for the Shielva platform. "
                "Analyse the supplied resilience test metrics and produce actionable tuning recommendations. "
                "Use your knowledge base for Kafka best practices and Shielva system specifics. "
                "Always output valid HTML (no markdown). "
                "Always append the [FIXES_JSON]...[/FIXES_JSON] block."
            ),
            "tool_instructions": "",
        },
    }

    # Upsert: if tenant exists update its bots array, else create.
    existing = await col.find_one({"tenant_id": tenant_id})
    if existing:
        await col.update_one(
            {"tenant_id": tenant_id},
            {"$pull": {"bots": {"id": BOT_ID}}},
        )
        await col.update_one(
            {"tenant_id": tenant_id},
            {"$push": {"bots": bot_doc}},
        )
        print(f"  Updated existing tenant {tenant_id!r} — bot upserted.")
    else:
        await col.insert_one({
            "tenant_id": tenant_id,
            "bots":      [bot_doc],
        })
        print(f"  Created tenant {tenant_id!r} with kafka-performance-bot.")

    client.close()


async def main():
    from config.settings import get_settings

    tenant_id = _resolve_tenant_id()
    kb_id = _make_kb_id(tenant_id)

    settings = get_settings()

    print(f"KB_ID (stable):  {kb_id}")
    print(f"Tenant:          {tenant_id}")
    print(f"Bot:             {BOT_ID}")
    print(f"Chunks to embed: {len(KNOWLEDGE_CHUNKS)}")
    print()

    # 1. Embed
    print("Step 1 — Embedding knowledge chunks via Gemini...")
    embedded = await embed_chunks(
        KNOWLEDGE_CHUNKS, settings.gemini_api_key.get_secret_value(),
    )

    # 2. Upsert to pgvector
    print("\nStep 2 — Upserting into pgvector...")
    await upsert_to_supabase(
        embedded,
        settings.mcp_vector_db_url.get_secret_value(),
        tenant_id,
        kb_id,
    )

    # 3. Register bot in MongoDB
    print("\nStep 3 — Registering bot in MongoDB...")
    await upsert_bot_to_mongo(
        settings.mongodb_url.get_secret_value(), tenant_id, kb_id,
    )

    print("\n✓ Done. kafka-performance-bot provisioned with Kafka KB.")
    print(f"  KB collection: shielva_kb_{tenant_id.replace('-','_')}_{kb_id.replace('-','_')}")
    print(f"  Run a resilience test to verify AI analysis uses the KB context.")


if __name__ == "__main__":
    asyncio.run(main())
