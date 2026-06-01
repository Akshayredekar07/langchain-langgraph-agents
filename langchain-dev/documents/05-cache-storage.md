# **Caching — Complete Engineering Notes**
> From first principles → Python implementations → Production systems

---

## **Table of Contents**
1. [What Is a Cache?](#1-what-is-a-cache)
2. [Why Caching Matters — Concrete Impact](#2-why-caching-matters--concrete-impact)
3. [Core Terminology](#3-core-terminology)
4. [Cache Layers in a Software System](#4-cache-layers-in-a-software-system)
5. [Why dict Is the Preferred Primitive](#5-why-dict-is-the-preferred-primitive)
6. [Python Caching — Level by Level](#6-python-caching--level-by-level)
7. [Eviction Policies — The Core Algorithm Problem](#7-eviction-policies--the-core-algorithm-problem)
8. [Cache Write Strategies](#8-cache-write-strategies)
9. [Production Cache Problems & Solutions](#9-production-cache-problems--solutions)
10. [Production-Grade Cache Architecture](#10-production-grade-cache-architecture)
11. [Caching in LLM / AI Systems](#11-caching-in-llm--ai-systems)
12. [Quick Reference](#12-quick-reference)

---

## 1. What Is a Cache?

A cache is a **faster, smaller storage layer** that sits between the requester and the original (slower) data source. Its sole job: return a pre-computed or pre-fetched answer without going back to the expensive source.

The fundamental asymmetry it exploits:

| Storage Tier | Access Time | Size |
|---|---|---|
| CPU L1 cache | ~1 ns | ~32 KB |
| CPU L3 cache | ~10 ns | ~8–32 MB |
| RAM (in-process dict) | ~100 ns | GBs |
| Redis (network) | ~1 ms | GBs |
| SSD (DB query) | ~100 ms | TBs |
| HDD / cold storage | ~10 ms–10 s | PBs |

> ↪ **Core insight:** Caching does not make your code faster — it makes you do less work. The underlying computation is unchanged; you just skip it on repeat calls.

**Real-world analogy:**  
A chef memorizes the 10 most-ordered dishes. When order #7 comes in, they don't re-read the recipe — they already know it. The recipe book (database) is only opened for rare or new orders.

---

## 2. Why Caching Matters — Concrete Impact

### Performance impact (numbers that matter)

| Scenario | Without Cache | With Cache | Speedup |
|---|---|---|---|
| DB query for user profile | 50–200 ms | ~1 ms (Redis) | 50–200x |
| ML model prediction (CPU) | 500 ms | ~0.1 ms (in-memory) | 5000x |
| Fibonacci(40) recursive | ~1.3 sec | ~0 ms (memoized) | ∞ |
| LLM API call (GPT-4) | 3–10 sec + $0.01 | 0 sec + $0 | ∞ |
| DNS resolution | ~200 ms | ~0 ms (OS cache) | ∞ |

### Business impact

- **Cost reduction** — Fewer DB reads, fewer API calls, lower cloud spend.
- **Scalability** — A single Redis node serving 100k req/s absorbs load that would need 50 DB replicas.
- **Availability** — Cache serves traffic even when the origin DB is slow or temporarily down.
- **User experience** — Sub-100ms responses feel instant; 500ms+ feels sluggish.

### Example: E-commerce product page

```
Without cache:
  GET /product/123 → DB query (joins 5 tables) → 180ms → render → 200ms total
  1000 concurrent users → 1000 simultaneous DB queries → DB meltdown

With cache:
  First request:   DB query (180ms) → store in Redis → 200ms
  Requests 2–N:   Redis lookup → 1ms → 5ms total
  1000 concurrent → 1 DB query + 999 Redis lookups → DB barely touched
```

---

## 3. Core Terminology

| Term | Definition |
|---|---|
| **Cache Hit** | Requested data IS in the cache — fast return, no origin query |
| **Cache Miss** | Data NOT in cache — must fetch from origin, then store in cache |
| **Hit Rate** | `hits / (hits + misses)` — target > 80% for most systems |
| **Cache Key** | The identifier used to look up a cached value (e.g. `"user:123"`) |
| **TTL (Time-To-Live)** | How long a cached value is valid before expiring automatically |
| **Eviction** | Removing entries from cache when it's full or TTL expires |
| **Invalidation** | Explicitly marking cache entries as stale after data changes |
| **Cache Warm-up** | Pre-populating cache before traffic hits (avoids cold-start misses) |
| **Cache Stampede** | Many requests miss simultaneously on an expired key → DB overwhelmed |
| **Stale Data** | Cached value that no longer matches the source — exists when TTL > update frequency |

---

## 4. Cache Layers in a Software System

A production system has **multiple cache layers**, each with a different scope:

```
Browser / Client
    └── Browser cache (JS, CSS, images) — HTTP Cache-Control headers
CDN Edge (Cloudflare, AWS CloudFront)
    └── Static assets, API responses — geographic proximity
Reverse Proxy (NGINX, Varnish)
    └── Full HTTP response cache — reduces app server load
Application Layer (Python process)
    └── In-process dict / lru_cache — sub-millisecond, no network
Distributed Cache (Redis, Memcached)
    └── Shared across all app instances — survives process restart
Database Query Cache (MySQL query cache, Postgres pgBouncer)
    └── Repeated identical SQL queries
Storage Layer (SSD, HDD)
    └── OS page cache, disk buffer cache
```

> 💡 **Tip:** Each layer closer to the user is faster but smaller and shorter-lived. Design from the outermost layer inward — a CDN cache hit is cheaper than a Redis hit, which is cheaper than a DB hit.

---

## 5. Why `dict` Is the Preferred Primitive

### O(1) everywhere that matters

Python `dict` (CPython implementation) is a **hash table**. Hash table operations:

| Operation | Average Case | Worst Case |
|---|---|---|
| `d[key]` (get) | O(1) | O(n) — hash collision |
| `d[key] = val` (set) | O(1) | O(n) |
| `key in d` (membership) | O(1) | O(n) |
| `del d[key]` (delete) | O(1) | O(n) |

No data structure beats O(1) for cache lookups. Lists are O(n) search. Trees are O(log n). `dict` is the only structure where "is this cached?" costs nothing regardless of cache size.

### Python dict is insertion-ordered (Python 3.7+)

```python
# dict preserves insertion order — critical for LRU implementation
cache = {}
cache['a'] = 1
cache['b'] = 2
cache['c'] = 3
list(cache.keys())  # ['a', 'b', 'c'] — guaranteed order
```

This makes it possible to build an LRU cache using only `dict` + `collections.OrderedDict`.

### Thread safety in CPython

CPython's GIL makes individual dict operations atomic:
- `d[key]` → safe to read from multiple threads
- `d[key] = val` → safe as a single assignment
- **NOT safe**: read-then-write sequences (`if key not in d: d[key] = compute()`)

```python
# ── Race condition (NOT safe even with GIL) ───────────────────────────────
if "result" not in cache:          # Thread A checks
    # Thread B also checks here — both see miss
    cache["result"] = expensive()  # Both compute → wasted work or corruption
```

```python
# ── Safe pattern with lock ─────────────────────────────────────────────────
import threading
lock = threading.Lock()

with lock:
    if "result" not in cache:
        cache["result"] = expensive()
```

### Why not other structures?

| Structure | Lookup | Problem |
|---|---|---|
| `list` | O(n) | Too slow for any real cache |
| `set` | O(1) | Stores keys only, no values |
| `collections.deque` | O(n) | Ordered but slow lookup |
| `dict` ✓ | O(1) | Perfect for key→value cache |
| `collections.OrderedDict` ✓ | O(1) | dict + move_to_end() for LRU |

---

## 6. Python Caching — Level by Level

### Level 1 — Bare dict (manual memoization)

The most explicit. You control everything.

```python
# ── basic manual cache ─────────────────────────────────────────────────────
_cache: dict = {}

def get_user(user_id: int) -> dict:
    if user_id in _cache:
        return _cache[user_id]                    # cache hit
    user = db.query(f"SELECT * FROM users WHERE id={user_id}")
    _cache[user_id] = user                        # store on miss
    return user

# ── problem: no TTL, no size limit, lives forever ─────────────────────────
```

### Level 2 — `functools.lru_cache` (standard library)

Best for **pure functions** (same input → same output always).

```python
# ── imports ────────────────────────────────────────────────────────────────
from functools import lru_cache, cache

# ── lru_cache: bounded, LRU eviction ──────────────────────────────────────
@lru_cache(maxsize=512)
def get_exchange_rate(currency: str) -> float:
    return external_api.fetch_rate(currency)      # expensive

# ── functools.cache: unbounded (no LRU overhead — ~40% faster) ────────────
@cache
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)               # O(n) with cache, O(2^n) without

# ── inspect cache stats ────────────────────────────────────────────────────
get_exchange_rate("USD")
get_exchange_rate("EUR")
get_exchange_rate("USD")                          # cache hit

info = get_exchange_rate.cache_info()
# CacheInfo(hits=1, misses=2, maxsize=512, currsize=2)
print(f"Hit rate: {info.hits / (info.hits + info.misses):.0%}")

# ── clear cache manually ───────────────────────────────────────────────────
get_exchange_rate.cache_clear()
```

**Limitations of `lru_cache`:**
- Arguments must be **hashable** (no lists, dicts as args)
- No TTL — cached forever until `maxsize` triggers eviction
- No async support — breaks with `async def`
- Not safe for impure functions (functions with side effects or time-dependent results)

```python
# ── lru_cache with unhashable args — workaround ───────────────────────────
@lru_cache(maxsize=128)
def process_items(items: tuple) -> list:          # tuple is hashable
    return [item * 2 for item in items]

process_items(tuple([1, 2, 3]))                   # convert list → tuple before calling
```

### Level 3 — TTL Cache (custom)

When you need entries to expire after N seconds.

```python
# ── imports ────────────────────────────────────────────────────────────────
import time
import threading
from typing import Any, Optional

class TTLCache:
    """In-process cache with per-entry time-to-live expiration."""

    def __init__(self, ttl_seconds: int, maxsize: int = 1000):
        self.ttl = ttl_seconds
        self.maxsize = maxsize
        self._cache: dict[Any, tuple[Any, float]] = {}  # key → (value, expiry_timestamp)
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None                           # cache miss
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._cache[key]                  # expired — evict
                return None
            return value                              # cache hit

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if len(self._cache) >= self.maxsize:
                # simple eviction: remove oldest key
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = (value, time.monotonic() + self.ttl)

    def delete(self, key: Any) -> None:
        with self._lock:
            self._cache.pop(key, None)

    @property
    def size(self) -> int:
        return len(self._cache)


# ── usage ──────────────────────────────────────────────────────────────────
cache = TTLCache(ttl_seconds=300, maxsize=500)     # 5-minute TTL

def get_weather(city: str) -> dict:
    cached = cache.get(city)
    if cached is not None:
        return cached
    data = weather_api.fetch(city)                 # expensive external call
    cache.set(city, data)
    return data
```

### Level 4 — `cachetools` (library)

Drop-in replacements for common cache types.

```python
# ── install ────────────────────────────────────────────────────────────────
# uv add cachetools

from cachetools import TTLCache, LRUCache, LFUCache, cached, cachedmethod
import threading

# ── TTLCache: LRU + TTL combined ──────────────────────────────────────────
ttl_cache = TTLCache(maxsize=256, ttl=600)        # 256 entries, 10-min TTL

@cached(cache=ttl_cache)
def get_stock_price(symbol: str) -> float:
    return market_api.get_price(symbol)

# ── LRU on a class method (thread-safe) ──────────────────────────────────
class UserService:
    def __init__(self):
        self._cache = LRUCache(maxsize=1000)
        self._lock = threading.Lock()

    @cachedmethod(lambda self: self._cache, lock=lambda self: self._lock)
    def get_user(self, user_id: int) -> dict:
        return db.fetch_user(user_id)

    def invalidate(self, user_id: int) -> None:
        self._cache.pop(user_id, None)            # manual invalidation on update
```

### Level 5 — Async Cache (for async/await code)

`lru_cache` does not work with `async def`. Use `asyncio.Lock` instead.

```python
# ── imports ────────────────────────────────────────────────────────────────
import asyncio
import time
from typing import Any, Optional

class AsyncTTLCache:
    """Thread-safe async cache with TTL."""

    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._cache[key]
                return None
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[key] = (value, time.monotonic() + self.ttl)


# ── usage with async LLM calls ────────────────────────────────────────────
cache = AsyncTTLCache(ttl_seconds=3600)

async def cached_llm_call(prompt: str) -> str:
    key = hashlib.md5(prompt.encode()).hexdigest()
    cached = await cache.get(key)
    if cached:
        return cached
    response = await llm.ainvoke(prompt)           # actual LLM call
    await cache.set(key, response.content)
    return response.content
```

### Level 6 — Redis (distributed cache)

When you have multiple app instances / workers — in-process cache is per-process. Redis is shared.

```python
# ── install ────────────────────────────────────────────────────────────────
# uv add redis

import redis
import json
import hashlib
from typing import Any, Optional

# ── connection ────────────────────────────────────────────────────────────
r = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
)

# ── cache-aside pattern ───────────────────────────────────────────────────
def get_user(user_id: int) -> dict:
    key = f"user:{user_id}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)                 # cache hit
    user = db.fetch_user(user_id)                 # cache miss → DB
    r.setex(key, 3600, json.dumps(user))          # store with 1hr TTL
    return user

def update_user(user_id: int, data: dict) -> None:
    db.update_user(user_id, data)                 # write to DB first
    r.delete(f"user:{user_id}")                   # invalidate cache (don't update — race risk)

# ── hash key for complex inputs (e.g. LLM prompts) ────────────────────────
def cache_key(prefix: str, *args, **kwargs) -> str:
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"

key = cache_key("llm_response", prompt="What is AI?", model="qwen3")
# → "llm_response:a3f4c8d1e9b2f5a7"
```

---

## 7. Eviction Policies — The Core Algorithm Problem

When the cache is full and a new item arrives, something must leave. The policy you choose directly determines your hit rate.

### LRU — Least Recently Used ✓ (default choice)

**Assumption:** If you haven't used it recently, you probably won't soon.  
**Implementation:** OrderedDict — move accessed items to end; evict from front.

```python
# ── LRU from scratch (interview-style) ────────────────────────────────────
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache = OrderedDict()               # maintains insertion/access order

    def get(self, key: Any) -> Optional[Any]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)              # mark as recently used
        return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)       # evict LRU (front of OrderedDict)


# ── demo ──────────────────────────────────────────────────────────────────
lru = LRUCache(capacity=3)
lru.put("a", 1)
lru.put("b", 2)
lru.put("c", 3)
lru.get("a")           # access "a" → moves to end
lru.put("d", 4)        # cache full → evicts "b" (LRU), not "a"
print(list(lru._cache.keys()))  # ['c', 'a', 'd']
```

### LFU — Least Frequently Used

**Assumption:** Popular items should stay; rarely-used items go first.  
**Use case:** News feed cache — viral articles accessed 10k times stay; old ones go.

```python
from cachetools import LFUCache

lfu = LFUCache(maxsize=100)
lfu["hot_article"] = "content"    # accessed frequently → stays longer
lfu["old_article"] = "content"    # accessed once → first to be evicted
```

### FIFO — First In, First Out

Evicts oldest-inserted item regardless of access pattern. Simple but suboptimal — a frequently accessed item inserted early gets evicted unfairly.

### TTL — Time-To-Live Expiration

Not strictly an eviction policy but works as one. Items expire after N seconds automatically. Best for data that changes on a known schedule (prices, weather, rates).

### Eviction Policy Decision Matrix

| Access Pattern | Best Policy | Why |
|---|---|---|
| Recently used data reused | LRU | Recency predicts future |
| Viral / hot items | LFU | Frequency predicts future |
| Time-sensitive data (prices) | TTL | Freshness is the constraint |
| Unknown / uniform | LRU | Safe default, widely tested |
| Simple / resource-limited | FIFO or Random | Minimal overhead |

---

## 8. Cache Write Strategies

How data gets INTO the cache when the origin is updated.

### Cache-Aside (Lazy Loading) ✓ Most common

Application manages the cache explicitly. Cache is populated only on a miss.

```
READ:  App → Cache? Miss → DB → Store in Cache → Return
WRITE: App → DB (write) → Invalidate cache key (delete, don't update)
```

```python
# ── cache-aside read ──────────────────────────────────────────────────────
def get_product(product_id: int) -> dict:
    key = f"product:{product_id}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    product = db.get_product(product_id)
    r.setex(key, 300, json.dumps(product))
    return product

# ── cache-aside write (invalidate, not update) ────────────────────────────
def update_product(product_id: int, data: dict) -> None:
    db.update_product(product_id, data)
    r.delete(f"product:{product_id}")             # next read will re-populate
```

**Pros:** Simple, cache only holds what's actually requested, DB is always source of truth.  
**Cons:** First request after miss is always slow (cold start). Risk of stale data between write and invalidation.

### Write-Through (Proactive)

Both DB and cache are updated on every write simultaneously.

```python
def update_product(product_id: int, data: dict) -> None:
    db.update_product(product_id, data)
    r.setex(f"product:{product_id}", 300, json.dumps(data))  # update cache too
```

**Pros:** Cache always fresh after writes, no stale reads.  
**Cons:** Slower writes (must wait for both DB + cache). Cache polluted with data that may never be read.

### Write-Behind (Write-Back)

Write to cache immediately, write to DB asynchronously later.

```
WRITE: App → Cache (immediate) → queue → DB (async, delayed)
```

**Pros:** Very fast writes from app's perspective.  
**Cons:** Data loss if cache dies before DB flush. Complex failure recovery. Only for non-critical write-heavy workloads.

### Refresh-Ahead

Cache proactively refreshes entries before TTL expires, while still serving the current (valid) value.

```python
import threading

def refresh_ahead(key: str, ttl: int, fetch_fn) -> Any:
    entry = r.get(key)
    remaining_ttl = r.ttl(key)

    if remaining_ttl < ttl * 0.2:                # within 20% of expiry
        threading.Thread(target=lambda: r.setex(key, ttl, fetch_fn())).start()

    return entry                                  # serve current value without waiting
```

---

## 9. Production Cache Problems & Solutions

### Problem 1: Cache Stampede (Thundering Herd)

A popular key expires. 10,000 concurrent requests all miss simultaneously → 10,000 DB queries → DB dies.

```python
# ── solution: mutex lock (only one thread rebuilds) ───────────────────────
import redis.lock

def get_with_lock(key: str, ttl: int, fetch_fn) -> Any:
    cached = r.get(key)
    if cached:
        return json.loads(cached)

    lock_key = f"lock:{key}"
    with r.lock(lock_key, timeout=10, blocking_timeout=5):
        # Re-check inside lock (another process may have populated it)
        cached = r.get(key)
        if cached:
            return json.loads(cached)
        value = fetch_fn()
        r.setex(key, ttl, json.dumps(value))
        return value
```

```python
# ── solution: probabilistic early expiration ──────────────────────────────
import math, random

def should_refresh(key: str, ttl: int, beta: float = 1.0) -> bool:
    """XFetch algorithm — recompute before expiry with increasing probability."""
    remaining = r.ttl(key)
    return -math.log(random.random()) * beta >= remaining
```

### Problem 2: Cache Invalidation

"There are only two hard things in CS: cache invalidation and naming things."  
The problem: after a DB write, when exactly does the cache reflect the new value?

```python
# ── pattern: delete-on-write (safest) ─────────────────────────────────────
def update_user(user_id: int, data: dict) -> None:
    db.update_user(user_id, data)
    r.delete(f"user:{user_id}")               # delete, never update — avoids race conditions

# ── pattern: versioned keys ────────────────────────────────────────────────
def cache_key_versioned(user_id: int) -> str:
    version = r.get(f"user:{user_id}:version") or "0"
    return f"user:{user_id}:v{version}"

def invalidate_versioned(user_id: int) -> None:
    r.incr(f"user:{user_id}:version")        # bump version → old key becomes orphan
```

### Problem 3: Stale Data

Cache returns old values after source data changes.

**Solutions:**
- Set appropriate TTLs based on how fast data changes (stock prices: 5s, user profiles: 1h)
- Use event-driven invalidation (DB triggers → publish event → cache deletes key)
- Short TTL as a safety net even with invalidation logic

### Problem 4: Memory Overflow

Cache grows unbounded → OOM crash.

```python
# ── always set maxsize ────────────────────────────────────────────────────
from functools import lru_cache

@lru_cache(maxsize=1024)             # NEVER use maxsize=None in long-running processes
def expensive_compute(x: int) -> int:
    return x ** 2

# ── Redis: set maxmemory in redis.conf ────────────────────────────────────
# maxmemory 512mb
# maxmemory-policy allkeys-lru       ← evict LRU when full (don't crash)
```

### Problem 5: Cold Start / Cache Warm-up

After deploy, cache is empty → all requests hit DB → spike.

```python
# ── warm-up script (run before traffic switch) ────────────────────────────
def warm_cache():
    top_products = db.get_top_products(limit=500)
    for product in top_products:
        r.setex(f"product:{product['id']}", 3600, json.dumps(product))
    print(f"Warmed {len(top_products)} products")

if __name__ == "__main__":
    warm_cache()
```

---

## 10. Production-Grade Cache Architecture

### Two-Level Cache (L1 in-process + L2 Redis)

Ultra-fast for hot data; Redis as fallback for less-hot data.

```python
# ── imports ────────────────────────────────────────────────────────────────
import json, redis
from cachetools import TTLCache
import threading

class TwoLevelCache:
    """
    L1: in-process TTLCache  — sub-millisecond, per-process
    L2: Redis               — shared across all workers, network hop
    """

    def __init__(self, l1_maxsize: int = 256, l1_ttl: int = 60, l2_ttl: int = 3600):
        self.l1 = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self.l2 = redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.l2_ttl = l2_ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        # ── L1 check ──────────────────────────────────────────────────────
        val = self.l1.get(key)
        if val is not None:
            return val                            # L1 hit — fastest path

        # ── L2 check ──────────────────────────────────────────────────────
        raw = self.l2.get(key)
        if raw:
            val = json.loads(raw)
            with self._lock:
                self.l1[key] = val                # backfill L1
            return val                            # L2 hit

        return None                               # full miss

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self.l1[key] = value
        self.l2.setex(key, self.l2_ttl, json.dumps(value))

    def delete(self, key: str) -> None:
        with self._lock:
            self.l1.pop(key, None)
        self.l2.delete(key)


# ── production usage ──────────────────────────────────────────────────────
cache = TwoLevelCache(l1_maxsize=200, l1_ttl=30, l2_ttl=3600)

def get_user(user_id: int) -> dict:
    key = f"user:{user_id}"
    cached = cache.get(key)
    if cached:
        return cached
    user = db.fetch_user(user_id)
    cache.set(key, user)
    return user
```

### Monitoring & Observability

```python
# ── cache metrics tracker ─────────────────────────────────────────────────
from dataclasses import dataclass, field
import threading

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_hit(self):
        with self._lock:
            self.hits += 1

    def record_miss(self):
        with self._lock:
            self.misses += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def report(self):
        print(f"Hit rate: {self.hit_rate:.1%} | Hits: {self.hits} | Misses: {self.misses}")


metrics = CacheMetrics()

def get_product(product_id: int) -> dict:
    key = f"product:{product_id}"
    cached = r.get(key)
    if cached:
        metrics.record_hit()
        return json.loads(cached)
    metrics.record_miss()
    product = db.get_product(product_id)
    r.setex(key, 300, json.dumps(product))
    return product

# ── Redis built-in stats ──────────────────────────────────────────────────
# redis-cli INFO stats | grep keyspace
# keyspace_hits:45000
# keyspace_misses:3000
# → hit rate: 93.75%
```

### FastAPI + Redis caching (production pattern)

```python
# ── imports ────────────────────────────────────────────────────────────────
import json, hashlib, redis
from functools import wraps
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def redis_cache(ttl: int = 300, prefix: str = "api"):
    """Decorator to cache FastAPI route responses in Redis."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key_data = json.dumps({"fn": func.__name__, "args": str(args), "kwargs": kwargs}, sort_keys=True)
            cache_key = f"{prefix}:{hashlib.sha256(key_data.encode()).hexdigest()[:12]}"

            cached = r.get(cache_key)
            if cached:
                return json.loads(cached)

            result = await func(*args, **kwargs)
            r.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator


@app.get("/products/{product_id}")
@redis_cache(ttl=300, prefix="product")
async def get_product(product_id: int):
    return await db.fetch_product(product_id)    # cached for 5 min
```

---

## 11. Caching in LLM / AI Systems

LLM calls are the most expensive cache targets: slow (seconds), costly ($), and often deterministic for the same prompt.

### Prompt-level caching

```python
# ── imports ────────────────────────────────────────────────────────────────
import hashlib, json, redis, os
from langchain_nebius import ChatNebius
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
model = ChatNebius(model="Qwen/Qwen3-30B-A3B-fast", temperature=0.0)  # temp=0 → deterministic

def prompt_cache_key(system: str, question: str, model_name: str) -> str:
    payload = json.dumps({"system": system, "question": question, "model": model_name})
    return "llm:" + hashlib.sha256(payload.encode()).hexdigest()[:20]

def cached_llm_invoke(system: str, question: str, ttl: int = 86400) -> str:
    """Cache LLM responses for 24h — safe for deterministic prompts."""
    key = prompt_cache_key(system, question, model.model)

    cached = r.get(key)
    if cached:
        print("[CACHE HIT]")
        return cached

    print("[CACHE MISS] — calling LLM")
    resp = model.invoke([SystemMessage(content=system), HumanMessage(content=question)])
    r.setex(key, ttl, resp.content)
    return resp.content


# ── legal Q&A bot — system prompt is static → same key every time ─────────
legal_policy = "You are a legal assistant. Policy: 1) All IP belongs to company..." * 50

answer = cached_llm_invoke(legal_policy, "Who owns my weekend invention?")
# First call: CACHE MISS → 3-5 seconds
answer = cached_llm_invoke(legal_policy, "Who owns my weekend invention?")
# Second call: CACHE HIT → ~1ms
```

### Semantic caching (cache by meaning, not exact string)

```python
# ── install: uv add sentence-transformers numpy ────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

class SemanticCache:
    """Cache LLM responses by semantic similarity — handles paraphrased queries."""

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self.entries: list[tuple[np.ndarray, str, str]] = []  # (embedding, query, response)

    def get(self, query: str) -> str | None:
        query_emb = encoder.encode(query)
        for emb, _, response in self.entries:
            similarity = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            if similarity >= self.threshold:
                return response                   # semantically similar → cache hit
        return None

    def set(self, query: str, response: str) -> None:
        emb = encoder.encode(query)
        self.entries.append((emb, query, response))


sem_cache = SemanticCache(threshold=0.92)

# "What is machine learning?" and "Explain ML to me" → same cached response
```

### KV cache (provider-level prompt caching — context window reuse)

This is what OpenAI/Anthropic do automatically. You don't control it directly; you exploit it by:
1. Keeping the **system prompt static** across requests
2. Placing the **long static context at the start** of the message
3. Sending similar conversations to the **same provider endpoint**

---

## 12. Quick Reference

### Python cache tools

| Tool | Type | TTL | Maxsize | Async | Thread-safe | When to use |
|---|---|---|---|---|---|---|
| `functools.lru_cache` | In-process LRU | ✕ | ✓ | ✕ | Partial | Pure functions, simple memoization |
| `functools.cache` | In-process unbounded | ✕ | ✕ | ✕ | Partial | Recursive algos, small input space |
| Manual `dict` | In-process | Manual | Manual | Manual | ✕ | Full control needed |
| `cachetools.TTLCache` | In-process LRU+TTL | ✓ | ✓ | ✕ | With lock | Time-sensitive in-process data |
| `redis.Redis` | Distributed | ✓ | Via policy | ✓ | ✓ | Multi-process, multi-worker, persistent |
| Two-level (dict + Redis) | Hybrid | ✓ | ✓ | ✓ | ✓ | Production, high-throughput |

### Eviction policy cheatsheet

| Policy | Evicts | When to use |
|---|---|---|
| LRU | Least recently accessed | Default — most workloads |
| LFU | Least frequently accessed | Popularity-driven (news, trending) |
| FIFO | Oldest inserted | Simple, no tracking overhead |
| TTL | Expired entries | Time-sensitive data |
| Random | Random entry | Uniform access, minimal overhead |

### Write strategy cheatsheet

| Strategy | Cache updated | Use when |
|---|---|---|
| Cache-Aside | On read miss | Default — most systems |
| Write-Through | On every write | Consistency > write speed |
| Write-Behind | Async after write | Write-heavy, tolerate data loss |
| Refresh-Ahead | Before TTL expires | Latency-sensitive, known access patterns |

### Common gotchas

| Gotcha | Fix |
|---|---|
| `lru_cache` with list/dict arg → `TypeError` | Convert to `tuple` / `frozenset` |
| `lru_cache` on async def → broken | Use `cachetools` + `asyncio.Lock` |
| Unbounded dict cache → OOM | Always set `maxsize` |
| Cache hit on stale data | Delete key on write, don't update |
| Cache stampede on expiry | Mutex lock or probabilistic refresh |
| Multiple workers, separate caches | Move to Redis |
| Caching impure functions (random, time) | Never cache — cache only deterministic functions |

### Redis key naming convention

```
{entity}:{id}                   → user:123
{entity}:{id}:{field}           → user:123:profile
{namespace}:{operation}:{hash}  → llm:response:a3f4c8d1
{entity}:lock:{id}              → product:lock:456  (distributed lock)
```

### Redis maxmemory-policy options

| Policy | Behavior |
|---|---|
| `allkeys-lru` ✓ | Evict any key using LRU — safest default |
| `volatile-lru` | Evict only keys with TTL set using LRU |
| `allkeys-lfu` | Evict any key using LFU |
| `allkeys-random` | Evict random key |
| `noeviction` | Return error when full — use only if cache loss is catastrophic |

---

> 🔗 **See also:**
> - [Redis caching patterns](https://docs.aws.amazon.com/whitepapers/latest/database-caching-strategies-using-redis/caching-patterns.html)
> - [cachetools docs](https://cachetools.readthedocs.io/)
> - [Python functools docs](https://docs.python.org/3/library/functools.html)