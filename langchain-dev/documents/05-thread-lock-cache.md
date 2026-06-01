## GIL — Global Interpreter Lock

CPython (the standard Python) has one rule: **only one thread executes Python bytecode at a time.** That's the GIL.

It's a mutex built into the interpreter itself. Before any thread runs Python code, it must acquire the GIL. When it's done with a bytecode instruction (or when the OS preempts it), it releases the GIL and another thread can grab it.

```
Thread A: ──[acquire GIL]──[run instruction]──[release GIL]──────────────────
Thread B: ──────────────────────────────────[acquire GIL]──[run instruction]──
```

This means Python threads don't run truly in parallel on CPU-bound work. But they still **switch between instructions** — and that's where the danger lives.

---

## Why single dict operations are safe

`d[key]` compiles down to a **single bytecode instruction**: `BINARY_SUBSCR`. The GIL is held for the entire instruction. No other thread can interrupt mid-instruction.

```python
import dis
dis.dis("d[key]")
# LOAD_NAME   d
# LOAD_NAME   key
# BINARY_SUBSCR    ← single atomic op under GIL
```

Same for `d[key] = val` → single `STORE_SUBSCR` instruction. Atomic. Safe.

---

## Why read-then-write is NOT safe

The dangerous pattern is **two separate instructions with a gap between them**:

```python
if "result" not in cache:       # instruction 1: CONTAINS_OP  ← GIL can switch HERE
    cache["result"] = compute() # instruction 2: STORE_SUBSCR
```

The GIL can be released and re-acquired **between instruction 1 and instruction 2**. That gap is the race condition.

Here's exactly what happens with two threads:

```
Time  Thread A                        Thread B
───────────────────────────────────────────────────────────
T1    "result" not in cache → True    (waiting for GIL)
T2    [GIL switches]                  "result" not in cache → True
T3    (waiting for GIL)               cache["result"] = compute()  ← B sets it
T4    cache["result"] = compute()     (done)
                         ↑
                 A computes again — wasted work.
                 If compute() has side effects (DB write, API call) → real corruption.
```

Both threads saw a miss. Both computed. Both wrote. You paid for two API calls and potentially got a torn write if `compute()` is not idempotent.

---

## Why the lock fixes it

```python
import threading
lock = threading.Lock()

with lock:
    if "result" not in cache:
        cache["result"] = compute()
```

`with lock:` means: **only one thread can be inside this block at a time.** Thread B cannot even check the condition until Thread A is completely done — including the write.

```
Time  Thread A                        Thread B
───────────────────────────────────────────────────────────
T1    acquire lock ✓                  (blocked — lock taken)
T2    "result" not in cache → True    (still blocked)
T3    cache["result"] = compute()     (still blocked)
T4    release lock                    acquire lock ✓
T5    (done)                          "result" in cache → False skipped
T6                                    returns existing value — no recompute
```

Thread B enters the block, checks again (`if "result" not in cache`), sees it's already there, and skips. One computation. One write.

---

## The key mental model

| Operation | Instructions | Safe? | Why |
|---|---|---|---|
| `d[key]` | 1 (atomic) | ✓ | GIL held for full instruction |
| `d[key] = val` | 1 (atomic) | ✓ | GIL held for full instruction |
| `if key not in d: d[key] = f()` | 2+ (non-atomic) | ✕ | GIL can switch in the gap |
| `with lock: if key not in d: d[key] = f()` | 2+ but serialized | ✓ | Lock prevents gap exploitation |

The GIL protects **individual bytecode instructions**. It does not protect **sequences of instructions that belong together logically**. That's your responsibility, via locks.