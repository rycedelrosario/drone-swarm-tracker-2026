# Threat Model — Drone Swarm Tactical Console (backend)

Scope: the FastAPI backend (`app/main.py` + supporting modules) serving live/replay track data, zones, and events over HTTP and WebSocket, normally bound to `127.0.0.1` only.

## Who could attack this, and what they'd want

**1. A malicious web page open in the same browser** (drive-by cross-site request while the console is running locally).
Wants: read live/replay track data or zones via `fetch`/WebSocket; plant bogus zones.
Mitigation: CORS is an explicit origin allowlist — `allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$"`, never `allow_origins=["*"]` — so a page from any other origin can't read the response even if it can send the request. (Note: this covers `fetch`, not the WebSocket handshake itself — browsers don't apply CORS to WS connects. The real backstop here is that the server should only ever bind to `127.0.0.1`, so a page can't reach it unless it's already running on the visitor's own machine.)

**2. Another local user/process on a shared machine** (lab computer, CI box).
Wants: read `radar.db` directly, or hit the API if it's ever misconfigured to bind `0.0.0.0`.
Mitigation: course convention is `--host 127.0.0.1` explicitly (never rely on uvicorn's default). No additional file-permission hardening on `radar.db` beyond the OS default — acceptable for a single-user training tool, not for a shared/multi-tenant box.

**3. An adversarial or just-curious user of the API itself** (a cadet poking at it with curl/devtools).
Wants: crash the server, corrupt the DB, pull data outside their intended window, or inject SQL.
Mitigation: this is the bulk of what this review checked — see findings below. Every parameter is type- and range-validated; all SQL uses `?` placeholders (verified: zero string-built queries anywhere in the codebase); malformed input returns 400/422, never a stack trace.

**4. A well-meaning teammate causing accidental resource exhaustion** (a giant polygon, a huge replay range, many tabs left open).
Wants: nothing malicious, but could still degrade the app.
Mitigation: polygon vertices capped at 256, zone names capped at 128 chars, `/replay` and `/events` ranges capped at 10 minutes and 20,000 rows, the buffered sample writer caps its in-memory queue at 5,000 rows.

## Accepted risk (out of scope for this project)

- **No authentication.** Anyone who can reach the port can read all data and write/delete zones. Fine for a localhost-only single-user tool; not fine if ever exposed beyond that.
- **No request body size limit** — Starlette has no default cap, so a very large JSON body could be sent to `POST /zones` before field-level validation even runs. Low real risk at localhost scale; would need a body-size-limiting middleware if this ever left a trusted machine.
- **Single shared SQLite connection** (`read_conn`), no pooling. Fine at this traffic scale.

## Findings from this pass (fixed)

**Integer overflow crash (was a real 500, now fixed).** `DELETE /zones/{zone_id}` and `start_ms`/`end_ms` on `/replay`, `/events`, and `/ws/replay` accepted unbounded Python integers. Python ints have arbitrary precision but SQLite's `INTEGER` is a signed 64-bit column; a value like `10**30` crashed the request with an uncaught `OverflowError: Python int too large to convert to SQLite INTEGER` — a raw 500, and notably the *sum-based* range check on `/replay`/`/events` (`end_ms - start_ms <= 10 minutes`) didn't catch it, since two huge-but-close-together timestamps still pass that check while individually overflowing SQLite. Fixed by bounding all of them to `le=9223372036854775807` via FastAPI's `Query`/`Path` constraints. Verified live: all previously-crashing requests now return 400/422; normal requests still work.

**Everything else checked and confirmed clean, not just by reading the code but by firing malformed input at every live endpoint:**
- Missing/non-numeric/negative/oversized params, SQL-injection-style strings, `NaN`/`Infinity`, wrong JSON types, malformed JSON, null bytes, path-traversal-style URLs — all rejected with 400/422/404/405, zero 500s, zero tracebacks reaching the client.
- Every `.execute()`/`.executemany()` call in the codebase uses `?` placeholders with a separate params tuple; grepped the whole backend for f-strings/`.format()`/concatenation building SQL text — none found.
- CORS origin allowlist confirmed explicit (regex-scoped to localhost/127.0.0.1, any port) — `allow_methods`/`allow_headers` are wildcarded, but that's standard and low-risk once origin is locked down.
- Zone/event names render through React JSX (auto-escaped) with no `dangerouslySetInnerHTML` anywhere in the frontend — no stored-XSS path for a malicious zone name.
