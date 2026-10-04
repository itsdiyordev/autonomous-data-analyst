# Security boundaries and audit

## Implemented controls

- Argon2 password hashing, bounded credential lengths, HS256 JWT verification/expiry and account-owned dataset/run/model/experiment access.
- SQLAlchemy parameterized lookups; caller IDs do not become SQL or filesystem path fragments.
- Upload reads stop at the configured limit + 1 bytes. CSV row/column limits and file-type allowlists remain enforced. Windows/POSIX path components and control characters are stripped from displayed filenames; storage uses server-generated UUIDs.
- XLSX/XLSM archives are checked before parsing: traversal/absolute/drive paths, encryption, excessive members, expanded bytes and compression ratio are rejected. `defusedxml` hardens XML parsing. VBA content is never executed.
- There is no uploaded pickle/model/code endpoint. Joblib loading is confined to server-generated artifact files under the model root, after run ownership checks. Joblib remains a **trusted-only serialization format**; filesystem compromise is outside this boundary.
- Reports escape dataset/objective/column/evidence text. React renders text without `dangerouslySetInnerHTML`. Production static pages use CSP; report downloads use a restrictive CSP. Responses include `nosniff`, same-origin referrer policy and frame denial.
- CORS uses configured origins, not unrestricted authenticated access. Secrets remain in environment/private persistent storage; Docker contexts exclude `.env`, local data, virtual environments and dependencies.
- Authentication, chat, prediction/scenario and plan requests have bounded per-process/IP request buckets; active jobs per account remain limited. Worker concurrency and numerical threads are bounded.
- Declared oversized JSON bodies are rejected; Pydantic bounds objectives, lists, budgets, folds and task types. Prediction numeric values/required keys are validated. Scenario changes cannot modify targets or unknown model inputs.

## LLM trust boundary

```text
Untrusted dataset/objective text
  → deterministic analytical code
  → structured computed evidence + approved narrative statements
  → optional provider selects existing evidence IDs
  → strict response validation
  → render original computed statements, or fallback
```

The provider cannot supply report prose, numerical facts, model scores, code or unsupported evidence IDs. Unknown/duplicate IDs, extra response fields, invalid JSON and provider failures are rejected. The LLM has no tools or Python executor. Provider keys and untrusted questions are not printed in fallback logs.

## Verification coverage

Tests cover account isolation, plan/experiment/re-run ownership, unauthenticated scenarios, malicious filenames, archive traversal/malformed ZIPs/compression bombs, oversized uploads, artifact containment, auth rate exhaustion, static traversal, provider failures and invented LLM facts. Browser tests exercise same-origin production routes, docs, reports and model scenarios. See `VERIFICATION.md` for actual results.

## Deployment limitations

This is a hardened engineering prototype, not a penetration-tested production certification. Before hostile internet exposure, configure TLS/reverse-proxy limits, backups/restore testing, storage/user quotas, secret rotation, monitoring and operational access controls. The in-process/IP limiter is not a deployment-wide identity-aware limiter. Header-based JSON limits do not provide a complete streaming quota for undeclared chunked bodies. SQLite is intended for one embedded API instance per volume; PostgreSQL deployment is available but separately requires operational verification.

Dependencies and browser/server behavior change over time; pinned scientific dependencies, repeatable checks and a future automated security pipeline are necessary.
