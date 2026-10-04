# Security boundaries and audit

## Implemented controls

- Argon2 password hashing, bounded credential lengths, required JWT `sub`/`exp`/`iat` claims with HS256 signature/expiry verification and account-owned dataset/run/model/experiment access. Trimmed blank names are rejected; legacy blank display names have a safe fallback.
- SQLAlchemy parameterized lookups; caller IDs do not become SQL or filesystem path fragments.
- ASGI body consumption is limited before JSON/multipart parsing: 2 MiB general bodies and configured file limit + 2 MiB multipart overhead for dataset POSTs. File reads also stop at the configured limit + 1 bytes. CSV row/column limits and file-type allowlists remain enforced. Windows/POSIX path components and control characters are stripped from displayed filenames; storage uses server-generated UUIDs.
- XLSX/XLSM archives are checked before parsing: traversal/absolute/drive paths, encryption, excessive members, expanded bytes and compression ratio are rejected. `defusedxml` hardens XML parsing. VBA content is never executed.
- There is no uploaded pickle/model/code endpoint. Joblib loading is confined to server-generated artifact files under the model root, after run ownership checks. Joblib remains a **trusted-only serialization format**; filesystem compromise is outside this boundary.
- Reports escape dataset/objective/column/evidence text. React renders text without `dangerouslySetInnerHTML`. Production static pages use CSP; report downloads use a restrictive CSP. Responses include `nosniff`, same-origin referrer policy and frame denial.
- CORS uses configured origins, not unrestricted authenticated access. Secrets remain in environment/private persistent storage; Docker contexts exclude `.env`, local data, virtual environments and dependencies.
- Authentication, chat, prediction/scenario and plan requests have bounded per-process/IP request buckets; expired entries are reclaimed at capacity. Active-run quota checks/inserts share a transactional account lock. Worker concurrency and numerical threads are bounded.
- Oversized bodies are rejected even without Content-Length. Pydantic bounds objectives, lists, budgets, folds and task types; prediction/scenario fields accept only finite scalar values. Numeric values/required keys are additionally validated against saved models. Scenario changes cannot modify targets or unknown model inputs.
- HTTP failures have structured codes/messages/details and request IDs/security headers. Validation issues omit raw inputs and contexts, including plaintext credentials. Expected analytical failures have safe codes; unexpected API/job/candidate/SHAP failures never expose their exception strings or tracebacks to clients.
- Cancellation and worker progress/plan/failure/publication updates are conditional on permitted state, protecting terminal records against late writes. Dataset deletion rejects both run and experiment references and handles concurrent foreign-key conflicts.

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

Tests cover account isolation, plan/experiment/re-run ownership, unauthenticated scenarios, malicious filenames, archive traversal/malformed ZIPs/compression bombs, streamed/oversized bodies, artifact containment, rejected uploaded pickle/joblib files without deserialization, token claims/signature/expiry, rate exhaustion/recovery, concurrent quotas/cancellation, static traversal, safe error redaction, provider failures and invented LLM facts. Browser tests exercise same-origin production routes, docs, reports, model scenarios and invalid-input recovery. Python/npm advisory scans reported no known vulnerabilities at the Task 1 verification time. See `VERIFICATION.md` for actual results.

## Deployment limitations

This is a hardened engineering prototype, not a penetration-tested production certification. Before hostile internet exposure, configure TLS/reverse-proxy limits, backups/restore testing, storage/user quotas, secret rotation, monitoring and operational access controls. The in-process/IP limiter is not deployment-wide; the optional Nginx deployment needs an explicit trusted-proxy/client-IP policy. SQLite is intended for one embedded API instance per volume; PostgreSQL deployment is available but separately requires operational verification. Concurrent first-start secret initialization across unsupported multi-instance setups and artifact cleanup need additional hardening.

Dependencies and browser/server behavior change over time; pinned scientific dependencies, repeatable checks and a future automated security pipeline are necessary.
