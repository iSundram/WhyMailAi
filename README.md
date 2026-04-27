# WhyMail AI

Production-oriented AI subsystem scaffold for WhyMail.

## Expanded Implementation

This repository now includes a significantly expanded architecture beyond the initial scaffold:

- **API Layer** (`cmd/ai-api`, `internal/api`)
  - Stable AI endpoints for scoring, drafting, rewriting, summarization, search, prioritization, admin insights, feedback
  - Async inference acceptance (`?async=true`) and job status polling (`GET /api/ai/jobs/{job_id}`)
  - Auth checks, tenant isolation, request size bounds, rate limiting
- **Orchestration Layer** (`internal/orchestration`)
  - Task routing with tenant-aware policy lookup
  - Preferred-model selection + confidence-based fallback
  - Policy-based decision thresholds (spam/phishing actions)
- **Model Layer** (`internal/models`)
  - Pluggable backend interface
  - Local primary backend (broad task coverage)
  - Local fallback backend for high-confidence spam/phishing fallback
- **Policy Layer** (`internal/policy`)
  - Tenant default policy + per-tenant overrides
  - Thresholds and preferred model controls
- **Safety Layer** (`internal/safety`)
  - Prompt-injection pattern detection
  - Input sanitization (control char cleanup, truncation)
  - Action override support for unsafe contexts
- **Data/Storage Layer** (`internal/storage`)
  - In-memory inference audit log
  - Feedback ingestion store
  - Async job lifecycle store
- **Telemetry Layer** (`internal/telemetry`)
  - Request/error/rate-limit counters
  - Task latency aggregation and metrics snapshot endpoint
- **Contracts & Types** (`pkg/contracts`, `pkg/types`)
  - Canonical typed request/response contracts
  - Confidence calibration and feature contribution support

## Run

```bash
go run ./cmd/ai-api
```

Default address: `:8080`

## Train models (standard / advanced)

```bash
# Standard
go run ./cmd/trainer --task all --data-dir data/ --models-dir models/

# Advanced profile (larger base models, optional extra data)
go run ./cmd/trainer --task all --profile advanced --include-extended-hf
```

## Key Environment Variables

- `WHYMAIL_AI_ADDRESS`
- `WHYMAIL_AI_READ_TIMEOUT_SECONDS`
- `WHYMAIL_AI_WRITE_TIMEOUT_SECONDS`
- `WHYMAIL_AI_MAX_REQUEST_BYTES`
- `WHYMAIL_AI_RATE_LIMIT_RPM`

## Core Endpoints

- `POST /api/ai/spam-score`
- `POST /api/ai/phishing-score`
- `POST /api/ai/draft`
- `POST /api/ai/rewrite`
- `POST /api/ai/summarize`
- `POST /api/ai/search`
- `POST /api/ai/prioritize`
- `POST /api/ai/admin/insights`
- `POST /api/ai/feedback`
- `GET /api/ai/jobs/{job_id}`
- `GET /api/ai/admin/metrics`

Required headers:

- `Authorization: Bearer <token>`
- `X-Tenant-ID: <tenant-id>` (required for task and job calls)

Request body shape:

```json
{
  "tenant_id": "tenant-a",
  "input": {},
  "options": {}
}
```
