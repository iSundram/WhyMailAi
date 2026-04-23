# WhyMail AI

Production-oriented AI subsystem scaffold for WhyMail.

## Implemented (Phase 1)

- Core API contracts (`pkg/contracts`, `pkg/types`)
- Configuration system with environment loading and validation (`internal/config`)
- AI API service with stable endpoints (`cmd/ai-api`, `internal/api`)
- Baseline orchestration router (`internal/orchestration`)
- Unit tests for config and API behavior

## Run

```bash
go run ./cmd/ai-api
```

Default address: `:8080`

## Key Environment Variables

- `WHEMAIL_AI_ADDRESS`
- `WHEMAIL_AI_READ_TIMEOUT_SECONDS`
- `WHEMAIL_AI_WRITE_TIMEOUT_SECONDS`
- `WHEMAIL_AI_MAX_REQUEST_BYTES`
- `WHEMAIL_AI_RATE_LIMIT_RPM`

## Endpoints

- `POST /api/ai/spam-score`
- `POST /api/ai/phishing-score`
- `POST /api/ai/draft`
- `POST /api/ai/rewrite`
- `POST /api/ai/summarize`
- `POST /api/ai/search`
- `POST /api/ai/prioritize`
- `POST /api/ai/admin/insights`
- `POST /api/ai/feedback`

Required headers:

- `Authorization: Bearer <token>`
- `X-Tenant-ID: <tenant-id>`

Request body shape:

```json
{
  "tenant_id": "tenant-a",
  "input": {},
  "options": {}
}
```
