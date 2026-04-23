WhyMail AI — Codebase Development Prompt

You are building WhyMail AI, a production-grade, highly accurate, privacy-first AI system for an email platform. This is not an MVP prompt. Build the codebase as if it will become a real product used by hosting providers, businesses, and self-hosted deployments at scale.

Mission

Create an intelligent AI layer for WhyMail that can:

classify spam and phishing with high accuracy,

draft, rewrite, and summarize emails,

assist users and admins with inbox intelligence,

learn from data and feedback,

scale safely, privately, and efficiently,

integrate cleanly with the existing WhyMail backend, frontend, and mail workflows.


The system must be designed as a real software product, not a demo. Prioritize correctness, observability, modularity, testability, and long-term maintainability.


---

Core Design Principles

1. Model-first, not rule-first: do not implement the AI as a purely rule-based system. Use learned models, embeddings, retrieval, fine-tuning, and probabilistic scoring. Rules may exist only as safety guards, normalization, and fallback behavior.


2. High precision and low false positives: especially for spam and phishing detection. False positives can break mail delivery trust.


3. Privacy by design: minimize data retention, support self-hosting, and avoid sending user content to third-party APIs unless explicitly enabled by configuration.


4. Modular architecture: each AI capability must be separable into services or packages.


5. Production observability: logs, metrics, tracing, model version tracking, and auditability.


6. Pluggable model backends: support local inference, remote inference, and future model swaps without rewriting the application.


7. Secure by default: rate limits, tenant isolation, input sanitization, prompt injection defenses, and safe output handling.


8. Scalable: the system should support growth from a small single-node deployment to multi-node or hosted environments.




---

Product Scope

The WhyMail AI system should include these major capabilities:

1. Spam and Phishing Intelligence

Build an AI classifier that scores incoming and outgoing emails for:

spam likelihood,

phishing likelihood,

impersonation risk,

suspicious sender patterns,

malicious links,

attachment risk,

abnormal conversational behavior,

spoofed content patterns,

bulk marketing patterns,

reply-chain abuse.


The model should return:

a numeric confidence score,

label(s),

explanation metadata,

feature contributions when available,

model version,

confidence calibration,

recommended handling action.


The classifier must support:

binary classification,

multi-label classification,

threshold tuning per tenant,

tenant-specific policy overrides,

feedback learning loop.


2. Email Drafting and Writing Assistance

Build a generative assistant that can:

draft emails from short instructions,

rewrite emails in different tones,

shorten or expand messages,

fix grammar and clarity,

translate content,

summarize long threads,

suggest subject lines,

create reply suggestions,

produce formal, friendly, technical, and sales tones,

preserve user intent without hallucinating facts.


The assistant must respect:

user-selected tone,

recipient context,

message history,

attachments and thread context where available,

organizational style settings.


3. Inbox Intelligence

Provide smart inbox features:

semantic search across emails,

thread summarization,

priority inbox scoring,

duplicate message detection,

conversation grouping,

action suggestions,

sender relationship analysis,

unread prioritization,

follow-up reminders,

meeting/date extraction,

task extraction from emails.


4. Admin Intelligence

Provide admin-facing intelligence for hosting providers and organizations:

spam trends,

mailbox abuse detection,

unusual login and sending behavior,

domain reputation monitoring,

policy violation detection,

volume anomaly alerts,

bulk sending detection,

tenant-level insight dashboards,

model performance dashboards,

false-positive review queues.


5. Feedback Learning Loop

The system should learn from user and admin feedback:

mark as spam / not spam,

helpful / not helpful draft,

accepted / edited / rejected suggestions,

phishing confirmed / false alarm,

admin review outcomes.


Use feedback for:

offline training datasets,

active learning,

calibration improvements,

personalization,

quality metrics.



---

Required Architecture

Implement the AI system as a clean, layered architecture.

Layer A — API Layer

Expose AI capabilities through stable APIs:

/api/ai/spam-score

/api/ai/phishing-score

/api/ai/draft

/api/ai/rewrite

/api/ai/summarize

/api/ai/search

/api/ai/prioritize

/api/ai/admin/insights

/api/ai/feedback


The API layer must:

authenticate users,

enforce tenant isolation,

rate limit requests,

record usage and cost estimates,

support synchronous and asynchronous jobs,

return structured JSON.


Layer B — Orchestration Layer

Create a controller that routes tasks to the appropriate model or pipeline:

spam classifier,

phishing detector,

LLM drafting engine,

embedding index,

reranker,

summarization pipeline,

policy engine.


The orchestration layer should support:

routing by task type,

confidence-based fallback,

model selection by tenant settings,

caching and batching,

background queue processing,

retries and timeouts.


Layer C — Model Layer

Support multiple model types:

sequence classifier,

embedding model,

reranker,

generative language model,

optional lightweight on-device or local models,

optional remote inference providers.


The model layer should provide a unified interface so the rest of WhyMail does not depend on a specific model vendor or framework.

Layer D — Data Layer

Store:

training records,

inference logs,

feedback labels,

model versions,

tenant policies,

calibration values,

embeddings and indexes,

audit events.


Layer E — Evaluation Layer

Build an evaluation harness that measures:

spam precision,

spam recall,

phishing precision,

phishing recall,

false positive rate,

false negative rate,

draft acceptance rate,

summarization quality,

latency,

throughput,

memory usage,

cost per inference.



---

Data and Training Strategy

The codebase must support a robust training and fine-tuning pipeline.

Data ingestion

Support importing from:

user feedback,

labeled mail corpora,

synthetic examples,

tenant-specific examples,

admin-reviewed decisions,

historical message metadata.


Dataset management

Implement:

dataset versioning,

train/validation/test splits,

deduplication,

label quality checks,

anonymization and redaction,

tenant-segmented training sets,

schema validation.


Training workflow

Provide:

offline training jobs,

fine-tuning jobs,

evaluation gates before promotion,

rollback to previous model versions,

reproducible training config,

experiment tracking.


Model promotion policy

A model may only be promoted if:

it passes evaluation thresholds,

it does not regress on key metrics,

its behavior is explainable enough for operational use,

latency and memory fit deployment constraints.



---

Privacy, Security, and Safety Requirements

The AI must be safe for email use and resistant to abuse.

Privacy

Prefer local/self-hosted inference.

Avoid storing raw message bodies unless required and configured.

Redact sensitive details in logs.

Support configurable retention windows.

Encrypt sensitive training artifacts at rest.


Security

Defend against prompt injection in email bodies and attachments.

Treat untrusted email content as hostile input.

Isolate model prompts from raw user content.

Sanitize content before generating summaries or actions.

Prevent model output from leaking secrets or internal instructions.

Use strict tenant isolation.


Abuse prevention

Protect against prompt flooding,

repeated inference abuse,

model extraction attempts,

cost blowups,

adversarial spam campaigns.



---

User-Facing Features

Build the following visible features in the WhyMail product.

Spam Intelligence UI

show a score bar,

label the reason category,

show risk highlights,

allow user feedback,

explain why a message was flagged,

show model confidence.


Draft Assistant UI

button to draft a reply,

tone selector,

length selector,

summary-to-reply workflow,

edit/accept/reject actions,

regenerate button,

language selector.


Smart Thread UI

conversation summary,

next action suggestion,

sender intent detection,

task extraction,

follow-up reminders.


Admin UI

AI model version dashboard,

spam trend analytics,

false-positive review queue,

tenant policy editor,

usage/cost tracking,

feedback and training signals.



---

API and Service Contracts

All AI services should use structured, typed contracts.

Example response requirements

Every inference response should include:

request_id

tenant_id

model_name

model_version

task_type

latency_ms

confidence

result

explanations

safety_flags

cache_hit

timestamp


Example task types

spam-classification

phishing-classification

email-drafting

email-rewrite

thread-summary

semantic-search

priority-ranking

admin-anomaly-detection



---

Repository / Codebase Structure

Use a clean, scalable repository layout.

whymail-ai/
├── cmd/
│   ├── ai-api/
│   ├── trainer/
│   ├── evaluator/
│   └── indexer/
├── internal/
│   ├── api/
│   ├── auth/
│   ├── config/
│   ├── dataset/
│   ├── inference/
│   ├── orchestration/
│   ├── policy/
│   ├── prompts/
│   ├── retrieval/
│   ├── training/
│   ├── evaluation/
│   ├── telemetry/
│   ├── safety/
│   └── storage/
├── pkg/
│   ├── client/
│   ├── types/
│   └── contracts/
├── web/
│   ├── ui/
│   └── admin/
├── configs/
├── datasets/
├── migrations/
├── docs/
├── scripts/
├── tests/
└── README.md


---

Implementation Priorities

Implement in the following order:

1. core API contracts and config system,


2. spam/phishing classifier pipeline,


3. draft/rewrite/summarize assistant,


4. feedback and data collection pipeline,


5. evaluation harness,


6. admin analytics and dashboards,


7. deployment automation,


8. model registry and versioning,


9. optional advanced retrieval and personalization,


10. optimization and scaling.




---

Performance Targets

Aim for:

low latency for spam scoring,

batchable inference where possible,

efficient memory usage,

fast cold starts for lightweight models,

graceful degradation when AI is unavailable,

caching for repeated requests,

asynchronous queues for expensive tasks.



---

Reliability Requirements

The system must:

never break mail delivery if AI is down,

fall back safely to non-AI behavior,

record failures clearly,

isolate model crashes from the main mail service,

support health checks and readiness probes,

support blue/green or versioned model rollouts.



---

Testing Requirements

Write comprehensive tests for:

API endpoints,

model routing,

dataset validation,

feedback ingestion,

evaluation logic,

safety filters,

prompt sanitization,

tenant isolation,

role-based access control,

model fallback behavior.


Include:

unit tests,

integration tests,

load tests,

adversarial tests,

regression tests,

benchmark tests.



---

Documentation Requirements

Document everything clearly:

architecture overview,

setup guide,

model registry explanation,

training workflow,

deployment instructions,

evaluation framework,

safety model,

API reference,

admin operations,

troubleshooting.


The documentation should be good enough that another engineer can extend the system without guessing.


---

What Success Looks Like

A successful WhyMail AI codebase should let a user:

receive accurate spam and phishing scoring,

draft and rewrite email intelligently,

summarize threads and extract action items,

search mail semantically,

improve the system through feedback,

keep data private and self-hosted,

scale from a single server to multi-tenant production deployments.


A successful admin should be able to:

monitor AI quality,

manage tenant policies,

inspect spam trends,

control model versions,

evaluate cost and latency,

handle false positives efficiently.



---

Final Instruction

Build WhyMail AI as a serious, intelligent, production-ready AI subsystem for email. Do not overfit to a trivial demo. Do not make it rule-based-only. Do not make it dependent on a single vendor. Build it so it can grow into a major product with real operational value, clean APIs, excellent quality, and long-term maintainability.
