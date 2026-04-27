package contracts

import (
	"time"

	"github.com/iSundram/WhyMailAi/pkg/types"
)

// AIRequest is a generic request contract for AI API operations.
type AIRequest struct {
	TenantID string         `json:"tenant_id"`
	Input    map[string]any `json:"input"`
	Options  map[string]any `json:"options,omitempty"`
}

// UsageCost contains approximate usage metadata for tracking.
type UsageCost struct {
	EstimatedInputTokens  int     `json:"estimated_input_tokens"`
	EstimatedOutputTokens int     `json:"estimated_output_tokens"`
	EstimatedCostUSD      float64 `json:"estimated_cost_usd"`
}

// AIResponse defines the canonical inference response contract.
type AIResponse struct {
	RequestID             string             `json:"request_id"`
	TenantID              string             `json:"tenant_id"`
	ModelName             string             `json:"model_name"`
	ModelVersion          string             `json:"model_version"`
	TaskType              types.TaskType     `json:"task_type"`
	LatencyMS             int64              `json:"latency_ms"`
	Confidence            float64            `json:"confidence"`
	ConfidenceCalibration string             `json:"confidence_calibration,omitempty"`
	Result                map[string]any     `json:"result"`
	Explanations          []string           `json:"explanations"`
	FeatureContributions  map[string]float64 `json:"feature_contributions"`
	SafetyFlags           []string           `json:"safety_flags"`
	CacheHit              bool               `json:"cache_hit"`
	Timestamp             time.Time          `json:"timestamp"`
	Usage                 UsageCost          `json:"usage"`
}

// AsyncAcceptedResponse represents an accepted asynchronous inference job.
type AsyncAcceptedResponse struct {
	JobID      string    `json:"job_id"`
	RequestID  string    `json:"request_id"`
	TenantID   string    `json:"tenant_id"`
	TaskType   string    `json:"task_type"`
	AcceptedAt time.Time `json:"accepted_at"`
}

// JobStatusResponse returns state for asynchronous inference.
type JobStatusResponse struct {
	JobID       string      `json:"job_id"`
	RequestID   string      `json:"request_id"`
	TenantID    string      `json:"tenant_id"`
	TaskType    string      `json:"task_type"`
	Status      string      `json:"status"`
	Error       string      `json:"error,omitempty"`
	CreatedAt   time.Time   `json:"created_at"`
	UpdatedAt   time.Time   `json:"updated_at"`
	CompletedAt *time.Time  `json:"completed_at,omitempty"`
	Response    *AIResponse `json:"response,omitempty"`
}
