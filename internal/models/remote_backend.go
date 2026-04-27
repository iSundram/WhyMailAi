// Package models provides the pluggable inference backend interface and
// built-in backend implementations for WhyMail AI.
//
// RemoteInferenceBackend delegates inference to the Python FastAPI server
// started by the ML training pipeline.  When the remote server is unavailable
// the orchestration layer falls back to LocalPrimaryBackend automatically.
package models

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/iSundram/WhyMailAi/pkg/types"
)

// RemoteInferenceBackend forwards every inference request to the Python
// FastAPI inference server (ml/serve/app.py).
//
// Configuration via environment variables:
//
//	WHYMAIL_REMOTE_INFERENCE_URL  base URL of the Python server
//	                              (default: http://localhost:9090)
//	WHYMAIL_REMOTE_TIMEOUT_SEC    per-request timeout in seconds (default: 30)
type RemoteInferenceBackend struct {
	baseURL    string
	httpClient *http.Client
}

// NewRemoteInferenceBackend creates a backend that calls the Python server.
//
// The base URL and timeout are taken from environment variables with
// sensible defaults; they can also be passed explicitly via the options map.
func NewRemoteInferenceBackend() *RemoteInferenceBackend {
	baseURL := strings.TrimRight(
		envOr("WHYMAIL_REMOTE_INFERENCE_URL", "http://localhost:9090"),
		"/",
	)
	timeoutSec := envIntOr("WHYMAIL_REMOTE_TIMEOUT_SEC", 30)
	return &RemoteInferenceBackend{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutSec) * time.Second,
		},
	}
}

// ID returns the backend identifier used in policy and routing decisions.
func (b *RemoteInferenceBackend) ID() string { return "remote-ml-server-v1" }

// Supports returns true for all task types handled by the Python server.
func (b *RemoteInferenceBackend) Supports(task types.TaskType) bool {
	switch task {
	case types.TaskSpamClassification,
		types.TaskPhishingClassify,
		types.TaskEmailDrafting,
		types.TaskEmailRewrite,
		types.TaskThreadSummary,
		types.TaskSemanticSearch,
		types.TaskPriorityRanking,
		types.TaskAdminAnomalyDetect,
		types.TaskFeedbackIngestion:
		return true
	default:
		return false
	}
}

// Run sends the inference request to the Python server and adapts the
// response into the canonical Output type.
func (b *RemoteInferenceBackend) Run(
	task types.TaskType,
	input map[string]any,
	options map[string]any,
) (Output, error) {
	endpoint := b.endpointFor(task)
	body := b.buildRequestBody(input, options)

	reqBody, err := json.Marshal(body)
	if err != nil {
		return Output{}, fmt.Errorf("remote backend marshal: %w", err)
	}

	resp, err := b.httpClient.Post(
		b.baseURL+endpoint,
		"application/json",
		bytes.NewReader(reqBody),
	)
	if err != nil {
		return Output{}, fmt.Errorf("remote backend request failed: %w", err)
	}
	defer resp.Body.Close()

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return Output{}, fmt.Errorf("remote backend read body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return Output{}, fmt.Errorf(
			"remote backend returned %d: %s",
			resp.StatusCode,
			strings.TrimSpace(string(rawBody)),
		)
	}

	return b.parseResponse(rawBody)
}

// Ping checks that the remote server is reachable and healthy.
// Returns nil on success.
func (b *RemoteInferenceBackend) Ping() error {
	resp, err := b.httpClient.Get(b.baseURL + "/healthz")
	if err != nil {
		return fmt.Errorf("remote backend unreachable: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("remote backend health returned %d", resp.StatusCode)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// taskEndpoints maps Go task types → Python server endpoint paths.
var taskEndpoints = map[types.TaskType]string{
	types.TaskSpamClassification: "/spam-score",
	types.TaskPhishingClassify:   "/phishing-score",
	types.TaskEmailDrafting:      "/draft",
	types.TaskEmailRewrite:       "/rewrite",
	types.TaskThreadSummary:      "/summarize",
	types.TaskSemanticSearch:     "/search",
	types.TaskPriorityRanking:    "/prioritize",
	types.TaskAdminAnomalyDetect: "/admin/insights",
	types.TaskFeedbackIngestion:  "/feedback",
}

func (b *RemoteInferenceBackend) endpointFor(task types.TaskType) string {
	if ep, ok := taskEndpoints[task]; ok {
		return ep
	}
	return "/feedback" // safe fallback
}

// buildRequestBody flattens the input map into the Python server's schema.
//
// The Python server expects a flat JSON body with named fields.
// Any key present in the input map is forwarded directly.
func (b *RemoteInferenceBackend) buildRequestBody(
	input map[string]any,
	options map[string]any,
) map[string]any {
	body := make(map[string]any, len(input)+len(options)+1)
	for k, v := range input {
		body[k] = v
	}
	// Merge options under "options" key (Python server ignores extras gracefully).
	if len(options) > 0 {
		body["options"] = options
	}
	return body
}

// remoteResponse mirrors the InferenceResponse schema from ml/serve/app.py.
type remoteResponse struct {
	Result       map[string]any     `json:"result"`
	Confidence   float64            `json:"confidence"`
	Explanations []string           `json:"explanations"`
	ModelName    string             `json:"model_name"`
	ModelVersion string             `json:"model_version"`
	LatencyMS    float64            `json:"latency_ms"`
	CacheHit     bool               `json:"cache_hit"`
}

func (b *RemoteInferenceBackend) parseResponse(raw []byte) (Output, error) {
	var r remoteResponse
	if err := json.Unmarshal(raw, &r); err != nil {
		return Output{}, fmt.Errorf("remote backend parse: %w", err)
	}

	// Extract feature_contributions from result if the Python server
	// embeds them (optional field).
	var featureContribs map[string]float64
	if fc, ok := r.Result["feature_contributions"]; ok {
		if fcMap, ok := fc.(map[string]any); ok {
			featureContribs = make(map[string]float64, len(fcMap))
			for k, v := range fcMap {
				if fv, ok := v.(float64); ok {
					featureContribs[k] = fv
				}
			}
			delete(r.Result, "feature_contributions")
		}
	}

	return Output{
		Result:               r.Result,
		Confidence:           r.Confidence,
		Explanations:         r.Explanations,
		FeatureContributions: featureContribs,
		CacheHit:             r.CacheHit,
	}, nil
}

// ---------------------------------------------------------------------------
// Environment helpers (unexported, package-scoped)
// ---------------------------------------------------------------------------

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envIntOr(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			return n
		}
	}
	return fallback
}
