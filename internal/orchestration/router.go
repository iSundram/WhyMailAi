package orchestration

import (
	"fmt"

	"github.com/iSundram/WhyMailAi/internal/models"
	"github.com/iSundram/WhyMailAi/internal/policy"
	"github.com/iSundram/WhyMailAi/pkg/types"
)

// RunRequest encapsulates orchestration inputs.
type RunRequest struct {
	TenantID string
	TaskType types.TaskType
	Input    map[string]any
	Options  map[string]any
}

// RouteResult is the output from the orchestration layer.
type RouteResult struct {
	Result               map[string]any
	Confidence           float64
	Explanations         []string
	FeatureContributions map[string]float64
	SafetyFlags          []string
	CacheHit             bool
	ModelName            string
	ModelVersion         string
	ConfidenceCalib      string
}

// Router routes requests by task type to model pipelines.
type Router struct {
	policy    *policy.Engine
	models    map[string]models.Backend
	fallback  models.Backend
	preferred string // ID of preferred backend (remote when configured)
}

// NewRouter creates a routing instance.
// When remoteBackend is non-nil it is registered as the preferred backend
// for all task types it supports, with the local backends as fallback.
func NewRouter(policyEngine *policy.Engine, remoteBackend ...models.Backend) *Router {
	primary := models.LocalPrimaryBackend{}
	fallback := models.LocalFallbackBackend{}
	r := &Router{
		policy: policyEngine,
		models: map[string]models.Backend{
			primary.ID():  primary,
			fallback.ID(): fallback,
		},
		fallback: fallback,
	}
	// Register remote backend if provided.
	if len(remoteBackend) > 0 && remoteBackend[0] != nil {
		rb := remoteBackend[0]
		r.models[rb.ID()] = rb
		r.preferred = rb.ID()
	}
	return r
}

// Run executes routing, model selection, confidence fallback, and policy post-processing.
func (r *Router) Run(req RunRequest) (RouteResult, error) {
	tenantPolicy := r.policy.PolicyFor(req.TenantID)
	backend := r.selectBackend(tenantPolicy.PreferredModel, req.TaskType)
	if backend == nil {
		return RouteResult{}, fmt.Errorf("no backend supports task %s", req.TaskType)
	}

	out, err := backend.Run(req.TaskType, req.Input, req.Options)
	if err != nil {
		return RouteResult{}, err
	}
	usedBackend := backend

	if out.Confidence < tenantPolicy.MinConfidence && r.fallback != nil && r.fallback.Supports(req.TaskType) && backend.ID() != r.fallback.ID() {
		fallbackOut, fallbackErr := r.fallback.Run(req.TaskType, req.Input, req.Options)
		if fallbackErr == nil && fallbackOut.Confidence >= out.Confidence {
			out = fallbackOut
			usedBackend = r.fallback
		}
	}

	result := out.Result
	if result == nil {
		result = map[string]any{}
	}
	applyPolicyDecision(req.TaskType, tenantPolicy, result)

	return RouteResult{
		Result:               result,
		Confidence:           out.Confidence,
		Explanations:         out.Explanations,
		FeatureContributions: out.FeatureContributions,
		SafetyFlags:          nil,
		CacheHit:             out.CacheHit,
		ModelName:            usedBackend.ID(),
		ModelVersion:         "1.0.0",
		ConfidenceCalib:      "isotonic-baseline-v1",
	}, nil
}

func (r *Router) selectBackend(preferred string, task types.TaskType) models.Backend {
	// Router-level preferred (e.g. remote ML server) takes priority over
	// per-tenant preference unless the tenant explicitly overrides it.
	effective := preferred
	if effective == "" {
		effective = r.preferred
	}
	if effective != "" {
		if m, ok := r.models[effective]; ok && m.Supports(task) {
			return m
		}
	}
	for _, m := range r.models {
		if m.Supports(task) {
			return m
		}
	}
	return nil
}

func applyPolicyDecision(task types.TaskType, p policy.TenantPolicy, result map[string]any) {
	switch task {
	case types.TaskSpamClassification:
		score := asFloat(result["spam_score"])
		if score >= p.SpamThreshold {
			result["label"] = "spam"
			result["recommended_action"] = "quarantine"
		} else {
			result["label"] = "ham"
			result["recommended_action"] = "allow"
		}
	case types.TaskPhishingClassify:
		score := asFloat(result["phishing_score"])
		if score >= p.PhishingThreshold {
			result["label"] = "phishing"
			result["recommended_action"] = "block"
		} else {
			result["label"] = "legitimate"
			result["recommended_action"] = "allow"
		}
	}
}

func asFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	default:
		return 0
	}
}
