package policy

import "sync"

// TenantPolicy holds per-tenant policy overrides.
type TenantPolicy struct {
	SpamThreshold     float64 `json:"spam_threshold"`
	PhishingThreshold float64 `json:"phishing_threshold"`
	MinConfidence     float64 `json:"min_confidence"`
	PreferredModel    string  `json:"preferred_model"`
}

// Engine manages policy lookup and overrides.
type Engine struct {
	mu        sync.RWMutex
	defaults  TenantPolicy
	overrides map[string]TenantPolicy
}

// NewEngine constructs a policy engine.
func NewEngine() *Engine {
	return &Engine{
		defaults: TenantPolicy{
			SpamThreshold:     0.8,
			PhishingThreshold: 0.75,
			MinConfidence:     0.86,
			PreferredModel:    "local-primary-v1",
		},
		overrides: map[string]TenantPolicy{},
	}
}

// PolicyFor returns effective policy for a tenant.
func (e *Engine) PolicyFor(tenantID string) TenantPolicy {
	e.mu.RLock()
	defer e.mu.RUnlock()
	p, ok := e.overrides[tenantID]
	if !ok {
		return e.defaults
	}
	return applyDefaults(e.defaults, p)
}

// SetOverride sets tenant-specific policy override.
func (e *Engine) SetOverride(tenantID string, p TenantPolicy) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.overrides[tenantID] = p
}

func applyDefaults(base, override TenantPolicy) TenantPolicy {
	out := base
	if override.SpamThreshold > 0 {
		out.SpamThreshold = override.SpamThreshold
	}
	if override.PhishingThreshold > 0 {
		out.PhishingThreshold = override.PhishingThreshold
	}
	if override.MinConfidence > 0 {
		out.MinConfidence = override.MinConfidence
	}
	if override.PreferredModel != "" {
		out.PreferredModel = override.PreferredModel
	}
	return out
}
