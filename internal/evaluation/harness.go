// Package evaluation provides types and helpers for evaluating model quality.
package evaluation

import (
	"time"
)

// MetricSet captures evaluation metrics for a single model.
type MetricSet struct {
	ModelName    string    `json:"model_name"`
	ModelVersion string    `json:"model_version"`
	Task         string    `json:"task"`
	Precision    float64   `json:"precision,omitempty"`
	Recall       float64   `json:"recall,omitempty"`
	F1           float64   `json:"f1,omitempty"`
	FalsePositiveRate float64 `json:"false_positive_rate,omitempty"`
	FalseNegativeRate float64 `json:"false_negative_rate,omitempty"`
	ROCAUC       float64   `json:"roc_auc,omitempty"`
	ROUGE1       float64   `json:"rouge1,omitempty"`
	RougeL       float64   `json:"rougeL,omitempty"`
	MeanLatencyMS float64  `json:"mean_latency_ms,omitempty"`
	P95LatencyMS  float64  `json:"p95_latency_ms,omitempty"`
	ThroughputRPS float64  `json:"throughput_rps,omitempty"`
	SampleCount  int       `json:"sample_count"`
	EvaluatedAt  time.Time `json:"evaluated_at"`
}

// EvaluationReport is the top-level result of an evaluation run.
type EvaluationReport struct {
	RunID       string      `json:"run_id"`
	CreatedAt   time.Time   `json:"created_at"`
	AllPromoted bool        `json:"all_promoted"`
	Models      []ModelEval `json:"models"`
}

// ModelEval combines MetricSet with promotion outcome.
type ModelEval struct {
	MetricSet
	Promoted      bool   `json:"promoted"`
	FailureReason string `json:"failure_reason,omitempty"`
}

// PromotionGate holds minimum thresholds for model promotion.
type PromotionGate struct {
	MinPrecision float64
	MinRecall    float64
	MinF1        float64
}

// DefaultSpamGate returns the default promotion gate for the spam model.
func DefaultSpamGate() PromotionGate {
	return PromotionGate{MinPrecision: 0.90, MinRecall: 0.85, MinF1: 0.88}
}

// DefaultPhishingGate returns the default promotion gate for the phishing model.
func DefaultPhishingGate() PromotionGate {
	return PromotionGate{MinPrecision: 0.92, MinRecall: 0.88, MinF1: 0.90}
}

// Passes returns true if the MetricSet clears the gate.
func (g PromotionGate) Passes(m MetricSet) bool {
	return m.Precision >= g.MinPrecision &&
		m.Recall >= g.MinRecall &&
		m.F1 >= g.MinF1
}
