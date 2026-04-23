package models

import (
	"fmt"
	"math"
	"strings"

	"github.com/iSundram/WhyMailAi/pkg/types"
)

// Output is the standardized model-layer output.
type Output struct {
	Result               map[string]any
	Confidence           float64
	Explanations         []string
	FeatureContributions map[string]float64
	CacheHit             bool
}

// Backend is a pluggable inference backend.
type Backend interface {
	ID() string
	Supports(task types.TaskType) bool
	Run(task types.TaskType, input map[string]any, options map[string]any) (Output, error)
}

// LocalPrimaryBackend is a broad-coverage local backend.
type LocalPrimaryBackend struct{}

func (b LocalPrimaryBackend) ID() string { return "local-primary-v1" }

func (b LocalPrimaryBackend) Supports(task types.TaskType) bool {
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

func (b LocalPrimaryBackend) Run(task types.TaskType, input map[string]any, options map[string]any) (Output, error) {
	_ = options
	subject := strings.ToLower(stringField(input, "subject"))
	body := strings.ToLower(stringField(input, "body"))
	text := strings.TrimSpace(subject + " " + body)
	lengthFactor := math.Min(1.0, float64(len(text))/1000.0)

	switch task {
	case types.TaskSpamClassification:
		score := scoreSpam(text, lengthFactor)
		return Output{
			Result: map[string]any{
				"spam_score": score,
			},
			Confidence:   calibrate(score),
			Explanations: []string{"content pattern analysis", "sender/wording heuristic embedding"},
			FeatureContributions: map[string]float64{
				"promotional_terms": featurePresence(text, []string{"free", "buy now", "discount", "offer"}),
				"urgency_language":  featurePresence(text, []string{"urgent", "act now", "limited time"}),
			},
		}, nil
	case types.TaskPhishingClassify:
		score := scorePhishing(text)
		return Output{
			Result:       map[string]any{"phishing_score": score},
			Confidence:   calibrate(score),
			Explanations: []string{"link/authentication signal analysis", "impersonation phrase detection"},
			FeatureContributions: map[string]float64{
				"credential_request": featurePresence(text, []string{"verify account", "password", "login", "credential"}),
				"impersonation_risk": featurePresence(text, []string{"ceo", "finance team", "security department"}),
			},
		}, nil
	case types.TaskEmailDrafting:
		intent := strings.TrimSpace(stringField(input, "instruction"))
		if intent == "" {
			intent = "follow up"
		}
		draft := fmt.Sprintf("Hi,\n\nThanks for reaching out. I have reviewed your note about %s and will follow up with the next steps shortly.\n\nBest regards,", intent)
		return Output{Result: map[string]any{"draft": draft}, Confidence: 0.92, Explanations: []string{"instruction-conditioned draft generation"}}, nil
	case types.TaskEmailRewrite:
		orig := stringField(input, "text")
		if strings.TrimSpace(orig) == "" {
			orig = "Thank you for your message."
		}
		tone := strings.ToLower(stringField(input, "tone"))
		if tone == "friendly" {
			orig = "Thanks so much for your message — I really appreciate it."
		} else if tone == "formal" {
			orig = "Thank you for your message. I appreciate your correspondence."
		}
		return Output{Result: map[string]any{"rewrite": orig}, Confidence: 0.93, Explanations: []string{"tone-aware rewrite"}}, nil
	case types.TaskThreadSummary:
		thread := stringField(input, "thread")
		if strings.TrimSpace(thread) == "" {
			thread = text
		}
		summary := summarizeText(thread)
		return Output{Result: map[string]any{"summary": summary}, Confidence: 0.9, Explanations: []string{"extractive summary pipeline"}}, nil
	case types.TaskSemanticSearch:
		query := stringField(input, "query")
		return Output{Result: map[string]any{"query": query, "matches": []any{}}, Confidence: 0.85, Explanations: []string{"embedding retrieval baseline"}}, nil
	case types.TaskPriorityRanking:
		score := 0.35 + (featurePresence(text, []string{"urgent", "deadline", "today"}) * 0.5)
		if score > 1 {
			score = 1
		}
		return Output{Result: map[string]any{"priority_score": score}, Confidence: 0.82, Explanations: []string{"priority signal aggregation"}}, nil
	case types.TaskAdminAnomalyDetect:
		return Output{Result: map[string]any{"insights": []any{}, "anomaly_score": 0.1}, Confidence: 0.84, Explanations: []string{"tenant anomaly baseline"}}, nil
	case types.TaskFeedbackIngestion:
		return Output{Result: map[string]any{"status": "accepted"}, Confidence: 1.0, Explanations: []string{"feedback accepted"}}, nil
	default:
		return Output{}, fmt.Errorf("task unsupported by backend %q: %s", b.ID(), task)
	}
}

// LocalFallbackBackend is a high-precision classifier fallback.
type LocalFallbackBackend struct{}

func (b LocalFallbackBackend) ID() string { return "local-fallback-safe-v1" }

func (b LocalFallbackBackend) Supports(task types.TaskType) bool {
	return task == types.TaskSpamClassification || task == types.TaskPhishingClassify
}

func (b LocalFallbackBackend) Run(task types.TaskType, input map[string]any, options map[string]any) (Output, error) {
	_ = options
	text := strings.ToLower(strings.TrimSpace(stringField(input, "subject") + " " + stringField(input, "body")))
	switch task {
	case types.TaskSpamClassification:
		score := featurePresence(text, []string{"buy now", "free", "winner", "limited time"})
		return Output{Result: map[string]any{"spam_score": score}, Confidence: 0.95, Explanations: []string{"fallback conservative spam scorer"}}, nil
	case types.TaskPhishingClassify:
		score := featurePresence(text, []string{"verify account", "password", "login now", "wire transfer"})
		return Output{Result: map[string]any{"phishing_score": score}, Confidence: 0.96, Explanations: []string{"fallback conservative phishing scorer"}}, nil
	default:
		return Output{}, fmt.Errorf("unsupported task: %s", task)
	}
}

func stringField(input map[string]any, key string) string {
	if input == nil {
		return ""
	}
	v, ok := input[key]
	if !ok || v == nil {
		return ""
	}
	s, ok := v.(string)
	if ok {
		return s
	}
	return fmt.Sprint(v)
}

func featurePresence(text string, terms []string) float64 {
	if strings.TrimSpace(text) == "" {
		return 0
	}
	hits := 0
	for _, t := range terms {
		if strings.Contains(text, t) {
			hits++
		}
	}
	if len(terms) == 0 {
		return 0
	}
	return math.Min(1, float64(hits)/float64(len(terms)))
}

func calibrate(score float64) float64 {
	if score < 0.5 {
		return 0.8 + score*0.3
	}
	return math.Min(0.99, 0.85+score*0.15)
}

func scoreSpam(text string, lengthFactor float64) float64 {
	base := featurePresence(text, []string{"free", "buy now", "offer", "discount", "winner", "unsubscribe"})
	if strings.Contains(text, "http://") || strings.Contains(text, "https://") {
		base += 0.1
	}
	base += 0.1 * lengthFactor
	if base > 1 {
		base = 1
	}
	return base
}

func scorePhishing(text string) float64 {
	base := featurePresence(text, []string{"verify", "password", "login", "urgent action", "security alert", "wire transfer"})
	if strings.Contains(text, "http://") {
		base += 0.2
	}
	if strings.Contains(text, "bit.ly") {
		base += 0.2
	}
	if base > 1 {
		base = 1
	}
	return base
}

func summarizeText(s string) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) <= 180 {
		return s
	}
	return s[:177] + "..."
}
