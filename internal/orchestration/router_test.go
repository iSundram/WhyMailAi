package orchestration

import (
	"testing"

	"github.com/iSundram/WhyMailAi/internal/policy"
	"github.com/iSundram/WhyMailAi/pkg/types"
)

func TestRunSpamAppliesPolicyDecision(t *testing.T) {
	p := policy.NewEngine()
	p.SetOverride("tenant-a", policy.TenantPolicy{SpamThreshold: 0.1})
	r := NewRouter(p)
	out, err := r.Run(RunRequest{
		TenantID: "tenant-a",
		TaskType: types.TaskSpamClassification,
		Input: map[string]any{
			"subject": "buy now free offer",
			"body":    "limited time discount",
		},
	})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if out.Result["label"] == "" {
		t.Fatal("expected label in result")
	}
	if out.ModelName == "" {
		t.Fatal("expected model name")
	}
}
