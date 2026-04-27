package policy

import "testing"

func TestPolicyDefaultsAndOverride(t *testing.T) {
	e := NewEngine()
	p := e.PolicyFor("tenant-a")
	if p.PreferredModel == "" || p.SpamThreshold <= 0 {
		t.Fatal("expected defaults")
	}
	e.SetOverride("tenant-a", TenantPolicy{SpamThreshold: 0.6, PreferredModel: "local-fallback-safe-v1"})
	p2 := e.PolicyFor("tenant-a")
	if p2.SpamThreshold != 0.6 {
		t.Fatalf("expected override spam threshold, got %f", p2.SpamThreshold)
	}
	if p2.PreferredModel != "local-fallback-safe-v1" {
		t.Fatalf("expected override model, got %s", p2.PreferredModel)
	}
}
