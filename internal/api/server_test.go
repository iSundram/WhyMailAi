package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/iSundram/WhyMailAi/internal/config"
)

func newTestServer() *Server {
	cfg := config.Config{
		ServerAddress:   ":0",
		ReadTimeout:     1,
		WriteTimeout:    1,
		MaxRequestBytes: 1 << 20,
		RateLimitRPM:    100,
	}
	return NewServer(cfg)
}

func TestUnauthorized(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/ai/spam-score", bytes.NewBufferString(`{"tenant_id":"t1","input":{}}`))
	req.Header.Set("X-Tenant-ID", "t1")
	w := httptest.NewRecorder()

	s.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", w.Code)
	}
}

func TestTenantMismatch(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/ai/spam-score", bytes.NewBufferString(`{"tenant_id":"t2","input":{}}`))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("X-Tenant-ID", "t1")
	w := httptest.NewRecorder()

	s.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d", w.Code)
	}
}

func TestSpamScoreSuccess(t *testing.T) {
	s := newTestServer()
	payload := map[string]any{"tenant_id": "tenant-a", "input": map[string]any{"subject": "hello"}}
	body, _ := json.Marshal(payload)

	req := httptest.NewRequest(http.MethodPost, "/api/ai/spam-score", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("X-Tenant-ID", "tenant-a")
	w := httptest.NewRecorder()

	s.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var got map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	for _, field := range []string{"request_id", "tenant_id", "model_name", "model_version", "task_type", "latency_ms", "confidence", "result", "explanations", "safety_flags", "cache_hit", "timestamp"} {
		if _, ok := got[field]; !ok {
			t.Fatalf("missing field %s", field)
		}
	}
}
