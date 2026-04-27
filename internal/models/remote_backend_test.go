package models_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/iSundram/WhyMailAi/internal/models"
	"github.com/iSundram/WhyMailAi/pkg/types"
)

// remoteResponse mirrors the Python server InferenceResponse schema.
type remoteResponse struct {
	Result       map[string]any     `json:"result"`
	Confidence   float64            `json:"confidence"`
	Explanations []string           `json:"explanations"`
	ModelName    string             `json:"model_name"`
	ModelVersion string             `json:"model_version"`
	LatencyMS    float64            `json:"latency_ms"`
	CacheHit     bool               `json:"cache_hit"`
}

func makeTestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(handler)
}

func spamOKHandler(w http.ResponseWriter, r *http.Request) {
	resp := remoteResponse{
		Result:       map[string]any{"spam_score": 0.85, "label": "spam"},
		Confidence:   0.90,
		Explanations: []string{"distilbert classifier"},
		ModelName:    "spam_classifier",
		ModelVersion: "1.0.0",
		LatencyMS:    12.3,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func TestRemoteBackendID(t *testing.T) {
	b := models.NewRemoteInferenceBackend()
	if b.ID() != "remote-ml-server-v1" {
		t.Errorf("unexpected ID: %s", b.ID())
	}
}

func TestRemoteBackendSupports(t *testing.T) {
	b := models.NewRemoteInferenceBackend()
	tasks := []types.TaskType{
		types.TaskSpamClassification,
		types.TaskPhishingClassify,
		types.TaskEmailDrafting,
		types.TaskEmailRewrite,
		types.TaskThreadSummary,
		types.TaskSemanticSearch,
		types.TaskPriorityRanking,
		types.TaskAdminAnomalyDetect,
		types.TaskFeedbackIngestion,
	}
	for _, task := range tasks {
		if !b.Supports(task) {
			t.Errorf("expected Supports(%s) = true", task)
		}
	}
}

func TestRemoteBackendRun_Success(t *testing.T) {
	srv := makeTestServer(t, spamOKHandler)
	defer srv.Close()

	t.Setenv("WHYMAIL_REMOTE_INFERENCE_URL", srv.URL)
	b := models.NewRemoteInferenceBackend()

	out, err := b.Run(
		types.TaskSpamClassification,
		map[string]any{"subject": "Free offer", "body": "Buy now"},
		nil,
	)
	if err != nil {
		t.Fatalf("Run() unexpected error: %v", err)
	}
	if out.Confidence != 0.90 {
		t.Errorf("confidence = %f, want 0.90", out.Confidence)
	}
	score, ok := out.Result["spam_score"].(float64)
	if !ok || score < 0.8 {
		t.Errorf("spam_score = %v, want ≥ 0.8", out.Result["spam_score"])
	}
}

func TestRemoteBackendRun_ServerError(t *testing.T) {
	srv := makeTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	})
	defer srv.Close()

	t.Setenv("WHYMAIL_REMOTE_INFERENCE_URL", srv.URL)
	b := models.NewRemoteInferenceBackend()

	_, err := b.Run(types.TaskSpamClassification, map[string]any{"text": "test"}, nil)
	if err == nil {
		t.Fatal("expected error on 500 response")
	}
}

func TestRemoteBackendPing_Success(t *testing.T) {
	srv := makeTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	defer srv.Close()

	t.Setenv("WHYMAIL_REMOTE_INFERENCE_URL", srv.URL)
	b := models.NewRemoteInferenceBackend()
	if err := b.Ping(); err != nil {
		t.Errorf("Ping() unexpected error: %v", err)
	}
}

func TestRemoteBackendPing_Failure(t *testing.T) {
	// Point at a port that has nothing listening.
	t.Setenv("WHYMAIL_REMOTE_INFERENCE_URL", "http://127.0.0.1:19999")
	b := models.NewRemoteInferenceBackend()
	if err := b.Ping(); err == nil {
		t.Error("expected Ping() to fail on unreachable server")
	}
}
