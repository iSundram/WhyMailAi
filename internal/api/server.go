package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/iSundram/WhyMailAi/internal/config"
	"github.com/iSundram/WhyMailAi/internal/orchestration"
	"github.com/iSundram/WhyMailAi/internal/policy"
	"github.com/iSundram/WhyMailAi/internal/safety"
	"github.com/iSundram/WhyMailAi/internal/storage"
	"github.com/iSundram/WhyMailAi/internal/telemetry"
	"github.com/iSundram/WhyMailAi/pkg/contracts"
	"github.com/iSundram/WhyMailAi/pkg/types"
)

type rateLimiter struct {
	mu      sync.Mutex
	maxRPM  int
	window  time.Time
	counter map[string]int
}

func newRateLimiter(maxRPM int) *rateLimiter {
	return &rateLimiter{maxRPM: maxRPM, window: time.Now(), counter: map[string]int{}}
}

func (r *rateLimiter) allow(tenantID string, now time.Time) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if now.Sub(r.window) >= time.Minute {
		r.window = now
		r.counter = map[string]int{}
	}
	if r.counter[tenantID] >= r.maxRPM {
		return false
	}
	r.counter[tenantID]++
	return true
}

// Server hosts the AI API endpoints.
type Server struct {
	cfg       config.Config
	router    *orchestration.Router
	limiter   *rateLimiter
	store     *storage.MemoryStore
	metrics   *telemetry.Metrics
	safety    *safety.Analyzer
	now       func() time.Time
	requestID func() string
}

// NewServer constructs an API server.
func NewServer(cfg config.Config) *Server {
	policyEngine := policy.NewEngine()
	return &Server{
		cfg:       cfg,
		router:    orchestration.NewRouter(policyEngine),
		limiter:   newRateLimiter(cfg.RateLimitRPM),
		store:     storage.NewMemoryStore(),
		metrics:   telemetry.NewMetrics(),
		safety:    safety.NewAnalyzer(),
		now:       time.Now,
		requestID: newRequestID,
	}
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/api/ai/admin/metrics", s.handleMetrics)
	mux.HandleFunc("/api/ai/jobs/", s.handleJobStatus)
	mux.HandleFunc("/api/ai/spam-score", s.handleTask(types.TaskSpamClassification))
	mux.HandleFunc("/api/ai/phishing-score", s.handleTask(types.TaskPhishingClassify))
	mux.HandleFunc("/api/ai/draft", s.handleTask(types.TaskEmailDrafting))
	mux.HandleFunc("/api/ai/rewrite", s.handleTask(types.TaskEmailRewrite))
	mux.HandleFunc("/api/ai/summarize", s.handleTask(types.TaskThreadSummary))
	mux.HandleFunc("/api/ai/search", s.handleTask(types.TaskSemanticSearch))
	mux.HandleFunc("/api/ai/prioritize", s.handleTask(types.TaskPriorityRanking))
	mux.HandleFunc("/api/ai/admin/insights", s.handleTask(types.TaskAdminAnomalyDetect))
	mux.HandleFunc("/api/ai/feedback", s.handleTask(types.TaskFeedbackIngestion))
	return mux
}

func (s *Server) handleTask(taskType types.TaskType) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := s.now()
		s.metrics.Inc("requests_total")
		if r.Method != http.MethodPost {
			s.metrics.Inc("errors_method_not_allowed")
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		if !isAuthorized(r) {
			s.metrics.Inc("errors_unauthorized")
			writeError(w, http.StatusUnauthorized, "missing or invalid authorization")
			return
		}

		tenantHeader := strings.TrimSpace(r.Header.Get("X-Tenant-ID"))
		if tenantHeader == "" {
			s.metrics.Inc("errors_missing_tenant")
			writeError(w, http.StatusBadRequest, "missing X-Tenant-ID header")
			return
		}
		if !s.limiter.allow(tenantHeader, s.now()) {
			s.metrics.Inc("rate_limited")
			writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxRequestBytes)
		defer r.Body.Close()

		var req contracts.AIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.metrics.Inc("errors_bad_json")
			writeError(w, http.StatusBadRequest, "invalid json body")
			return
		}
		if strings.TrimSpace(req.TenantID) == "" {
			s.metrics.Inc("errors_missing_payload_tenant")
			writeError(w, http.StatusBadRequest, "tenant_id is required")
			return
		}
		if req.TenantID != tenantHeader {
			s.metrics.Inc("errors_tenant_mismatch")
			writeError(w, http.StatusForbidden, "tenant header and payload mismatch")
			return
		}

		safetyFlags, sanitizedInput := s.safety.AnalyzeAndSanitize(req.Input)
		req.Input = sanitizedInput

		requestID := s.requestID()
		if strings.EqualFold(r.URL.Query().Get("async"), "true") {
			jobID := s.requestID()
			job := s.store.NewJob(jobID, requestID, req.TenantID, string(taskType), req.Input, req.Options)
			go s.runAsyncJob(job)
			resp := contracts.AsyncAcceptedResponse{
				JobID:      job.JobID,
				RequestID:  job.RequestID,
				TenantID:   req.TenantID,
				TaskType:   string(taskType),
				AcceptedAt: s.now().UTC(),
			}
			writeJSON(w, http.StatusAccepted, resp)
			return
		}

		resp, err := s.executeTask(taskType, requestID, req, start, safetyFlags)
		if err != nil {
			s.metrics.Inc("errors_inference")
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, resp)
	}
}

func (s *Server) executeTask(taskType types.TaskType, requestID string, req contracts.AIRequest, start time.Time, safetyFlags []string) (contracts.AIResponse, error) {
	out, err := s.router.Run(orchestration.RunRequest{TenantID: req.TenantID, TaskType: taskType, Input: req.Input, Options: req.Options})
	if err != nil {
		return contracts.AIResponse{}, fmt.Errorf("orchestration failed: %w", err)
	}

	if taskType == types.TaskFeedbackIngestion {
		s.store.SaveFeedback(storage.FeedbackRecord{
			TenantID:   req.TenantID,
			MessageID:  asString(req.Input["message_id"]),
			Feedback:   asString(req.Input["feedback"]),
			Attributes: req.Input,
			CreatedAt:  s.now().UTC(),
		})
	}

	latency := s.now().Sub(start).Milliseconds()
	if latency < 0 {
		latency = 0
	}
	s.metrics.ObserveTaskLatency(string(taskType), latency)

	combinedSafety := uniqueStrings(append(safetyFlags, out.SafetyFlags...))
	if action, ok := out.Result["recommended_action"].(string); ok && safety.IsActionBlocked(action, combinedSafety) {
		out.Result["recommended_action"] = "review"
		combinedSafety = uniqueStrings(append(combinedSafety, "action-overridden-by-safety"))
	}

	resp := contracts.AIResponse{
		RequestID:             requestID,
		TenantID:              req.TenantID,
		ModelName:             out.ModelName,
		ModelVersion:          out.ModelVersion,
		TaskType:              taskType,
		LatencyMS:             latency,
		Confidence:            out.Confidence,
		ConfidenceCalibration: out.ConfidenceCalib,
		Result:                out.Result,
		Explanations:          out.Explanations,
		FeatureContributions:  out.FeatureContributions,
		SafetyFlags:           combinedSafety,
		CacheHit:              out.CacheHit,
		Timestamp:             s.now().UTC(),
		Usage: contracts.UsageCost{
			EstimatedInputTokens:  estimateTokens(req.Input),
			EstimatedOutputTokens: 128,
			EstimatedCostUSD:      0.0005,
		},
	}

	s.store.SaveInference(storage.InferenceLog{
		RequestID: requestID,
		TenantID:  req.TenantID,
		TaskType:  string(taskType),
		CreatedAt: s.now().UTC(),
		Response:  resp,
	})
	return resp, nil
}

func (s *Server) runAsyncJob(job storage.Job) {
	s.store.StartJob(job.JobID)
	start := s.now()
	resp, err := s.executeTask(types.TaskType(job.TaskType), job.RequestID, contracts.AIRequest{TenantID: job.TenantID, Input: job.Input, Options: job.Options}, start, nil)
	if err != nil {
		s.store.FailJob(job.JobID, err.Error())
		return
	}
	s.store.CompleteJob(job.JobID, resp)
}

func (s *Server) handleJobStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !isAuthorized(r) {
		writeError(w, http.StatusUnauthorized, "missing or invalid authorization")
		return
	}
	tenantHeader := strings.TrimSpace(r.Header.Get("X-Tenant-ID"))
	if tenantHeader == "" {
		writeError(w, http.StatusBadRequest, "missing X-Tenant-ID header")
		return
	}
	jobID := strings.TrimPrefix(r.URL.Path, "/api/ai/jobs/")
	if strings.TrimSpace(jobID) == "" {
		writeError(w, http.StatusBadRequest, "missing job id")
		return
	}
	job, ok := s.store.JobByID(jobID)
	if !ok {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	if job.TenantID != tenantHeader {
		writeError(w, http.StatusForbidden, "tenant is not allowed to access this job")
		return
	}
	writeJSON(w, http.StatusOK, contracts.JobStatusResponse{
		JobID:       job.JobID,
		RequestID:   job.RequestID,
		TenantID:    job.TenantID,
		TaskType:    job.TaskType,
		Status:      job.Status,
		Error:       job.Error,
		CreatedAt:   job.CreatedAt,
		UpdatedAt:   job.UpdatedAt,
		CompletedAt: job.CompletedAt,
		Response:    job.Response,
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !isAuthorized(r) {
		writeError(w, http.StatusUnauthorized, "missing or invalid authorization")
		return
	}
	writeJSON(w, http.StatusOK, s.metrics.Snapshot())
}

func isAuthorized(r *http.Request) bool {
	a := strings.TrimSpace(r.Header.Get("Authorization"))
	return strings.HasPrefix(a, "Bearer ") && len(a) > len("Bearer ")
}

func estimateTokens(input map[string]any) int {
	if len(input) == 0 {
		return 0
	}
	b, err := json.Marshal(input)
	if err != nil {
		return 0
	}
	// Very rough approximation.
	return len(b) / 4
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]any{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func newRequestID() string {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err != nil {
		return "req-fallback"
	}
	return "req_" + hex.EncodeToString(buf)
}

func uniqueStrings(in []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	s, ok := v.(string)
	if ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

// ErrServerClosed wraps expected shutdown behavior.
var ErrServerClosed = errors.New("server closed")
