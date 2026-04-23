package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/iSundram/WhyMailAi/internal/config"
	"github.com/iSundram/WhyMailAi/internal/orchestration"
	"github.com/iSundram/WhyMailAi/pkg/contracts"
	"github.com/iSundram/WhyMailAi/pkg/types"
)

const (
	modelName    = "whymail-baseline"
	modelVersion = "0.1.0"
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
	now       func() time.Time
	requestID func() string
}

// NewServer constructs an API server.
func NewServer(cfg config.Config) *Server {
	return &Server{
		cfg:       cfg,
		router:    orchestration.NewRouter(),
		limiter:   newRateLimiter(cfg.RateLimitRPM),
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
		if r.Method != http.MethodPost {
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
		if !s.limiter.allow(tenantHeader, s.now()) {
			writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxRequestBytes)
		defer r.Body.Close()

		var req contracts.AIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid json body")
			return
		}
		if strings.TrimSpace(req.TenantID) == "" {
			writeError(w, http.StatusBadRequest, "tenant_id is required")
			return
		}
		if req.TenantID != tenantHeader {
			writeError(w, http.StatusForbidden, "tenant header and payload mismatch")
			return
		}

		if strings.EqualFold(r.URL.Query().Get("async"), "true") {
			resp := contracts.AsyncAcceptedResponse{
				JobID:      s.requestID(),
				RequestID:  s.requestID(),
				TenantID:   req.TenantID,
				TaskType:   string(taskType),
				AcceptedAt: s.now().UTC(),
			}
			writeJSON(w, http.StatusAccepted, resp)
			return
		}

		out := s.router.Run(taskType, req.Input)
		latency := s.now().Sub(start).Milliseconds()
		if latency < 0 {
			latency = 0
		}

		resp := contracts.AIResponse{
			RequestID:    s.requestID(),
			TenantID:     req.TenantID,
			ModelName:    modelName,
			ModelVersion: modelVersion,
			TaskType:     taskType,
			LatencyMS:    latency,
			Confidence:   out.Confidence,
			Result:       out.Result,
			Explanations: out.Explanations,
			SafetyFlags:  out.SafetyFlags,
			CacheHit:     false,
			Timestamp:    s.now().UTC(),
			Usage: contracts.UsageCost{
				EstimatedInputTokens:  estimateTokens(req.Input),
				EstimatedOutputTokens: 128,
				EstimatedCostUSD:      0.0005,
			},
		}

		writeJSON(w, http.StatusOK, resp)
	}
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

// ErrServerClosed wraps expected shutdown behavior.
var ErrServerClosed = errors.New("server closed")
