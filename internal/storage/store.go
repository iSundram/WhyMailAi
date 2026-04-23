package storage

import (
	"sync"
	"time"

	"github.com/iSundram/WhyMailAi/pkg/contracts"
)

const (
	JobStatusQueued    = "queued"
	JobStatusRunning   = "running"
	JobStatusCompleted = "completed"
	JobStatusFailed    = "failed"
)

// InferenceLog is stored for auditability.
type InferenceLog struct {
	RequestID string
	TenantID  string
	TaskType  string
	CreatedAt time.Time
	Response  contracts.AIResponse
}

// FeedbackRecord stores feedback labels.
type FeedbackRecord struct {
	TenantID   string         `json:"tenant_id"`
	MessageID  string         `json:"message_id"`
	Feedback   string         `json:"feedback"`
	Attributes map[string]any `json:"attributes,omitempty"`
	CreatedAt  time.Time      `json:"created_at"`
}

// Job tracks async execution.
type Job struct {
	JobID       string                `json:"job_id"`
	RequestID   string                `json:"request_id"`
	TenantID    string                `json:"tenant_id"`
	TaskType    string                `json:"task_type"`
	Input       map[string]any        `json:"-"`
	Options     map[string]any        `json:"-"`
	Status      string                `json:"status"`
	Error       string                `json:"error,omitempty"`
	CreatedAt   time.Time             `json:"created_at"`
	UpdatedAt   time.Time             `json:"updated_at"`
	CompletedAt *time.Time            `json:"completed_at,omitempty"`
	Response    *contracts.AIResponse `json:"response,omitempty"`
}

// MemoryStore is an in-memory persistence layer.
type MemoryStore struct {
	mu        sync.RWMutex
	jobs      map[string]*Job
	logs      []InferenceLog
	feedbacks []FeedbackRecord
}

// NewMemoryStore creates memory-backed storage.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{jobs: map[string]*Job{}, logs: []InferenceLog{}, feedbacks: []FeedbackRecord{}}
}

func (s *MemoryStore) SaveInference(log InferenceLog) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.logs = append(s.logs, log)
}

func (s *MemoryStore) SaveFeedback(record FeedbackRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.feedbacks = append(s.feedbacks, record)
}

func (s *MemoryStore) NewJob(jobID, requestID, tenantID, taskType string, input, options map[string]any) Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	job := &Job{
		JobID:     jobID,
		RequestID: requestID,
		TenantID:  tenantID,
		TaskType:  taskType,
		Input:     input,
		Options:   options,
		Status:    JobStatusQueued,
		CreatedAt: now,
		UpdatedAt: now,
	}
	s.jobs[jobID] = job
	return *job
}

func (s *MemoryStore) StartJob(jobID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if job, ok := s.jobs[jobID]; ok {
		job.Status = JobStatusRunning
		job.UpdatedAt = time.Now().UTC()
	}
}

func (s *MemoryStore) CompleteJob(jobID string, response contracts.AIResponse) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if job, ok := s.jobs[jobID]; ok {
		now := time.Now().UTC()
		job.Status = JobStatusCompleted
		job.UpdatedAt = now
		job.CompletedAt = &now
		resp := response
		job.Response = &resp
	}
}

func (s *MemoryStore) FailJob(jobID, err string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if job, ok := s.jobs[jobID]; ok {
		now := time.Now().UTC()
		job.Status = JobStatusFailed
		job.Error = err
		job.UpdatedAt = now
		job.CompletedAt = &now
	}
}

func (s *MemoryStore) JobByID(jobID string) (Job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.jobs[jobID]
	if !ok {
		return Job{}, false
	}
	copy := *job
	return copy, true
}
