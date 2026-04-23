// Package training manages training job lifecycle and metadata.
package training

import (
	"sync"
	"time"
)

// JobStatus represents the lifecycle state of a training job.
type JobStatus string

const (
	JobQueued    JobStatus = "queued"
	JobRunning   JobStatus = "running"
	JobCompleted JobStatus = "completed"
	JobFailed    JobStatus = "failed"
)

// Job represents a single training or fine-tuning run.
type Job struct {
	ID            string         `json:"id"`
	Name          string         `json:"name"`
	ModelDomain   string         `json:"model_domain"` // e.g. "spam", "phishing"
	BaseModel     string         `json:"base_model"`
	DatasetVersion string        `json:"dataset_version"`
	Config        map[string]any `json:"config"`
	Status        JobStatus      `json:"status"`
	Error         string         `json:"error,omitempty"`
	Metrics       map[string]float64 `json:"metrics,omitempty"`
	OutputDir     string         `json:"output_dir,omitempty"`
	CreatedAt     time.Time      `json:"created_at"`
	StartedAt     *time.Time     `json:"started_at,omitempty"`
	CompletedAt   *time.Time     `json:"completed_at,omitempty"`
}

// Registry tracks all training jobs.
type Registry struct {
	mu   sync.RWMutex
	jobs map[string]*Job
}

// NewRegistry creates an empty training job registry.
func NewRegistry() *Registry {
	return &Registry{jobs: map[string]*Job{}}
}

// Create registers a new job in Queued state.
func (r *Registry) Create(job Job) Job {
	r.mu.Lock()
	defer r.mu.Unlock()
	if job.CreatedAt.IsZero() {
		job.CreatedAt = time.Now().UTC()
	}
	job.Status = JobQueued
	cp := job
	r.jobs[job.ID] = &cp
	return *r.jobs[job.ID]
}

// Start transitions a job to Running.
func (r *Registry) Start(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	j, ok := r.jobs[id]
	if !ok {
		return false
	}
	now := time.Now().UTC()
	j.Status = JobRunning
	j.StartedAt = &now
	return true
}

// Complete transitions a job to Completed and records metrics.
func (r *Registry) Complete(id string, metrics map[string]float64, outputDir string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	j, ok := r.jobs[id]
	if !ok {
		return false
	}
	now := time.Now().UTC()
	j.Status = JobCompleted
	j.CompletedAt = &now
	j.Metrics = metrics
	j.OutputDir = outputDir
	return true
}

// Fail transitions a job to Failed with an error message.
func (r *Registry) Fail(id, errMsg string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	j, ok := r.jobs[id]
	if !ok {
		return false
	}
	now := time.Now().UTC()
	j.Status = JobFailed
	j.Error = errMsg
	j.CompletedAt = &now
	return true
}

// Get returns a copy of the job with the given ID.
func (r *Registry) Get(id string) (Job, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	j, ok := r.jobs[id]
	if !ok {
		return Job{}, false
	}
	return *j, true
}

// List returns all jobs.
func (r *Registry) List() []Job {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Job, 0, len(r.jobs))
	for _, j := range r.jobs {
		out = append(out, *j)
	}
	return out
}
