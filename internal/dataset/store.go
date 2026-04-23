// Package dataset manages training dataset records and metadata.
package dataset

import (
	"sync"
	"time"
)

// DatasetSplit indicates which portion of a dataset a record belongs to.
type DatasetSplit string

const (
	SplitTrain DatasetSplit = "train"
	SplitVal   DatasetSplit = "val"
	SplitTest  DatasetSplit = "test"
)

// TaskDomain classifies which ML task a record is relevant to.
type TaskDomain string

const (
	DomainSpam          TaskDomain = "spam"
	DomainPhishing      TaskDomain = "phishing"
	DomainSummarization TaskDomain = "summarization"
	DomainPriority      TaskDomain = "priority"
)

// DataRecord is a single labelled example stored for training.
type DataRecord struct {
	ID        string         `json:"id"`
	TenantID  string         `json:"tenant_id"`
	Domain    TaskDomain     `json:"domain"`
	Split     DatasetSplit   `json:"split"`
	Text      string         `json:"text"`
	Label     int            `json:"label"`
	Summary   string         `json:"summary,omitempty"`
	Source    string         `json:"source"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
}

// DatasetVersion tracks a named, immutable snapshot of a dataset.
type DatasetVersion struct {
	Version     string         `json:"version"`
	Domain      TaskDomain     `json:"domain"`
	TrainCount  int            `json:"train_count"`
	ValCount    int            `json:"val_count"`
	TestCount   int            `json:"test_count"`
	CreatedAt   time.Time      `json:"created_at"`
	Description string         `json:"description,omitempty"`
}

// Store is an in-memory dataset record store.  In production this would be
// backed by a relational DB or object store.
type Store struct {
	mu       sync.RWMutex
	records  []DataRecord
	versions []DatasetVersion
}

// NewStore creates an empty dataset store.
func NewStore() *Store {
	return &Store{}
}

// Add appends a single record to the store.
func (s *Store) Add(r DataRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if r.CreatedAt.IsZero() {
		r.CreatedAt = time.Now().UTC()
	}
	s.records = append(s.records, r)
}

// AddBatch appends multiple records atomically.
func (s *Store) AddBatch(records []DataRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	for i := range records {
		if records[i].CreatedAt.IsZero() {
			records[i].CreatedAt = now
		}
	}
	s.records = append(s.records, records...)
}

// Query returns all records matching the given domain and split.
// Pass empty strings to match all values.
func (s *Store) Query(domain TaskDomain, split DatasetSplit) []DataRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []DataRecord
	for _, r := range s.records {
		if (domain == "" || r.Domain == domain) &&
			(split == "" || r.Split == split) {
			out = append(out, r)
		}
	}
	return out
}

// Count returns the number of stored records.
func (s *Store) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.records)
}

// TagVersion creates a named version snapshot for the current record set.
func (s *Store) TagVersion(v DatasetVersion) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if v.CreatedAt.IsZero() {
		v.CreatedAt = time.Now().UTC()
	}
	s.versions = append(s.versions, v)
}

// Versions returns all tagged dataset versions.
func (s *Store) Versions() []DatasetVersion {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]DatasetVersion, len(s.versions))
	copy(out, s.versions)
	return out
}
