package telemetry

import "sync"

// Snapshot is an immutable metrics view.
type Snapshot struct {
	Counters           map[string]int64 `json:"counters"`
	TaskCount          map[string]int64 `json:"task_count"`
	LatencyAvgMSByTask map[string]int64 `json:"latency_avg_ms_by_task"`
}

// Metrics tracks in-memory counters and latency aggregates.
type Metrics struct {
	mu           sync.RWMutex
	counters     map[string]int64
	taskCount    map[string]int64
	latencySumMs map[string]int64
}

// NewMetrics creates metrics collector.
func NewMetrics() *Metrics {
	return &Metrics{
		counters:     map[string]int64{},
		taskCount:    map[string]int64{},
		latencySumMs: map[string]int64{},
	}
}

func (m *Metrics) Inc(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.counters[name]++
}

func (m *Metrics) ObserveTaskLatency(task string, ms int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.taskCount[task]++
	m.latencySumMs[task] += ms
}

func (m *Metrics) Snapshot() Snapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()
	counters := make(map[string]int64, len(m.counters))
	taskCount := make(map[string]int64, len(m.taskCount))
	latAvg := make(map[string]int64, len(m.taskCount))
	for k, v := range m.counters {
		counters[k] = v
	}
	for k, v := range m.taskCount {
		taskCount[k] = v
		sum := m.latencySumMs[k]
		if v > 0 {
			latAvg[k] = sum / v
		}
	}
	return Snapshot{Counters: counters, TaskCount: taskCount, LatencyAvgMSByTask: latAvg}
}
