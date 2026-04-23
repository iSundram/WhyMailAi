package types

// TaskType identifies a supported AI operation.
type TaskType string

const (
	TaskSpamClassification TaskType = "spam-classification"
	TaskPhishingClassify   TaskType = "phishing-classification"
	TaskEmailDrafting      TaskType = "email-drafting"
	TaskEmailRewrite       TaskType = "email-rewrite"
	TaskThreadSummary      TaskType = "thread-summary"
	TaskSemanticSearch     TaskType = "semantic-search"
	TaskPriorityRanking    TaskType = "priority-ranking"
	TaskAdminAnomalyDetect TaskType = "admin-anomaly-detection"
	TaskFeedbackIngestion  TaskType = "feedback-ingestion"
)
