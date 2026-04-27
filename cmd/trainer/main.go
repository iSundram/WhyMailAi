// cmd/trainer — WhyMail AI Training Job Launcher
//
// This command launches Python training jobs for the WhyMail AI ML pipeline.
// It orchestrates data download, preprocessing, training, and evaluation.
//
// Usage
//
//	go run ./cmd/trainer [flags]
//
// Flags
//
//	--task       Training task: spam | phishing | summarizer | embedder |
//	             priority | all (default: all)
//	--data-dir   Path to data directory (default: data/)
//	--models-dir Path to models output directory (default: models/)
//	--python     Path to Python interpreter (default: python3)
//	--dry-run    Print commands without executing them
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/iSundram/WhyMailAi/internal/training"
)

func main() {
	task := flag.String("task", "all", "Training task: spam|phishing|summarizer|embedder|priority|all")
	dataDir := flag.String("data-dir", "data", "Root data directory")
	modelsDir := flag.String("models-dir", "models", "Root models output directory")
	python := flag.String("python", "python3", "Python interpreter")
	profile := flag.String("profile", "standard", "Training profile: standard|advanced")
	includeExtendedHF := flag.Bool("include-extended-hf", false, "Include optional extra HuggingFace spam datasets during download")
	kaggleDataset := flag.String("kaggle-dataset", "", "Optional Kaggle dataset slug for additional spam data")
	dryRun := flag.Bool("dry-run", false, "Print commands without executing")
	flag.Parse()

	registry := training.NewRegistry()

	if *profile == "advanced" {
		*includeExtendedHF = true
	}

	tasks := expandTask(*task)
	for _, t := range tasks {
		runTask(
			t,
			*dataDir,
			*modelsDir,
			*python,
			*profile,
			*includeExtendedHF,
			*kaggleDataset,
			*dryRun,
			registry,
		)
	}

	fmt.Println("--- Training summary ---")
	for _, job := range registry.List() {
		status := string(job.Status)
		if job.Error != "" {
			status += " (" + job.Error + ")"
		}
		fmt.Printf("  %-25s %s\n", job.Name, status)
	}
}

func expandTask(task string) []string {
	if task == "all" {
		return []string{"download", "spam", "phishing", "summarizer", "embedder", "priority"}
	}
	return []string{task}
}

func runTask(
	task, dataDir, modelsDir, python, profile string,
	includeExtendedHF bool,
	kaggleDataset string,
	dryRun bool,
	registry *training.Registry,
) {
	id := fmt.Sprintf("%s-%d", task, time.Now().UnixNano())
	job := registry.Create(training.Job{
		ID:          id,
		Name:        task,
		ModelDomain: task,
		CreatedAt:   time.Now().UTC(),
	})

	var args []string
	switch task {
	case "download":
		args = []string{"-m", "ml.data.download", dataDir}
		if includeExtendedHF {
			args = append(args, "--include-extended-hf")
		}
		if kaggleDataset != "" {
			args = append(args, "--kaggle-dataset", kaggleDataset)
		}
	case "spam":
		args = []string{
			"-m", "ml.train.spam_classifier",
			"--data-dir", filepath.Join(dataDir, "spam"),
			"--output-dir", filepath.Join(modelsDir, "spam_classifier"),
		}
		if profile == "advanced" {
			args = append(args,
				"--model-name", "microsoft/deberta-v3-base",
				"--epochs", "5",
				"--batch-size", "16",
			)
		}
	case "phishing":
		args = []string{
			"-m", "ml.train.phishing_classifier",
			"--data-dir", filepath.Join(dataDir, "phishing"),
			"--output-dir", filepath.Join(modelsDir, "phishing_classifier"),
		}
		if profile == "advanced" {
			args = append(args,
				"--model-name", "microsoft/deberta-v3-base",
				"--epochs", "5",
				"--batch-size", "16",
			)
		}
	case "summarizer":
		args = []string{
			"-m", "ml.train.summarizer",
			"--data-dir", filepath.Join(dataDir, "summarization"),
			"--output-dir", filepath.Join(modelsDir, "summarizer"),
		}
		if profile == "advanced" {
			args = append(args,
				"--model-name", "facebook/bart-large-cnn",
				"--epochs", "4",
				"--batch-size", "2",
			)
		}
	case "embedder":
		args = []string{
			"-m", "ml.train.embedder",
			"--output-dir", filepath.Join(modelsDir, "embedder"),
		}
	case "priority":
		args = []string{
			"-m", "ml.train.priority_ranker",
			"--output-dir", filepath.Join(modelsDir, "priority_ranker"),
			"--data-dir", filepath.Join(dataDir, "spam"),
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown task: %s\n", task)
		registry.Fail(job.ID, "unknown task")
		return
	}

	fullCmd := append([]string{python}, args...)
	fmt.Printf("[%s] Running: %v\n", task, fullCmd)

	if dryRun {
		registry.Complete(job.ID, nil, "")
		return
	}

	registry.Start(job.ID)

	cmd := exec.Command(python, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()

	if err := cmd.Run(); err != nil {
		registry.Fail(job.ID, err.Error())
		fmt.Fprintf(os.Stderr, "[%s] FAILED: %v\n", task, err)
		return
	}

	registry.Complete(job.ID, nil, filepath.Join(modelsDir, task))
	fmt.Printf("[%s] DONE\n", task)
}
