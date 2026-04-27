// cmd/evaluator — WhyMail AI Model Evaluation Runner
//
// This command triggers the Python evaluation harness and prints a
// human-readable summary of model quality.
//
// Usage
//
//	go run ./cmd/evaluator [flags]
//
// Flags
//
//	--models-dir  Path to root models directory (default: models/)
//	--data-dir    Path to root data directory (default: data/)
//	--report-out  Write JSON report to this file (optional)
//	--python      Python interpreter (default: python3)
//	--dry-run     Print commands without executing
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
)

func main() {
	modelsDir := flag.String("models-dir", "models", "Root models directory")
	dataDir := flag.String("data-dir", "data", "Root data directory")
	reportOut := flag.String("report-out", "", "Write JSON report to this file")
	python := flag.String("python", "python3", "Python interpreter")
	dryRun := flag.Bool("dry-run", false, "Print command without executing")
	flag.Parse()

	args := []string{
		"-c",
		fmt.Sprintf(
			`from ml.evaluate.harness import run_full_harness; `+
				`results = run_full_harness(%q, %q, report_path=%q or None); `+
				`[print(r.model_name, "→", "PROMOTED" if r.promoted else "FAILED") for r in results]`,
			*modelsDir, *dataDir, *reportOut,
		),
	}

	fullCmd := append([]string{*python}, args...)
	fmt.Printf("Running: %v\n", fullCmd)

	if *dryRun {
		return
	}

	cmd := exec.Command(*python, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Evaluator failed: %v\n", err)
		os.Exit(1)
	}
}
