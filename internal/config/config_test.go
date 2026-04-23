package config

import (
	"os"
	"testing"
)

func TestLoadFromEnvDefaults(t *testing.T) {
	t.Setenv("WHEMAIL_AI_ADDRESS", "")
	t.Setenv("WHEMAIL_AI_READ_TIMEOUT_SECONDS", "")
	t.Setenv("WHEMAIL_AI_WRITE_TIMEOUT_SECONDS", "")
	t.Setenv("WHEMAIL_AI_MAX_REQUEST_BYTES", "")
	t.Setenv("WHEMAIL_AI_RATE_LIMIT_RPM", "")

	cfg, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv() error = %v", err)
	}
	if cfg.ServerAddress == "" {
		t.Fatal("expected default ServerAddress")
	}
	if cfg.ReadTimeout <= 0 || cfg.WriteTimeout <= 0 || cfg.MaxRequestBytes <= 0 || cfg.RateLimitRPM <= 0 {
		t.Fatal("expected positive default values")
	}
}

func TestLoadFromEnvInvalid(t *testing.T) {
	t.Setenv("WHEMAIL_AI_RATE_LIMIT_RPM", "-1")
	_, err := LoadFromEnv()
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestGetenvIntInvalidFallsBack(t *testing.T) {
	os.Setenv("_TEST_INT", "x")
	defer os.Unsetenv("_TEST_INT")
	if v := getenvInt("_TEST_INT", 42); v != 42 {
		t.Fatalf("expected fallback 42, got %d", v)
	}
}
