package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"
)

const (
	defaultAddress                = ":8080"
	defaultReadTimeoutSec         = 15
	defaultWriteTimeoutSec        = 15
	defaultMaxRequestBytes        = int64(1 << 20) // 1 MiB
	defaultRateLimitRPM           = 120
	defaultRemoteInferenceURL     = ""  // empty = disabled; set to http://localhost:9090 to enable
	defaultRemoteTimeoutSec       = 30
)

// Config contains runtime configuration for the AI API service.
type Config struct {
	ServerAddress      string
	ReadTimeout        time.Duration
	WriteTimeout       time.Duration
	MaxRequestBytes    int64
	RateLimitRPM       int
	// RemoteInferenceURL is the base URL of the Python ML inference server.
	// When non-empty the orchestration layer will prefer the remote backend
	// over the built-in heuristic backends.
	RemoteInferenceURL string
	RemoteTimeoutSec   int
}

// LoadFromEnv builds config from environment variables with safe defaults.
func LoadFromEnv() (Config, error) {
	cfg := Config{
		ServerAddress:      getenv("WHYMAIL_AI_ADDRESS", defaultAddress),
		ReadTimeout:        time.Duration(getenvInt("WHYMAIL_AI_READ_TIMEOUT_SECONDS", defaultReadTimeoutSec)) * time.Second,
		WriteTimeout:       time.Duration(getenvInt("WHYMAIL_AI_WRITE_TIMEOUT_SECONDS", defaultWriteTimeoutSec)) * time.Second,
		MaxRequestBytes:    getenvInt64("WHYMAIL_AI_MAX_REQUEST_BYTES", defaultMaxRequestBytes),
		RateLimitRPM:       getenvInt("WHYMAIL_AI_RATE_LIMIT_RPM", defaultRateLimitRPM),
		RemoteInferenceURL: getenv("WHYMAIL_REMOTE_INFERENCE_URL", defaultRemoteInferenceURL),
		RemoteTimeoutSec:   getenvInt("WHYMAIL_REMOTE_TIMEOUT_SEC", defaultRemoteTimeoutSec),
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

// Validate checks config invariants.
func (c Config) Validate() error {
	if c.ServerAddress == "" {
		return errors.New("server address is required")
	}
	if c.ReadTimeout <= 0 {
		return fmt.Errorf("read timeout must be positive: %s", c.ReadTimeout)
	}
	if c.WriteTimeout <= 0 {
		return fmt.Errorf("write timeout must be positive: %s", c.WriteTimeout)
	}
	if c.MaxRequestBytes <= 0 {
		return fmt.Errorf("max request bytes must be positive: %d", c.MaxRequestBytes)
	}
	if c.RateLimitRPM <= 0 {
		return fmt.Errorf("rate limit rpm must be positive: %d", c.RateLimitRPM)
	}
	return nil
}

func getenv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func getenvInt64(key string, fallback int64) int64 {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if i, err := strconv.ParseInt(v, 10, 64); err == nil {
			return i
		}
	}
	return fallback
}
