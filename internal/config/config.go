package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"
)

const (
	defaultAddress         = ":8080"
	defaultReadTimeoutSec  = 15
	defaultWriteTimeoutSec = 15
	defaultMaxRequestBytes = int64(1 << 20) // 1 MiB
	defaultRateLimitRPM    = 120
)

// Config contains runtime configuration for the AI API service.
type Config struct {
	ServerAddress   string
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	MaxRequestBytes int64
	RateLimitRPM    int
}

// LoadFromEnv builds config from environment variables with safe defaults.
func LoadFromEnv() (Config, error) {
	cfg := Config{
		ServerAddress:   getenv("WHEMAIL_AI_ADDRESS", defaultAddress),
		ReadTimeout:     time.Duration(getenvInt("WHEMAIL_AI_READ_TIMEOUT_SECONDS", defaultReadTimeoutSec)) * time.Second,
		WriteTimeout:    time.Duration(getenvInt("WHEMAIL_AI_WRITE_TIMEOUT_SECONDS", defaultWriteTimeoutSec)) * time.Second,
		MaxRequestBytes: getenvInt64("WHEMAIL_AI_MAX_REQUEST_BYTES", defaultMaxRequestBytes),
		RateLimitRPM:    getenvInt("WHEMAIL_AI_RATE_LIMIT_RPM", defaultRateLimitRPM),
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
