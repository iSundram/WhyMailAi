package main

import (
	"errors"
	"log"
	"net/http"

	"github.com/iSundram/WhyMailAi/internal/api"
	"github.com/iSundram/WhyMailAi/internal/config"
)

func main() {
	cfg, err := config.LoadFromEnv()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	srv := api.NewServer(cfg)
	httpServer := &http.Server{
		Addr:         cfg.ServerAddress,
		Handler:      srv.Handler(),
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	log.Printf("ai-api listening on %s", cfg.ServerAddress)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("listen: %v", err)
	}
}
