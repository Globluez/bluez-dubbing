.PHONY: start-ui start-api stack-up stop restart check-ports test install-dep install-model-dep

ROOT := $(CURDIR)
BACKEND_ROOT := $(ROOT)/apps/backend
PYTHONPATH_BASE := $(BACKEND_ROOT):$(ROOT)
UV ?= uv
RELOAD ?= --reload
UI_PORT ?= 5173
SERVICES := asr translation tts orchestrator
export UV_CACHE_DIR := $(ROOT)/.uv_cache
# Prefer system cuDNN over bundled nvidia-cudnn-cu12 to avoid version mismatch
export LD_LIBRARY_PATH := /usr/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH)

define start_service
	@echo "▶ starting $(1) service on port $(2)"
	@cd apps/backend/services/$(1) && $(strip $(3)) PYTHONPATH=$(PYTHONPATH_BASE) $(UV) run uvicorn app.main:app $(RELOAD) --host 0.0.0.0 --port $(2) &
endef

define stop_port
	@-fuser -k $(1)/tcp 2>/dev/null || true
endef

stack-up:
	@echo "Starting Bluez dubbing stack (ASR + translation + TTS + orchestrator)…"
	$(call start_service,asr,8001,)
	$(call start_service,translation,8002,)
	$(call start_service,tts,8003,)
	$(call start_service,orchestrator,8000,)
	@echo "All backend services running. REST API ⇒ http://localhost:8000/api"

start-api:
	@$(MAKE) stack-up

start-ui:
	@$(MAKE) stack-up
	@echo "Starting Bluez dubbing UI…"
	@cd apps/frontend && uv run python -m http.server $(UI_PORT) &
	@echo "UI running at http://localhost:$(UI_PORT)"

stop:
	@echo "Stopping Bluez dubbing stack…"
	@-pkill -f "uvicorn app.main:app" || true
	@-pkill -f "http.server $(UI_PORT)" || true
	@sleep 1
	$(call stop_port,8000)
	$(call stop_port,8001)
	$(call stop_port,8002)
	$(call stop_port,8003)
	@echo "All services stopped."

restart: stop
	@sleep 1
	@$(MAKE) stack-up

restart-ui: stop
	@sleep 1
	@$(MAKE) start-ui

check-ports:
	@echo "Checking port status..."
	@echo "Port 8000 (orchestrator):"
	@-lsof -i :8000 || echo "  Available"
	@echo "Port 8001 (ASR):"
	@-lsof -i :8001 || echo "  Available"
	@echo "Port 8002 (translation):"
	@-lsof -i :8002 || echo "  Available"
	@echo "Port 8003 (TTS):"
	@-lsof -i :8003 || echo "  Available"

test:
	cd apps/backend/services/asr && $(UV) run --with pytest pytest
	cd apps/backend/services/translation && $(UV) run --with pytest pytest
	cd apps/backend/services/tts && $(UV) run --with pytest pytest
	cd apps/backend/services/orchestrator && $(UV) run --with pytest --with pytest-asyncio pytest

install-dep:
	@echo "Installing all service dependencies..."
	@for service in $(SERVICES); do \
		echo "▶ $$service"; \
		( cd apps/backend/services/$$service && $(UV) sync ); \
	done
	@$(MAKE) --no-print-directory install-model-dep
	@echo "All dependencies installed."

install-model-dep:
	@echo "Installing model dependencies..."
	@for service in $(SERVICES); do \
		models_dir="apps/backend/services/$$service/models"; \
		if [ -d "$$models_dir" ]; then \
			echo "▶ $$service models"; \
			for model_dir in "$$models_dir"/*; do \
				if [ -d "$$model_dir" ]; then \
					if [ -f "$$model_dir/pyproject.toml" ]; then \
						echo "  -> $$model_dir"; \
						( cd "$$model_dir" && $(UV) sync ); \
					else \
						echo "  -> $$model_dir (skipping: no pyproject.toml)"; \
					fi; \
				fi; \
			done; \
		fi; \
	done
