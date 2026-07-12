.DEFAULT_GOAL := check

PRESET ?= debug
GIT_COMMIT_COUNT = $(shell git rev-list --count HEAD)
GIT_COMMIT = $(shell git rev-parse --short=12 HEAD)
GIT_BUILD_ID = $(shell printf "%06d-%s" $(GIT_COMMIT_COUNT) $(GIT_COMMIT))
GIT_BUILD_DIR = build/by-commit/$(GIT_BUILD_ID)/$(PRESET)
NEW_ENGINE ?=
BASE_ENGINE ?=
NEW_ENGINE_ARGS ?=
BASE_ENGINE_ARGS ?=
BASE ?=
BOOK ?= eval/openings/systematic-2ply-v1.txt
GAMES ?= 1000
SMOKE_GAMES ?= 100
TIMEOUT_MS ?= 1000
GO_MOVETIME_MS ?= 900
SEQUENTIAL_NULL ?= 0.50
SEQUENTIAL_ALT ?= 0.55
SEQUENTIAL_ALPHA ?= 0.05
SEQUENTIAL_BETA ?= 0.05

.PHONY: configure build test format format-check tidy tidy-fix fix check full-check ready clean debug release require-clean git-configure git-build git-test require-base require-eval-engines eval-gate eval-smoke

configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build --preset $(PRESET)

test: build
	ctest --preset $(PRESET)

format: configure
	cmake --build --preset $(PRESET) --target format

format-check: configure
	cmake --build --preset $(PRESET) --target format-check

tidy: build
	cmake --build --preset $(PRESET) --target tidy

tidy-fix: build
	cmake --build --preset $(PRESET) --target tidy-fix

fix:
	$(MAKE) tidy-fix PRESET=$(PRESET)
	$(MAKE) format PRESET=$(PRESET)

check:
	$(MAKE) test PRESET=$(PRESET)
	$(MAKE) format-check PRESET=$(PRESET)
	$(MAKE) tidy PRESET=$(PRESET)

full-check:
	$(MAKE) check PRESET=debug
	$(MAKE) check PRESET=release

ready:
	$(MAKE) fix PRESET=debug
	$(MAKE) full-check

clean:
	cmake -E rm -rf build/debug build/release

debug:
	$(MAKE) check PRESET=debug

release:
	$(MAKE) check PRESET=release

require-clean:
	@test -z "$$(git status --porcelain)" || \
	  (echo "working tree is dirty; commit or stash changes before using git-build" >&2; \
	   git status --short >&2; \
	   exit 1)

require-base:
	@test -n "$(BASE)" || \
	  (echo "BASE=<build-id|build-dir|engine-binary> is required" >&2; \
	   exit 1)

require-eval-engines:
	@test -n "$(NEW_ENGINE)" || \
	  (echo "NEW_ENGINE=<engine-binary-name> is required" >&2; \
	   exit 1)
	@test -n "$(BASE_ENGINE)" || \
	  (echo "BASE_ENGINE=<engine-binary-name> is required" >&2; \
	   exit 1)

git-configure: require-clean
	cmake --preset $(PRESET) -B $(GIT_BUILD_DIR)

git-build: git-configure
	cmake --build $(GIT_BUILD_DIR)

git-test: git-build
	ctest --test-dir $(GIT_BUILD_DIR) --output-on-failure

eval-gate: require-base require-eval-engines git-test
	$(GIT_BUILD_DIR)/runner/poe2_runner eval \
	  --new-build $(GIT_BUILD_DIR) \
	  --base $(BASE) \
	  --new-engine $(NEW_ENGINE) \
	  --base-engine $(BASE_ENGINE) \
	  $(if $(NEW_ENGINE_ARGS),--new-engine-args '$(NEW_ENGINE_ARGS)',) \
	  $(if $(BASE_ENGINE_ARGS),--base-engine-args '$(BASE_ENGINE_ARGS)',) \
	  --preset $(PRESET) \
	  --kind gate \
	  --opening-book $(BOOK) \
	  --shuffle-openings \
	  --games $(GAMES) \
	  --timeout-ms $(TIMEOUT_MS) \
	  --go-movetime-ms $(GO_MOVETIME_MS) \
	  --sequential-stop \
	  --sequential-null $(SEQUENTIAL_NULL) \
	  --sequential-alt $(SEQUENTIAL_ALT) \
	  --sequential-alpha $(SEQUENTIAL_ALPHA) \
	  --sequential-beta $(SEQUENTIAL_BETA) \
	  --require-accept-alt

eval-smoke: require-base require-eval-engines git-test
	$(GIT_BUILD_DIR)/runner/poe2_runner eval \
	  --new-build $(GIT_BUILD_DIR) \
	  --base $(BASE) \
	  --new-engine $(NEW_ENGINE) \
	  --base-engine $(BASE_ENGINE) \
	  $(if $(NEW_ENGINE_ARGS),--new-engine-args '$(NEW_ENGINE_ARGS)',) \
	  $(if $(BASE_ENGINE_ARGS),--base-engine-args '$(BASE_ENGINE_ARGS)',) \
	  --preset $(PRESET) \
	  --kind smoke \
	  --opening-book $(BOOK) \
	  --shuffle-openings \
	  --games $(SMOKE_GAMES) \
	  --timeout-ms $(TIMEOUT_MS) \
	  --go-movetime-ms $(GO_MOVETIME_MS)
