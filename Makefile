.DEFAULT_GOAL := check

PRESET ?= debug
GIT_COMMIT_COUNT = $(shell git rev-list --count HEAD)
GIT_COMMIT = $(shell git rev-parse --short=12 HEAD)
GIT_BUILD_ID = $(shell printf "%06d-%s" $(GIT_COMMIT_COUNT) $(GIT_COMMIT))
GIT_BUILD_DIR = build/by-commit/$(GIT_BUILD_ID)/$(PRESET)

.PHONY: configure build test format format-check tidy tidy-fix fix check clean debug release require-clean git-configure git-build git-test

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
	$(MAKE) format PRESET=$(PRESET)
	$(MAKE) tidy-fix PRESET=$(PRESET)

check:
	$(MAKE) test PRESET=$(PRESET)
	$(MAKE) format-check PRESET=$(PRESET)
	$(MAKE) tidy PRESET=$(PRESET)

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

git-configure: require-clean
	cmake --preset $(PRESET) -B $(GIT_BUILD_DIR)

git-build: git-configure
	cmake --build $(GIT_BUILD_DIR)

git-test: git-build
	ctest --test-dir $(GIT_BUILD_DIR) --output-on-failure
