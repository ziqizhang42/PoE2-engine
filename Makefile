.DEFAULT_GOAL := check

PRESET ?= debug

.PHONY: configure build test format format-check tidy tidy-fix fix check clean debug release

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
