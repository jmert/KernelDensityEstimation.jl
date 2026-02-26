.PHONY: serve build instantiate test fix

DOCARGS :=
JULIA ?= julia
mkfile_path := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

serve: instantiate
	cd $(realpath $(mkfile_path)..) && \
	$(JULIA) --startup-file=no --project=docs/ \
		-e 'using LiveServer; LiveServer.servedocs()'

build: instantiate
	cd $(realpath $(mkfile_path)..) && \
	$(JULIA) --startup-file=no --project=docs/ \
		docs/make.jl ${DOCARGS}

instantiate:
	cd $(realpath $(mkfile_path)..) && \
	$(JULIA) --startup-file=no --project=docs/ \
		-e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'

test: DOCARGS+= --test
test: build
	@true

fix: DOCARGS+= --fix
fix: build
	@true
