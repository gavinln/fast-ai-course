SCRIPT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: help
.DEFAULT_GOAL=help
help:  ## help for this Makefile
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: jupyter-lab
jupyter:  ## start jupyter lab
	poetry run jupyter lab --ip=127.0.0.1 --no-browser --notebook-dir=$(SCRIPT_DIR)/

.PHONY: to_py
to_py:  ## convert notebook to python: make to_py fl=f.ipynb
	@test -n "$(fl)" || { echo "fl= not specified"; exit 1; }
	@test -f "$(fl)" || { echo "Notebook $(fl) does not exist"; exit 1; }
	poetry run jupytext --to py "$(fl)"

.PHONY: nvim
nvim:  ## open nvim -S Session.vim
	poetry run nvim -S Session.vim
