.PHONY: clean clean-pyc clean-build clean-test lint test_environment
	environment update_environment remove_environment src_env src_update
	src_remove

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = GeoAI2025
PYTHON_INTERPRETER = python3
ENVIRONMENT_FILE = environment.yml
ENVIRONMENT_NAME = geoai

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

##############################################################################
# VARIABLES FOR COMMANDS                                                     #
##############################################################################
src_pip_install:=pip install -e .

src_pip_uninstall:= pip uninstall --yes src

cosmo_utils_pip_install:=pip install cosmo-utils

cosmo_utils_pip_upgrade:= pip install --upgrade cosmo-utils

cosmo_utils_pip_uninstall:= pip uninstall cosmo-utils

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Deletes all build, test, coverage, and Python artifacts
clean: clean-build clean-pyc clean-test

## Removes Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove test and coverage artifacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Set up python interpreter environment - Using environment.yml
environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		# conda config --add channels conda-forge
		conda env create -f $(ENVIRONMENT_FILE)
endif
	$(src_pip_install)

## Update python interpreter environment
update_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env update -f $(ENVIRONMENT_FILE)
		$(cosmo_utils_pip_upgrade)
endif
	$(src_pip_uninstall)
	$(src_pip_install)

## Delete python interpreter environment
remove_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, removing conda environment"
		conda env remove -n $(ENVIRONMENT_NAME)
endif

## Import local source directory package
src_env:
	$(src_pip_install)

## Updated local source directory package
src_update:
	$(src_pip_uninstall)
	$(src_pip_install)

## Remove local source directory package
src_remove:
	$(src_pip_uninstall)

## Installing cosmo-utils
cosmo_utils_install:
	$(cosmo_utils_pip_install)

## Upgrading cosmo-utils
cosmo_utils_upgrade:
	$(cosmo_utils_pip_upgrade)

## Removing cosmo-utils
cosmo_utils_remove:
	$(cosmo_utils_pip_uninstall)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
