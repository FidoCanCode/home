.PHONY: models

# Extract Rocq models to Python and deposit in kennel/models_generated/.
# Requires: docker with buildx.
models:
	./build
