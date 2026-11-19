PY ?= python3
IMAGE ?= auditenv:dev
GOLDEN := tests/fixtures/golden_v0.1/episode

.PHONY: verify test qa docker-build docker-verify clean

# The one command. Same script is the default CMD of the Docker image.
verify:
	PY=$(PY) sh scripts/verify.sh

test:
	$(PY) -m unittest discover -s tests -t . -v

qa:
	$(PY) -I qa/qa_golden.py $(GOLDEN)

docker-build:
	docker build -t $(IMAGE) .

docker-verify: docker-build
	docker run --rm --network none $(IMAGE)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
