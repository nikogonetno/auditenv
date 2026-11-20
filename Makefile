PY ?= python3
IMAGE ?= auditenv:dev
GOLDEN := tests/fixtures/golden_v0.1/episode

.PHONY: verify test qa demo docker-build docker-verify clean

# The one command. Same script is the default CMD of the Docker image.
verify:
	PY=$(PY) sh scripts/verify.sh

test:
	$(PY) -m unittest discover -s tests -t . -v

qa:
	$(PY) -I qa/qa_golden.py $(GOLDEN)

# Demo (case §09): the three baseline policies in eval mode on the case §03 test book with 8 errors.
# Writes runs/trajectories.jsonl and runs/grader_log.jsonl; prints one summary line per policy.
DEMO_EPISODE := tests/fixtures/mini_v03

demo:
	rm -rf runs/trajectories.jsonl runs/grader_log.jsonl
	mkdir -p runs
	for policy in random flag_all oracle; do \
		$(PY) -m auditenv.run --policy $$policy --mode eval --seed 0 --episode-dir $(DEMO_EPISODE) \
			--trajectories runs/trajectories.jsonl --grader-log runs/grader_log.jsonl || exit 1; \
	done

docker-build:
	docker build -t $(IMAGE) .

docker-verify: docker-build
	docker run --rm --network none $(IMAGE)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
