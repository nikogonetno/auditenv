PY ?= python3
N ?= 300
# The four jury baselines and the diagnostic spray, as in the committed docs/METRICS.md.
METRICS_POLICIES := random,reconcile,flag_all,oracle,spray
IMAGE ?= auditenv:dev
GOLDEN := tests/fixtures/golden_v0.1/episode

.PHONY: verify test qa demo stress episodes metrics paired docker-build docker-verify clean

# The one command. Same script is the default CMD of the Docker image.
verify:
	PY=$(PY) sh scripts/verify.sh

test:
	$(PY) -m unittest discover -s tests -t . -v

qa:
	$(PY) -I qa/qa_golden.py $(GOLDEN)

# Demo (case §09): the four baseline policies in eval mode on the case §03 test book with 8 errors.
# Writes runs/trajectories.jsonl and runs/grader_log.jsonl; prints one summary line per policy.
DEMO_EPISODE := tests/fixtures/mini_v03

demo:
	rm -rf runs/trajectories.jsonl runs/grader_log.jsonl
	mkdir -p runs
	for policy in random reconcile flag_all oracle; do \
		$(PY) -m auditenv.run --policy $$policy --mode eval --seed 0 --episode-dir $(DEMO_EPISODE) \
			--trajectories runs/trajectories.jsonl --grader-log runs/grader_log.jsonl || exit 1; \
	done

# Generator stress test (docs/tasks/B_next.md item 7): N seeds, verifier, oracle 1.0, 60 s, determinism.
stress:
	$(PY) scripts/stress.py --n $(N)

# Open train and dev episodes (case §09) from configs/train_v1.json and configs/dev_v1.json.
episodes:
	$(PY) scripts/build_episodes.py train
	$(PY) scripts/build_episodes.py dev

# Diagnostics of the four policies on train and dev (case §10): docs/METRICS.md, raw rows, train trajectories.
metrics:
	$(PY) scripts/metrics.py episodes/train episodes/dev --policies $(METRICS_POLICIES) --out docs/METRICS.md --json runs/metrics.json

# 20 paired checks (case §10): repeat seed and actions; permutation and new ids once A's transform lands.
paired:
	$(PY) scripts/paired.py episodes/dev --n 20

docker-build:
	docker build -t $(IMAGE) .

docker-verify: docker-build
	docker run --rm --network none $(IMAGE)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
