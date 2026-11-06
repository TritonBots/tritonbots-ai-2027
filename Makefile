.PHONY: proto test lint fmt tools clean

proto:
	./scripts/gen_proto.sh

test: proto
	pytest -q

# mypy covers core, the backends, the motion planner and GoToPoint, and the
# env base -- not yet all of src.
# The backends are in scope because that is where the Backend / SimBackend
# split is enforced (TASK-072): if CI does not type-check them, "you cannot
# point a training run at hardware" is a comment, not a guarantee.
lint:
	ruff check src tests
	mypy src/tbots/core src/tbots/backends src/tbots/motion src/tbots/skills/go_to_point.py src/tbots/rl/envs/base.py

fmt:
	ruff format src tests
	ruff check --fix src tests

tools:
	./scripts/fetch_tools.sh

clean:
	rm -rf src/tbots/_pb .pytest_cache .ruff_cache
