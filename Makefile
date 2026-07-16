.PHONY: install smoke test pipeline demo

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .

test:
	. .venv/bin/activate && pytest -q

smoke:
	. .venv/bin/activate && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
		python scripts/run_pipeline.py --config configs/smoke.yaml run-all

pipeline:
	. .venv/bin/activate && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
		python scripts/run_pipeline.py --config configs/default.yaml run-all

demo:
	. .venv/bin/activate && KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
		python scripts/run_pipeline.py --config configs/smoke.yaml demo
