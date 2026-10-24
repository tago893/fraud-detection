.PHONY: split train evaluate api simulate dashboard test readme
split:
	python -m pipeline.split
train:
	python -m pipeline.train
evaluate:
	python -m pipeline.evaluate
api:
	uvicorn api.main:app --port 8000
simulate:
	python -m simulator.stream_txns --rate 10
dashboard:
	streamlit run dashboard/app.py
test:
	pytest -q
readme:
	python -m pipeline.readme_table
