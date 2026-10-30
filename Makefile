.PHONY: run report test

run:
	python -m geocompass experiment --seeds 0,1,2 --output experiments/geocompass

report:
	python -m geocompass report

test:
	python -m unittest discover -s tests -v
