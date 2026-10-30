.PHONY: run report test

run:
	python -m geoskillcompass experiment --seeds 0,1,2 --output experiments/geoskillcompass

report:
	python -m geoskillcompass report

test:
	python -m unittest discover -s tests -v
