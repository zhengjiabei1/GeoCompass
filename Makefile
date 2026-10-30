.PHONY: run report test

run:
	python -m geoskillevo experiment --seeds 0,1,2 --output experiments/geoskillevo

report:
	python -m geoskillevo report

test:
	python -m unittest discover -s tests -v
