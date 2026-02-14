PYTHON ?= python3

.PHONY: setup data features causal predict text dashboard report all

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.cli data

features:
	$(PYTHON) -m src.cli features

causal:
	$(PYTHON) -m src.cli causal

predict:
	$(PYTHON) -m src.cli predict

text:
	$(PYTHON) -m src.cli text

dashboard:
	$(PYTHON) -m src.cli dashboard

report:
	$(PYTHON) -m src.cli report

all:
	$(PYTHON) -m src.cli all
