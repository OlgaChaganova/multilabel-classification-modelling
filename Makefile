.PHONY: install
install:
	pip install -r requirements.txt
	pip install .
	pip install -U rich


.PHONY: prepare_data
prepare_data:
	cd raw_data/
	7za x train-jpg.tar.7z && tar -xvkf train-jpg.tar && rm train-jpg.tar train-jpg.tar.7z
	7za x test-jpg.tar.7z && tar -xvkf test-jpg.tar && rm test-jpg.tar test-jpg.tar.7z
	unzip train_v2.csv.zip && rm train_v2.csv.zip


.PHONY: help
help:
	python src/train_model.py --help

.PHONY: lint
lint:
	PYTHONPATH=. flake8 src/


.PHONY: run_unit_tests
run_unit_tests:
	PYTHONPATH=. pytest tests/unit/


.PHONY: run_integration_tests
run_integration_tests:
	PYTHONPATH=. pytest tests/integration/


.PHONY: run_tests
run_tests:
	make run_unit_tests
	make run_integration_tests


.PHONY: generate_coverage_report
generate_coverage_report:
	PYTHONPATH=. pytest --cov=src --cov-report html  tests/
