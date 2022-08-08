.PHONY: install
install:
	pip install -r requirements.txt


.PHONY: prepare_data
prepare_data:
	cd raw_data/
	7za x train-jpg.tar.7z && tar -xvkf train-jpg.tar && rm train-jpg.tar train-jpg.tar.7z
	7za x test-jpg.tar.7z && tar -xvkf test-jpg.tar && rm test-jpg.tar test-jpg.tar.7z
	unzip train_v2.csv.zip && rm train_v2.csv.zip


.PHONY: lint
lint:
	flake8 src/


.PHONY: run_unit_tests
run_unit_tests:
	PYTHONPATH=. pytest src/tests/unit/


.PHONY: run_integration_tests
run_integration_tests:
	PYTHONPATH=. pytest -s tests/integration/


.PHONY: run_all_tests
run_all_tests:
	make run_unit_tests
	make run_integration_tests
