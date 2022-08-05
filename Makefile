.ONESHELL:

.PHONY: prepare_venv
prepare_venv:
	conda create --name hw1-venv python=3.10
	conda activate hw1-venv

.PHONY: install
install:
	conda install pip
	pip install -r requirements.txt

.PHONY: prepare_data
prepare_data:
	cd raw_data/
	7za x train-jpg.tar.7z && tar -xvkf train-jpg.tar && rm train-jpg.tar train-jpg.tar.7z
	7za x test-jpg.tar.7z && tar -xvkf test-jpg.tar && rm test-jpg.tar test-jpg.tar.7z
	unzip train_v2.csv.zip && rm train_v2.csv.zip