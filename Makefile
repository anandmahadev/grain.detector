.PHONY: help install run train test docker-build docker-run clean

help:
	@echo "Usage:"
	@echo "  make install      Install dependencies"
	@echo "  make run          Run the Streamlit app"
	@echo "  make train        Run the YOLO training script"
	@echo "  make test         Run pytest suite"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo "  make clean        Remove temporary files"

install:
	pip install -r requirements.txt

run:
	streamlit run grain_counter.py

train:
	python train_custom_yolo.py

test:
	pytest tests/

docker-build:
	docker build -t grain-detector .

docker-run:
	docker run -p 8501:8501 grain-detector

clean:
	rm -rf __pycache__ .pytest_cache runs/
