install:
	pip install --upgrade pip &&\
		pip install -r Requirements.txt
		
format:
	black *.py
	
lint:
	pylint --disable=R,C spaceship_titanic.py
	
test:
	python -m pytest -vv --cov=spaceship_titanic teat_titanic.py

all: install format lint