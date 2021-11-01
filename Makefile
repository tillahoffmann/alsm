.PHONY : clean docs lint sync tests

NBEXEC = jupyter nbconvert --execute --to=html

build : lint tests docs

lint :
	flake8

tests :
	pytest -v --cov=alsm --cov-fail-under=100 --cov-report=term-missing --cov-report=html

docs :
	sphinx-build . docs/_build

sync : requirements.txt
	pip-sync

requirements.txt : requirements.in setup.py test_requirements.txt
	pip-compile -v -o $@ $<

test_requirements.txt : test_requirements.in setup.py
	pip-compile -v -o $@ $<

workspace/simulation.html : scripts/simulation.ipynb
	${NBEXEC} --output-dir=$(dir $@) --output=$(notdir $@) $<

workspace/group_simulation.html : scripts/simulation.ipynb
	GROUP_SIMULATION=1 ${NBEXEC} --output-dir=$(dir $@) --output=$(notdir $@) $<

clean :
	rm -rf docs/_build
