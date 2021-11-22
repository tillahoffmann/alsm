.PHONY : analysis clean data data/addhealth docs lint sync tests

build : lint tests docs

NOTEBOOK_FLAKE8_TARGETS = $(addsuffix .flake8,$(wildcard scripts/*.ipynb))

lint : ${NOTEBOOK_FLAKE8_TARGETS}
	flake8 --exclude playground

${NOTEBOOK_FLAKE8_TARGETS} : %.flake8 : %
	jupyter nbconvert --to python --stdout $< | flake8 - --stdin-display-name=$< --ignore=W391 \
		--show-source

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

ANALYSIS_TARGETS = workspace/simulation.html workspace/addhealth.html workspace/theory.html

analysis : ${ANALYSIS_TARGETS}

${ANALYSIS_TARGETS} : workspace/%.html : scripts/%.ipynb
	jupyter nbconvert --execute --to=html --output-dir=$(dir $@) --output=$(notdir $@) $<

ADDHEALTH_FILES = comm72.dat comm72_att.dat
ADDHEALTH_TARGETS = $(addprefix data/addhealth/,${ADDHEALTH_FILES})

${ADDHEALTH_TARGETS} : data/addhealth/% :
	mkdir -p $(dir $@)
	curl -L -o $@ https://web.archive.org/web/0if_/http://moreno.ss.uci.edu/$(notdir $@)

data/addhealth : ${ADDHEALTH_TARGETS}

data : data/addhealth

clean :
	rm -rf docs/_build workspace

clear_output :
	jupyter nbconvert --clear-output --Exporter.preprocessors=whitespace_remover.WhitespaceRemover \
		scripts/*.ipynb
