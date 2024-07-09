.PHONY : analysis clean data data/addhealth lint sync tests workspace/prior-sensitivity workspace/validation

build : lint tests

NOTEBOOK_FLAKE8_TARGETS = $(addsuffix .flake8,$(wildcard scripts/*.ipynb))
NB_EXECUTE = jupyter nbconvert --execute --to=html --output-dir=$(dir $@) --output=$(notdir $@) $<

lint : ${NOTEBOOK_FLAKE8_TARGETS}
	flake8 --exclude playground,build
	black --check .

${NOTEBOOK_FLAKE8_TARGETS} : %.flake8 : %
	jupyter nbconvert --to python --stdout $< | flake8 - --stdin-display-name=$< --ignore=W391 \
		--show-source

tests :
	pytest -v --cov=alsm --cov-fail-under=100 --cov-report=term-missing --cov-report=html

sync : requirements.txt
	pip-sync

requirements.txt : requirements.in setup.py
	pip-compile -v -o $@ $<

NOTEBOOKS = scripts/simulation.md scripts/theory.md scripts/mode-separation-demo.md \
	scripts/validation.md scripts/validation-statistics.md scripts/addhealth-fit.md \
	scripts/addhealth-plot.md
IPYNBS = ${NOTEBOOKS:.md=.ipynb}
ANALYSIS_TARGETS = $(addprefix workspace/,$(notdir ${NOTEBOOKS:.md=.html}))

ipynb : ${IPYNBS}
${IPYNBS} : scripts/%.ipynb : scripts/%.md
	jupytext --output $@ $<

analysis : workspace/prior-sensitivity.html workspace/validation-statistics.html \
	workspace/addhealth-plot.html workspace/simulation.html workspace/theory.html \
	scripts/mode-separation-demo.html

ADDHEALTH_FILES = comm72.dat comm72_att.dat
ADDHEALTH_TARGETS = $(addprefix data/addhealth/,${ADDHEALTH_FILES})

${ADDHEALTH_TARGETS} : data/addhealth/% :
	mkdir -p $(dir $@)
	curl -L -o $@ https://web.archive.org/web/0if_/http://moreno.ss.uci.edu/$(notdir $@)

data/addhealth : ${ADDHEALTH_TARGETS}

data : data/addhealth

workspace/prior-sensitivity : workspace/simulation-cauchy-1.html \
	workspace/simulation-cauchy-5.html workspace/simulation-normal-1.html \
	workspace/simulation-normal-5.html workspace/simulation-exponential-1.html \
	workspace/simulation-exponential-5.html

workspace/simulation-cauchy-1.html : scripts/simulation.ipynb
	SCALE_PRIOR_TYPE=cauchy SCALE_PRIOR_SCALE=1 OUTPUT=`pwd`/${@:.html=.pkl} ${NB_EXECUTE}

workspace/simulation-cauchy-5.html : scripts/simulation.ipynb
	SCALE_PRIOR_TYPE=cauchy SCALE_PRIOR_SCALE=5 OUTPUT=`pwd`/${@:.html=.pkl} ${NB_EXECUTE}

workspace/simulation-normal-1.html : scripts/simulation.ipynb
	SCALE_PRIOR_TYPE=normal SCALE_PRIOR_SCALE=1 OUTPUT=`pwd`/${@:.html=.pkl} ${NB_EXECUTE}

workspace/simulation-normal-5.html : scripts/simulation.ipynb
	SCALE_PRIOR_TYPE=normal SCALE_PRIOR_SCALE=5 OUTPUT=`pwd`/${@:.html=.pkl} ${NB_EXECUTE}

workspace/simulation-exponential-1.html : scripts/simulation.ipynb
	SCALE_PRIOR_TYPE=exponential SCALE_PRIOR_SCALE=1 OUTPUT=`pwd`/${@:.html=.pkl} ${NB_EXECUTE}

workspace/simulation-exponential-5.html : scripts/simulation.ipynb
	SCALE_PRIOR_TYPE=exponential SCALE_PRIOR_SCALE=5 OUTPUT=`pwd`/${@:.html=.pkl} ${NB_EXECUTE}

ifeq ($(CI),)
  VALIDATION_TARGETS = $(addsuffix .html,$(addprefix workspace/validation-,$(shell seq 100)))
else
  VALIDATION_TARGETS = $(addsuffix .html,$(addprefix workspace/validation-,$(shell seq 2)))
endif

workspace/validation : ${VALIDATION_TARGETS}

${VALIDATION_TARGETS} : workspace/validation-%.html : scripts/validation.ipynb
	SEED=$* OUTPUT=`pwd`/${@:.html=.pkl} ${NB_EXECUTE}

workspace/validation-statistics.html : scripts/validation-statistics.ipynb ${VALIDATION_TARGETS}
	${NB_EXECUTE}

workspace/addhealth-fit.html : scripts/addhealth-fit.ipynb data/addhealth
	${NB_EXECUTE}

workspace/addhealth-plot.html : scripts/addhealth-plot.ipynb workspace/addhealth-fit.html
	${NB_EXECUTE}

workspace/simulation.html : scripts/simulation.ipynb
	${NB_EXECUTE}

workspace/theory.html : scripts/theory.ipynb
	${NB_EXECUTE}

workspace/mode-separation-demo.html : scripts/mode-separation-demo.ipynb
	${NB_EXECUTE}
