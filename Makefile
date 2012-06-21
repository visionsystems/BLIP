VERSION = 1.0
TARFILE = simplesim-$(VERSION).tar.gz
PYTHON = python # pypy is much faster

all: tar

tar: clean
	tar -cvz --exclude=".hg*" --exclude=".git*" --exclude=$(TARFILE) -f $(TARFILE) *

# mercurial (hg) version control targets

co:
	echo "hg pull"
	echo "hg merge"

ci:
	echo" #hg add -I* -I*/* -I*/*/*"
	echo" hg commit"
	echo" hg push"

st:
	hg stat

# the following targets are just playthings -- you must read README.txt

ref:
	time -p $(PYTHON) violajones/reference.py
	-display violajones_out.png

try: default testit run

run:
	time -p $(PYTHON) violajones/gen_code.py
	
default:
	$(PYTHON) blip/simulator/interpreter.py
	
testit:
	cd test; $(PYTHON) test_all.py

planarity-filters:
	time -p $(PYTHON) planarity/gen_code.py
	display blip_detections.png
	
regress:
	$(PYTHON) violajones/reference.py | grep -v progress | tee run.stdout
	diff run.stdout regr-data/run-ref.stdout
	diff violajones_out.png regr-dat/aviolajones_out-ref.png

doc:
	make -C docs html

docsee:
	firefox docs/_build/html/index.html

clean:
	@rm -f *.aux *.log *.bbl *.blg *.bib *.bak
