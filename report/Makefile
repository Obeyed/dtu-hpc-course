root = report
latexmkopts = -xelatex -shell-escape
date = $(shell date +%d%m%y-%H:%M)

all: $(root).pdf
.PHONY: $(root).pdf 

$(root).pdf: $(root).tex
	latexmk $(latexmkopts) $(root)

clean:
	rm -rf `biber --cache`
	rm -rf _minted-$(root)
	latexmk -bibtex -CA

watch: clean
	latexmk $(latexmkopts) -pvc $(root)

quick: clean
	xelatex -shell-escape $(root)

view:
	open $(root).pdf
