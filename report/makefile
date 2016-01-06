XELATEX=xelatex -file-line-error -interaction=nonstopmode

.PHONY: all
all: report.tex
	latexmk -pdf -pdflatex="$(XELATEX)" -use-make report.tex

.PHONY: auto
auto: report.tex
	latexmk -pdf -pdflatex="$(XELATEX)" -use-make -pvc report.tex

.PHONY: clean
clean:
	rm -f report.aux *.bbl *bcf *.blg *.log *.out *.tdo *.toc *.xdv *.synctex.gz *.fdb_latexmk *.fls *.run.xml; \
	rm -f **/*.aux
