# Build the paper PDF
# Requires: pdflatex + biber (install with: brew install --cask mactex)

PDF = main.pdf
SRC = main.tex

.PHONY: all clean watch

all: $(PDF)

$(PDF): $(SRC) references.bib sections/*.tex
	/Library/TeX/texbin/pdflatex -interaction=nonstopmode main
	/Library/TeX/texbin/biber main
	/Library/TeX/texbin/pdflatex -interaction=nonstopmode main
	/Library/TeX/texbin/pdflatex -interaction=nonstopmode main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz $(PDF)

# Auto-rebuild on file change (requires: brew install fswatch)
watch:
	fswatch -o $(SRC) sections/*.tex references.bib | xargs -n1 -I{} make all
