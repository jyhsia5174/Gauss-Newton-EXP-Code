FILES = algo5-illu-iter.pdf algo5-illu-greedy.pdf algo5-illu-algo5.pdf 
TEX_DEPS += *.tex
CLEAN_FILES += *.log *.aux *tmp*.dat *tmp*.script *tmp*.table
PDFLATEX = pdflatex --shell-escape

all: $(FILES)

clean:
	rm -f $(CLEAN_FILES)

$(FILES): $(TEX_DEPS)

.SUFFIXES: .tex .dvi .ps .pdf .jpg .eps

.tex.pdf: $(TEX_DEPS)
	-${PDFLATEX} $*.tex
