FILES = fig1.pdf fig2.pdf fig3.pdf fig4.pdf fig5.pdf
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
