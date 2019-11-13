NAME = thesis
#NAME = siam
COVERNAME = cover
LIB_NAME = R05922082 
DEPT_NAME = R05922082周弘逸

FILES = $(NAME).pdf
COVERFILE = $(COVERNAME).pdf
LIB_FILE = $(LIB_NAME).pdf
DEPT_FILE = $(DEPT_NAME).pdf

TEX_DEPS = *.tex *.sty
CLEAN_FILES = *.cjk $(LIB_FILE) $(DEPT_FILE) $(COVERFILE) $(FILES) $(NAME).run.xml

all:	clean pre-build $(FILES)

pre-build:
	if [ ! -f sdp.bib ]; then \
		if [ ! -f ../sdp.bib ]; then \
			scp r05922038@linux1:~cjlin/latex/sdp.bib ../; \
		fi; \
		ln -s ../sdp.bib .; \
	fi

# convenient targets

# generate side for hard-copy thesis
side:
	$(LATEX) side.tex
	dvipdfmx -l side.dvi
	#cp side.pdf tmpside.pdf
	#pdftk tmpside.pdf cat 1E output side.pdf

cover:
	sed -e "/ntuwatermark/d" $(NAME).tex > tmpcover.tex
	make NAME=tmpcover all
	pdftk tmpcover.pdf cat 1 output - > $(COVERFILE)

library-version: $(LIB_FILE)

$(LIB_FILE): $(FILES)
	pdftk $< output $(LIB_FILE) owner_pw PROMPT encrypt_128bit allow Printing
	cp $(LIB_FILE) $(DEPT_FILE)

# slide needs cross reference, so make sure the main file
# is built before we build slides
slide:	do-slide
do-slide: $(FILES)
	@cd ../slide && make

# tell you the start page # of each chapter
# the chapters must be labeled as chap:xxx
show-chapters:
	@grep '^\\newlabel{chap:' *.aux | \
		sed 's,^\\newlabel{\([^}]*\)}{{[^}]*}{\([^}]*\)}.*},\1 \2,'

# BSD make(1) works, too
include Makefile.inc

