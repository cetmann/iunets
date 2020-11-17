# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
OUTPUTDIR     = .
SOURCEDIR     = ../iunets
BUILDDIR      = _build
INSTALLDIR    = ../../docs.iunets

EXCLUDE = ../iunets/experiments
EXCLUDE += $(shell git ls-files -z --others --exclude-standard ../dival | xargs -0)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	@sphinx-apidoc -f -e -o "$(OUTPUTDIR)" "$(SOURCEDIR)" $(EXCLUDE)

install:
	@cp -r "$(BUILDDIR)/html/." "$(INSTALLDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(OUTPUTDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)