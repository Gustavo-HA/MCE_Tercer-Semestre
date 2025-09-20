#!/bin/bash

# Build script for R Sweave documents
# Usage: ./build.sh filename.Rtex

if [ $# -eq 0 ]; then
    echo "Usage: $0 filename.Rtex"
    exit 1
fi

FILENAME="$1"
BASENAME="${FILENAME%.Rtex}"

echo "Processing R Sweave document: $FILENAME"

# Step 1: Process R chunks with knitr
echo "Step 1: Processing R chunks..."
Rscript -e "
library(knitr)
opts_knit\$set(concordance = TRUE)
knit('$FILENAME')
"

if [ $? -ne 0 ]; then
    echo "Error: R processing failed"
    exit 1
fi

# Step 2: Compile LaTeX
echo "Step 2: Compiling LaTeX..."
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf "$BASENAME.tex"

if [ $? -eq 0 ]; then
    echo "Build successful! PDF generated: $BASENAME.pdf"
else
    echo "Error: LaTeX compilation failed"
    exit 1
fi