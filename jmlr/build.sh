#!/bin/sh
# Build main.pdf and supplement.pdf. They cross-reference each other through
# xr-hyper (supplement labels carry the prefix S), so the order matters:
# main, supplement, then main again.
set -e
T=${TECTONIC:-tectonic}
$T -X compile --keep-intermediates --keep-logs main.tex
$T -X compile --keep-intermediates --keep-logs supplement.tex
$T -X compile --keep-intermediates --keep-logs main.tex
