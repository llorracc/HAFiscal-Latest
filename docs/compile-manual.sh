#!/bin/bash
echo "=== Manual Cross-Reference Compilation ===" ; echo "Step 1: HAFiscal.tex" ; latexmk -f HAFiscal.tex ; echo "Step 2: HAFiscal-online-appendix.tex" ; latexmk -f HAFiscal-online-appendix.tex ; echo "Step 3: HAFiscal.tex (final)" ; latexmk -f HAFiscal.tex ; echo "✅ Done!"
