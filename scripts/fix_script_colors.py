# DEPRECATED: phrase-based color override helper
# This script was used during debugging to apply phrase-level colors from
# the human-labeled `data/color_training/*p1.right.phrases.txt` files to
# the auto-extracted `script_*.json` outputs. Relying on phrase-based
# mapping is not acceptable for production or extension across PDFs
# because it uses ground-truth labels and not the document's span/glyph
# evidence. The script is intentionally disabled.

import sys
print('fix_script_colors.py is deprecated and should not be used.')
print('Use span-driven extraction and convolution fallback instead.')
sys.exit(0)

