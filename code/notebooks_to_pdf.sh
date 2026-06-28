#!/bin/bash
#
# notebooks_to_pdf.sh — convert all 16 thesis notebooks to PDF.
#
# Pipeline: jupyter nbconvert (notebook → HTML) → Chrome headless (HTML → PDF).
# No LaTeX or pandoc required. Outputs go to ml_dataset/notebooks_appendix_pdf/.
#
# Re-run after any notebook is modified or re-executed; the PDFs reflect the
# latest cell outputs that are saved inside the .ipynb file.
#
# Usage:
#   bash ml_dataset/scripts/notebooks_to_pdf.sh           # convert everything
#   bash ml_dataset/scripts/notebooks_to_pdf.sh --force   # overwrite existing PDFs
#
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT="/Users/davidtoma/Library/CloudStorage/GoogleDrive-tomadavid001@gmail.com/My Drive/MSC THESIS"
NB_V6_DIR="$PROJECT_ROOT/ml_dataset/data/model_ready/movie_success_v6"
NB_V7_DIR="$PROJECT_ROOT/ml_dataset/data/model_ready/movie_success_v7"
OUT_DIR="$PROJECT_ROOT/ml_dataset/notebooks_appendix_pdf"
TMP_HTML_DIR="/tmp/nb_html"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ─── Flags ────────────────────────────────────────────────────────────────
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# ─── Prepare ──────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR" "$TMP_HTML_DIR"
echo "Output: $OUT_DIR"
echo ""

# ─── Collect notebooks (v6 numbered + v6 tables + v7 DL) ──────────────────
declare -a NBS
for nb in "$NB_V6_DIR"/0*.ipynb "$NB_V6_DIR"/1*.ipynb "$NB_V6_DIR"/thesis_tables*.ipynb; do
    [[ -f "$nb" ]] && NBS+=("$nb|")
done
for nb in "$NB_V7_DIR"/01_deep_learning.ipynb; do
    [[ -f "$nb" ]] && NBS+=("$nb|v7_")
done

# ─── Convert ──────────────────────────────────────────────────────────────
echo "=== Step 1 — notebook → HTML (jupyter nbconvert) ==="
for entry in "${NBS[@]}"; do
    nb="${entry%|*}"; prefix="${entry##*|}"
    base="${prefix}$(basename "$nb" .ipynb)"
    printf "  %-45s  " "$base.ipynb"
    jupyter nbconvert --to html --output-dir="$TMP_HTML_DIR" \
        --output="$base" "$nb" 2>&1 | tail -1
done

echo ""
echo "=== Step 2 — HTML → PDF (Chrome headless) ==="
for entry in "${NBS[@]}"; do
    nb="${entry%|*}"; prefix="${entry##*|}"
    base="${prefix}$(basename "$nb" .ipynb)"
    html="$TMP_HTML_DIR/$base.html"
    pdf="$OUT_DIR/$base.pdf"

    if [[ -f "$pdf" && $FORCE -eq 0 ]]; then
        size=$(stat -f%z "$pdf")
        printf "  %-45s  SKIP (exists, %d bytes; use --force to overwrite)\n" "$base.pdf" "$size"
        continue
    fi

    printf "  %-45s  " "$base.pdf"
    "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
        --print-to-pdf="$pdf" "file://$html" 2>&1 | tail -1
done

echo ""
echo "=== Done ==="
ls -la "$OUT_DIR/"*.pdf | wc -l | xargs echo "PDFs produced:"
du -sh "$OUT_DIR"
