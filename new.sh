#!/bin/bash
# Usage: ./new.sh <section> <slug>
# Example: ./new.sh ideas dense-reward-shaping
#          ./new.sh paper-notes attention-is-all-you-need
#          ./new.sh notes gpu-profiling-results

set -e

SECTION="${1:?Usage: ./new.sh <ideas|paper-notes|notes> <slug>}"
SLUG="${2:?Usage: ./new.sh <ideas|paper-notes|notes> <slug>}"
DATE=$(date +%Y-%m-%d)
DIR="${SECTION}/${DATE}-${SLUG}"

if [[ ! -f "${SECTION}/_template.qmd" ]]; then
  echo "Error: unknown section '${SECTION}'. Use: ideas, paper-notes, or notes"
  exit 1
fi

if [[ -d "${DIR}" ]]; then
  echo "Error: ${DIR} already exists"
  exit 1
fi

mkdir -p "${DIR}"
cp "${SECTION}/_template.qmd" "${DIR}/index.qmd"

# Pre-fill the date
sed -i "s/^date: \"\"$/date: \"${DATE}\"/" "${DIR}/index.qmd"

echo "Created ${DIR}/index.qmd"
echo "Open it and fill in the title, description, and content."
