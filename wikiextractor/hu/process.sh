#!/usr/bin/env bash

# ../conda-profiles/wikiextractor/create-conda-env-wikiextractor.sh

SCRIPT_DIR=$(dirname "$0")

rm -rf "${SCRIPT_DIR}/huwiki_extracted"

# conda run -n wikiextractor wikiextractor "${SCRIPT_DIR}/huwiki-latest-pages-articles.xml" -o "${SCRIPT_DIR}/huwiki_extracted"
# conda run -n wikiextractor wikiextractor "${SCRIPT_DIR}/huwiki-latest-pages-articles.xml" -o "${SCRIPT_DIR}/huwiki_extracted" -b 2M -ns ns0,ns4,ns14,ns118,ns100 --json
conda run -n wikiextractor wikiextractor "${SCRIPT_DIR}/huwiki-latest-pages-articles.xml" -o "${SCRIPT_DIR}/huwiki_extracted" -b 0 -ns ns0,ns4,ns14,ns118,ns100 --json