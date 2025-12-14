#!/usr/bin/env bash

wget -N https://dumps.wikimedia.org/huwiki/latest/huwiki-latest-pages-articles.xml.bz2
bzip2 -d -k -f huwiki-latest-pages-articles.xml.bz2
