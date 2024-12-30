#!/bin/bash

text_files="aggregated_f7KSfjv4Oq0.txt  aggregated_MUWUHf-rzks.txt  aggregated_QImCld9YubE.txt  aggregated_QOCaacO8wus.txt  aggregated_TYPFenJQciw.txt"

### ### ###

external_dir=../..
rootdir=$external_dir/graphragtranscripts
inpdir=$rootdir/input
mkdir -p $inpdir

for txtf in $text_files
do
cp $txtf $inpdir/$txtf
done

python -m graphrag.index --init --root $rootdir

openai_api_key=$(cat ../../LM_Tests/openai_api_key.txt)
echo \
'

GRAPHRAG_API_KEY='$openai_api_key > $rootdir/.env

python -m graphrag.index --root $rootdir

