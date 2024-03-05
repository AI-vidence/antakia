#!/bin/sh
if [ "$1" == "" ]
then
   release=patch
else
   release=$1
fi

$(./scripts/release_scripts/version_update.sh $release)
$(./scripts/release_scripts/publish.sh)