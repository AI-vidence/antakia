#!/bin/sh
echo $0

if [ "$0" == "" ]
then
   echo 'Please provide version update type (patch/minor/major)'
   exit 1
else
   echo ''
fi

cd ../antakia-core
atkc=$(echo $(poetry version) | awk '{print $2}')
cd ../antakia
atk=$(echo $(poetry version) | awk '{print $2}')

if [ "$atkc" != "$atk" ]
then
   echo 'version do not match'
   exit 1
else
   echo 'same version'
fi

cd ../antakia-core
git stash
git checkout -f dev
git pull
new_v=$(poetry version $1)
new_v=$(echo $new_v | awk '{print $6}')
git add pyproject.toml
git commit -m 'version increased'
git stash pop

cd ../antakia
git stash
git checkout -f dev
git pull
new_v=$(poetry version $1)
new_v=$(echo $new_v | awk '{print $6}')
poetry run python scripts/release_scripts/version_sync.py
git add pyproject.toml
git commit -m 'version increased'
git stash pop
