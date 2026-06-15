if [ $# -eq 0 ]
then
   release=patch
else
   release=$1
fi

cd ../antakia-core
git stash
atkc_branch=$(git rev-parse --abbrev-ref HEAD)
git checkout -f dev
git pull
atkc=$(echo $(poetry version) | awk '{print $2}')

cd ../antakia
git stash
atk_branch=$(git rev-parse --abbrev-ref HEAD)
git checkout -f dev
git pull
atk=$(echo $(poetry version) | awk '{print $2}')

if [ "$atkc" != "$atk" ]
then
   echo 'version do not match'
   exit 1
else
   echo 'same version'
fi

cd ../antakia-core
new_v=$(poetry version release)
new_v=$(echo $new_v | awk '{print $6}')
git add pyproject.toml
git commit -m 'version increased'
git checkout $atkc_branch
git stash pop
echo increased antakia-core version

cd ../antakia
new_v=$(poetry version release)
new_v=$(echo $new_v | awk '{print $6}')
poetry run python scripts/release_scripts/version_sync.py
git add pyproject.toml
git commit -m 'version increased'
git checkout $atk_branch
git stash pop
echo increased antakia version

cd ../antakia-core
poetry install
poetry build
poetry publish

cd ../antakia
poetry install
poetry build
poetry publish