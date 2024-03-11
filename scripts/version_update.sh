echo $1

if [ $# -eq 0 ]
then
   echo 'Please provide version update type (patch/minor/major)'
   exit 1
else
   echo ''
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
new_v=$(poetry version $1)
new_v=$(echo $new_v | awk '{print $6}')
git add pyproject.toml
git commit -m 'version increased'
git checkout $atkc_branch
git stash pop
echo increased antakia-core version

cd ../antakia
new_v=$(poetry version $1)
new_v=$(echo $new_v | awk '{print $6}')
poetry run python scripts/release_scripts/version_sync.py
git add pyproject.toml
git commit -m 'version increased'
git checkout $atk_branch
git stash pop
echo increased antakia version

cd ../antakia-core
poetry shell
poetry install
poetry build
poetry publish

cd ../antakia
poetry shell
poetry install
poetry build
poetry publish
