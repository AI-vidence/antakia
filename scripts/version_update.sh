echo $1

if [ $# -eq 0 ]
then
   echo 'Please provide version update type (patch/minor/major)'
   exit 1
else
   echo ''
fi

cd ../antakia-core
test_output=$(./scripts/test.sh)
test_exit_code=$?
if [ $test_exit_code -ne 0 ]
then
  echo 'antakia core test failed'
  echo $test_output
  exit 1
else
  echo 'atk core test SUCCESS'
fi

type_check_output=$(./scripts/type-check.sh)
type_check_exit_code=$?
if [ $type_check_exit_code -ne 0 ]
then
  echo 'antakia core type check failed'
  echo $type_check_output
  exit 1
else
  echo 'atk core type SUCCESS'
fi

format_output=$(./scripts/format.sh)
format_exit_code=$?
if [ $format_exit_code -ne 0 ]
then
  echo 'antakia core format failed'
  echo $format_output
  exit 1
else
  echo 'atk core format SUCCESS'
fi

git stash
atkc_branch=$(git rev-parse --abbrev-ref HEAD)
git checkout -f dev
git pull
atkc=$(echo $(poetry version) | awk '{print $2}')

cd ../antakia
test_output=$(./scripts/code_quality/test.sh)
test_exit_code=$?
if [ $test_exit_code -ne 0 ]
then
  echo 'antakia test failed'
  echo $test_output
  exit 1
else
  echo 'atk test SUCCESS'
fi

type_check_output=$(./scripts/code_quality/type-check.sh)
type_check_exit_code=$?
if [ $type_check_exit_code -ne 0 ]
then
  echo 'antakia type check failed'
  echo $type_check_output
  exit 1
else
  echo 'atk type SUCCESS'
fi

format_output=$(./scripts/code_quality/format.sh)
format_exit_code=$?
if [ $format_exit_code -ne 0 ]
then
  echo 'antakia format failed'
  echo $format_output
  exit 1
else
  echo 'atk format SUCCESS'
fi

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
git push
poetry install
poetry build
poetry publish
git checkout $atkc_branch
git stash pop
echo increased antakia-core version

cd ../antakia
new_v=$(poetry version $1)
new_v=$(echo $new_v | awk '{print $6}')
poetry run python scripts/version_sync.py
git add pyproject.toml
git commit -m 'version increased'
git push
poetry install
poetry build
poetry publish
git checkout $atk_branch
git stash pop
echo increased antakia version

