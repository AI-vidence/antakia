if [ "$1" == "" ]
then
   release=patch
else
   release=$1
fi

version_output=$(./scripts/release_scripts/version_update.sh $release)
version_exit_code=$?
if [ $version_exit_code -eq 0 ]
then
  $(./scripts/release_scripts/publish.sh)
fi