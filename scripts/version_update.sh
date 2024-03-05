cd ../antakia-core
atkc= $(echo $(poetry version) | awk '{print $2}')
cd ../antakia
atk= $(echo $(poetry version) | awk '{print $2}')

if [ "$atkc" != "$atk" ]; then
   echo 'version do not match'
   exit 1
fi

cd ../antakia-core
git stash
git checkout -f dev
git pull
new_v= $(poetry version $0)
new_v= $(echo $new_v | awk '{print $6}')
git add pyproject.toml
git commit -m 'version $0 -> $new_v'
git stash pop

cd ../antakia
poetry version $0
git stash
git checkout -f dev
git pull
new_v= $(poetry version $0)
new_v= $(echo $new_v | awk '{print $6}')
git add pyproject.toml
git commit -m 'version $0 -> $new_v'
git stash pop
