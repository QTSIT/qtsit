# This script may work on only Bash and Zsh
# usage: source scripts/flake8_for_ci.sh

items=(
    "qtsit/algorithms"
)

for item in "${items[@]}" ; do
    echo ${item}; flake8 ${item} --exclude=__init__.py --count --show-source --statistics
done