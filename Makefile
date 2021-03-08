default: pylint pytest

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs pylint --output-format=colorized; true

pytest:
	$ pytest -v --color=yes
