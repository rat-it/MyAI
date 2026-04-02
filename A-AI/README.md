uv init my-new-project
cd my-new-project

uv venv --python 3.10

uv add jupyter ipykernel pandas

uv run python -m ipykernel install --user --name=my-new-project --display-name "Python 3.10 (My New Project)"

uv run jupyter notebook
# or
uv run python my_script.py

ollama serve

curl.exe http://127.0.0.1:11435/api/tags