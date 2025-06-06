# Installation

Currently, GAICo is not available on PyPI. To use it, you'll need to clone the repository and set up the environment using UV.

1. First, make sure you have UV installed. If not, you can install it by following the instructions on the [official UV website](https://docs.astral.sh/uv/#installation).

2. Clone the repository:

   ```shell
   git clone https://github.com/ai4society/GenAIResultsComparator.git
   cd GenAIResultsComparator
   ```

3. Ensure the dependencies are installed by creating a virtual env. (python 3.12 is recommended):

   ```shell
   uv venv
   uv sync
   ```

4. (Optional) Activate the virtual environment (doing this avoids prepending `uv run` to any proceeding commands):
   ```shell
   source .venv/bin/activate
   ```

_If you don't want to use `uv`,_ you can install the dependencies with the following commands:

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

However note that the `requirements.txt` is generated automatically with the pre-commit file and might not include all the dependencies (in such case, a manual pip install might be needed).

Now you're ready to use GAICo!
