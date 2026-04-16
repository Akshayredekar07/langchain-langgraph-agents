# Resolved Issues

This file is a running troubleshooting log for this project.

Use it to record:
- The problem
- The symptoms or error message
- The root cause
- The exact solution
- The final verification

Important note:
- For future package installs, prefer `uv pip install ...` instead of plain `pip install ...`
- `uv` is the preferred package manager for this project because it is much faster and works well with virtual environments

---

## Issue 001: Jupyter notebook kernel could not run because `ipykernel` was missing

### Problem
VS Code showed this message when trying to run notebook cells:

`Running cells with 'Langgraph Agents (3.13.3) (Python 3.13.3)' requires the ipykernel package.`

### Symptoms
- Notebook cells would not run
- VS Code asked to install `ipykernel`
- The target environment was:
  `d:/Langchain-dev/.venv/Scripts/python.exe`

### Root Cause
There were multiple issues in the virtual environment:

1. `pip` was missing from the `.venv`
2. After repairing `pip`, `ipykernel` could not be installed from the sandbox because network access was restricted
3. After installation, `ipykernel` import failed because `orjson` in the environment was broken and outdated
4. The broken `orjson` install had missing uninstall metadata (`RECORD`), so it had to be cleaned manually before reinstalling

### Solution We Applied

#### Step 1: Repair `pip` inside the virtual environment
We repaired `pip` with:

```powershell
d:/Langchain-dev/.venv/Scripts/python.exe -m ensurepip --upgrade
```

#### Step 2: Install `ipykernel`
We installed `ipykernel` into the same `.venv`:

```powershell
d:/Langchain-dev/.venv/Scripts/python.exe -m pip install ipykernel -U --force-reinstall
```

#### Step 3: Verify the import
Importing `ipykernel` failed because of `orjson`:

```powershell
d:/Langchain-dev/.venv/Scripts/python.exe -c "import ipykernel; print(ipykernel.__version__)"
```

The error pointed to this kind of issue:

`AttributeError: module 'orjson' has no attribute 'OPT_NAIVE_UTC'`

#### Step 4: Repair broken `orjson`
The installed `orjson` package was corrupted and had broken metadata.

We removed the broken package folders:

```powershell
Remove-Item -LiteralPath 'd:/Langchain-dev/.venv/Lib/site-packages/orjson' -Recurse -Force
Remove-Item -LiteralPath 'd:/Langchain-dev/.venv/Lib/site-packages/orjson-3.11.7.dist-info' -Recurse -Force
```

Then we reinstalled `orjson` cleanly:

```powershell
d:/Langchain-dev/.venv/Scripts/python.exe -m pip install orjson==3.11.8
```

#### Step 5: Final verification
We confirmed both packages import correctly:

```powershell
d:/Langchain-dev/.venv/Scripts/python.exe -c "import ipykernel, orjson; print(ipykernel.__version__); print(orjson.__version__)"
```

### Final Verified State
- `ipykernel == 7.2.0`
- `orjson == 3.11.8`

### Result
The notebook kernel became usable again, and VS Code should now run notebook cells with the selected Python 3.13.3 environment.

### Preferred Future Commands
For future installs, prefer `uv pip` commands like these:

```powershell
uv pip install ipykernel
uv pip install orjson
```

If you want to target the current `.venv` explicitly, first activate it or run the command from the intended environment setup for the project.

---

## Issue 002: Pylance error with LangChain structured output in notebook

### Problem
In `langchain-dev/documents/03-structured-output.ipynb`, Pylance showed errors when reading values from `response` after `model.with_structured_output(...)`.

### Symptoms
- `Cannot access attribute "get" for class "BaseModel"`
- `"__getitem__" method not defined on type "BaseModel"`

### Root Cause
`invoke()` can return either a normal dictionary or a Pydantic `BaseModel`.  
Because of that, Pylance could not guarantee that `response.get(...)` or `response["key"]` was always safe.

### Solution We Applied
We normalized the result first:
- If `response` is a `BaseModel`, convert it with `model_dump()`
- If it is already a mapping, convert it to a plain `dict`

Then we used the converted object for all key access.

### Result
The notebook code became type-safe, and the Pylance errors were resolved without changing the schema behavior.

---

## Template For Future Issues

Copy this section and add a new numbered issue below.

```md
## Issue 00X: Short issue title

### Problem
Describe the problem clearly.

### Symptoms
- Error message
- Where it happened
- What failed

### Root Cause
Explain the actual cause.

### Solution We Applied
#### Step 1
Command or change

#### Step 2
Command or change

### Final Verification
- What command was used
- What output confirmed the fix

### Result
What is working now

### Preferred Future Command
Use `uv pip install ...` where package installation is needed.
```
