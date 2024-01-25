# Formatting and style
This repository uses `isort` for import management and `black` for general formatting. Both can be installed using `pip` or `conda` and on a Linux system, they can also be installed at the system level using `apt`. For example, using `pip`, you'd simply do
```
pip install isort
pip install black
```
We use the default arguments so the tools can be executed by simply pointing them at all files in the repository. From the root directory of the project, you should execute the following:
```
isort .
black .
```
Note that it's important that `isort` is run first, because it doesn't produce a format that's consistent with `black`. 

### Branch naming
If you are adding a branch to this repository, please use the following convention: `{feature, bugfix, hotfix, release, docs}/{developer initials}/{short-hyphenated-description}`. For example, `docs/DR/add-branch-naming-convention` for this change. For a description of the prefixes, please see [here](https://medium.com/@abhay.pixolo/naming-conventions-for-git-branches-a-cheatsheet-8549feca2534).

### Docstrings
For documentation, we use the [Google](https://github.com/NilsJPWerner/autoDocstring/blob/HEAD/docs/google.md) format. I personally use [VSCode autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) plugin for templating. 

### Type hints
Typing hints, as introduced by [PEP 484](https://peps.python.org/pep-0484/), are strongly encouraged. This helps provide additional documentation and allows some code editors to make additional autocompletes. 
