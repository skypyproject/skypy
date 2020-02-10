Contributing to SkyPy
=======================

GitHub Workflow
---------------

### Fork and Clone the SkyPy Repository
**You should only need to do this step once**

First *fork* the SkyPy repository. A fork is your own remote copy of the repository on GitHub. To create a fork:

  1. Go to the [SkyPy GitHub Repository](https://github.com/skypyproject/skypy)
  2. Click the **Fork** button (in the top-right-hand corner)
  3. Choose where to create the fork, typically your personal GitHub account

Next *clone* your fork. Cloning creates a local copy of the repository on your computer to work with. To clone your fork:

  ```bash
  git clone https://github.com/<your-account>/skypy.git
  ```

Finally add the `skypyproject` repository as a *remote*. This will allow you to fetch changes made to the codebase. To add the `skypyproject` remote:

  ```bash
  cd skypy
  git remote add skypyproject https://github.com/skypyproject/skypy.git
  ```

### Create a branch for your new feature

Create a *branch* off the `skypyproject` development branch. Working on unique branches for each new feature simplifies the development, review and merge processes by maintining logical separation. To create a feature branch:

  ```bash
  git fetch skypyproject
  git checkout -b <your-branch-name> skypyproject/develop
  ```

### Hack away!

Write the new code you would like to contribute and *commit* it to the feature branch on your local repository. Ideally commit small units of work often with clear and descriptive commit messages describing the changes you made. To commit changes to a file:

  ```bash
  git add file_containing_your_contribution
  git commit -m 'Your clear and descriptive commit message'
  ```

*Push* the contributions in your feature branch to your remote fork on GitHub:

  ```bash
  git push origin <your-branch-name>
  ```

**Note:** The first time you *push* a feature branch you will probably need to use `--set-upstream origin` to link to your remote fork:

  ```bash
  git push --set-upstream origin <your-branch-name>
  ```

### Open a Pull Request

When you feel that work on your new feature is complete, you should create a *Pull Request*. This will propose your work to be merged into the main SkyPy repository.

  1. Go to [SkyPy Pull Requests](https://github.com/skypyproject/skypy/pulls)
  2. Click the green **New pull request** button
  3. Click **compare across forks**
  4. Confirm that the base fork is `skypyproject/skypy` and the base branch is `develop`
  5. Confirm the head fork is `<your-account>/skypy` and the compare branch is `<your-branch-name>`
  6. Give your pull request a title and fill out the the template for the description
  7. Click the green **Create pull request** button

### More Information

More information regarding the usage of GitHub can be found in the [GitHub Guides](https://guides.github.com/).

Coding Guidelines
-----------------

Before your pull request can be merged into the codebase, it will be reviewed by one of the SkyPy developers and required to pass a number of automated checks. Below are a minimum set of guidelines for developers to follow:

### General Guidelines

- SkyPy is compatible with Python>=3.5 (see [setup.cfg](setup.cfg)). SkyPy *does not* support backwards compatibility with Python 2.x; `six`, `__future__` and `2to3` should not be used.
- All contributions should follow the [PEP8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/). We recommend using [flake8](https://flake8.pycqa.org/) to check your code for PEP8 compliance.
- Importing SkyPy should only depend on having [NumPy](https://www.numpy.org), [SciPy](https://www.scipy.org/) and [Astropy](https://www.astropy.org/) installed.
- Code is grouped into submodules based on broad science areas e.g. [linear](skypy/linear), [nonlinear](skypy/nonlinear) and [galaxy](skypy/galaxy). There is also a [utils](skypy/utils) submodule for general utility functions.
- For more information see the [Astropy Coding Guidelines](http://docs.astropy.org/en/latest/development/codeguide.html)

### Unit Tests

Pull requests will require existing unit tests to pass before they can be merged. Additionally, new unit tests should be written for all new public methods and functions. Unit tests for each submodule are contained in subdirectories called `tests` and you can run them locally using `python setup.py test`. For more information see the [Astropy Testing Guidelines](https://docs.astropy.org/en/stable/development/testguide.html).

### Docstrings

All public classes, methods and functions require docstrings. You can build documentation locally by installing [sphinx-astropy](https://github.com/astropy/sphinx-astropy) and calling `python setup.py build_docs`. Docstrings should include the following sections:

  - Description
  - Parameters
  - Notes
  - Examples
  - References

For more information see the Astropy guide to [Writing Documentation](https://docs.astropy.org/en/stable/development/docguide.html).
