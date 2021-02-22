Contributor Guidelines
======================

Developing new features
-----------------------

To develop new features for SkyPy, you will normally want to create a local copy of the SkyPy repository, instead of installing e.g. a pip or conda package.


Clone the SkyPy Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^

Cloning creates a local copy of the repository on your computer. To clone the SkyPy repository, navigate to where you keep your code files in a terminal, and run `git clone`::

  # clone the repository ...
  git clone https://github.com/skypyproject/skypy.git
  # ... which will create the `skypy` folder for you
  cd skypy


Create a branch for your new feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a *branch* off the ``skypyproject`` main branch. Working on unique branches for each new feature simplifies the development, review, and merge processes by maintining logical separation. To create a feature branch::

  # first, make sure we are on the main branch
  git checkout main
  # pull in potential changes from the repository
  git pull
  # now create your new branch with name "BRANCH_NAME"
  git checkout -b BRANCH_NAME


Hack away!
^^^^^^^^^^

Write the new code you would like to contribute and *commit* it to the feature branch on your local repository. Ideally commit small units of work often, with clear and descriptive commit messages describing the changes you made. To commit changes to a file::

  # see the changes in the working directory
  git status
  # add a file called "FILE_WITH_CHANGES" to the index to be committed
  git add FILE_WITH_CHANGES
  # commit all staged changes from the index
  git commit


Contributing
------------

Once you have developed your new SkyPy feature, you may want to contribute it to the main repository. To do so, you have to create your own copy of the repository (called a "fork") once, push your changes there, and then propose them for merging into the main repository. Please keep the coding guidelines below in mind.


Fork the SkyPy Repository
^^^^^^^^^^^^^^^^^^^^^^^^^

**You should only need to do this step once**

First *fork* the SkyPy repository. A fork is your own remote copy of the repository on GitHub. To create a fork:

  1. Go to the `SkyPy GitHub Repository <https://github.com/skypyproject/skypy>`_
  2. Click the **Fork** button (in the top-right-hand corner)
  3. Choose where to create the fork, typically your personal GitHub account


Add your fork as a remote repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, you need to make your fork known to your local SkyPy repository. To add a *remote* repository::

  git remote add GITHUB_USER https://github.com/GITHUB_USER/skypy.git

Here ``GITHUB_USER`` is your GitHub account name (i.e. the account where you created the fork in step 3 above) and the name of the new remote.


Push your changes to your fork
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make your proposed changes visible to the world, you need to *push* your feature branch to your remote fork on GitHub::

  # on the first push, use -u to tell git that this feature branch always goes to your fork
  git push -u GITHUB_USER BRANCH_NAME
  # on subsequent pushes, you can then `git push` without options


Open a Pull Request
^^^^^^^^^^^^^^^^^^^

When you feel that work on your new feature is complete, you should create a *pull request*. This will propose your work to be merged into the main SkyPy repository.

  1. Go to `SkyPy Pull Requests <https://github.com/skypyproject/skypy/pulls>`_
  2. Click the green **New pull request** button
  3. Click **compare across forks**
  4. Confirm that the base fork is ``skypyproject/skypy`` and the base branch is ``main``
  5. Confirm the head fork is ``<your-account>/skypy`` and the compare branch is ``<your-branch-name>``
  6. Give your pull request a title and fill out the the template for the description
  7. Click the green **Create pull request** button


Status checks
^^^^^^^^^^^^^

A series of automated checks will be run on your pull request, some of which will be required to pass before it can be merged into the main codebase:

  - ``Tests`` (Required) runs the `unit tests`_ in four predefined environments; `latest supported versions`, `oldest supported versions`, `macOS latest supported` and `Windows latest supported`. Click "Details" to view the output including any failures.
  - ``Code Style`` (Required) runs `flake8 <https://flake8.pycqa.org/en/latest/>`__ to check that your code conforms to the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines. Click "Details" to view any errors.
  - ``codecov`` reports the test coverage for your pull request; you should aim for `codecov/patch — 100.00%`. Click "Details" to view coverage data.
  - ``docs`` (Required) builds the `docstrings`_ on `readthedocs <https://readthedocs.org/>`_. Click "Details" to view the documentation or the failed build log.


Updating your branch
^^^^^^^^^^^^^^^^^^^^

As you work on your feature, new commits might be made to the ``skypyproject`` main branch. You will need to update your branch with these new commits before your pull request can be accepted. You can achieve this in a few different ways:

  - If your pull request has no conflicts, click **Update branch** on its GitHub pull request page.
  - If your pull request has conflicts, click **Resolve conflicts** on its GitHub pull request page, manually resolve the conflicts and click **Mark as resolved**.
  - You can also *merge* the ``skypyproject`` main branch from the command line::

        # first go to the main branch to receive changes
        git checkout main
        # pull in the changes from the main repository
        git pull
        # now switch back to your branch
        git checkout BRANCH_NAME
        # merge in the changes, you may need to resolve conflicts
        git merge main

For more information about resolving conflicts see the GitHub guides:
  - `Resolving a merge conflict on GitHub <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-on-github>`_
  - `Resolving a merge conflict using the command line <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line>`_
  - `About Git rebase <https://help.github.com/en/github/using-git/about-git-rebase>`_

More Information
^^^^^^^^^^^^^^^^

More information regarding the usage of GitHub can be found in the `GitHub Guides <https://guides.github.com/>`_.


Coding Guidelines
-----------------

Before your pull request can be merged into the codebase, it will be reviewed by one of the SkyPy developers and required to pass a number of automated checks. Below are a minimum set of guidelines for developers to follow:


General Guidelines
^^^^^^^^^^^^^^^^^^

- SkyPy is compatible with Python>=3.6 (see `setup.cfg <https://github.com/skypyproject/skypy/blob/master/setup.cfg>`_). SkyPy *does not* support backwards compatibility with Python 2.x; `six`, `__future__` and `2to3` should not be used.
- All contributions should follow the `PEP8 Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. We recommend using `flake8 <https://flake8.pycqa.org/>`__ to check your code for PEP8 compliance.
- Importing SkyPy should only depend on having `NumPy <https://www.numpy.org>`_, `SciPy <https://www.scipy.org/>`_ and `Astropy <https://www.astropy.org/>`__ installed.
- Code is grouped into submodules based on broad science areas e.g. `galaxies <https://skypy.readthedocs.io/en/stable/galaxies.html>`_. There is also a `utils <https://skypy.readthedocs.io/en/stable/utils/index.html>`_ submodule for general utility functions.
- For more information see the `Astropy Coding Guidelines <http://docs.astropy.org/en/latest/development/codeguide.html>`_.


Unit Tests
^^^^^^^^^^

Pull requests will require existing unit tests to pass before they can be merged. Additionally, new unit tests should be written for all new public methods and functions. Unit tests for each submodule are contained in subdirectories called ``tests`` and you can run them locally using ``pytest``. For more information see the `Astropy Testing Guidelines <https://docs.astropy.org/en/stable/development/testguide.html>`_.

If your unit tests check the statistical distribution of a random sample, the test outcome itself is a random variable, and the test will fail from time to time. Please mark such tests with the ``@pytest.mark.flaky`` decorator, so that they will be automatically tried again on failure. To prevent non-random test failures from being run multiple times, please isolate random statistical tests and deterministic tests in their own test cases.


Docstrings
^^^^^^^^^^

All public classes, methods and functions require docstrings. You can build documentation locally by installing `sphinx-astropy <https://github.com/astropy/sphinx-astropy>`_ and calling ``make html`` in the ``docs`` subdirectory. Docstrings should include the following sections:

  - Description
  - Parameters
  - Notes
  - Examples
  - References

For more information see the Astropy guide to `Writing Documentation <https://docs.astropy.org/en/stable/development/docguide.html>`_.
