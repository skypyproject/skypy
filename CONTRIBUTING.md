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
  3. On the Compare page, click **compare across forks**
  4. Confirm that the base fork is `skypyproject/skypy` and the base branch is `develop`
  5. Confirm the head fork is `<your-account>/skypy` and the compare branch is `<your-branch-name>`
  6. Give your pull request a title and fill out the the template for the description
  7. Click the green **Create pull request** button

## More Information

More information regarding the usage of GitHub can be found in the [GitHub Guides](https://guides.github.com/).
