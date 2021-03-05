Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ComputationalPhysiology/pulse_adjoint/issues.

If you are reporting a bug, please include:

-  Your operating system name and version.
-  Any details about your local setup that might be helpful in
   troubleshooting.
-  Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with “bug” and
“help wanted” is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with
“enhancement” and “help wanted” is open to whoever wants to implement
it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Pulse-Adjoint could always use more documentation, whether as part of
the official Pulse-Adjoint docs, in docstrings, or even on the web in
blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/ComputationalPhysiology/pulse_adjoint/issues.

If you are proposing a feature:

-  Explain in detail how it would work.
-  Keep the scope as narrow as possible, to make it easier to implement.
-  Remember that this is a volunteer-driven project, and that
   contributions are welcome :)

Get Started!
------------

Ready to contribute? Here’s how to set up pulse_adjoint for local
development.

1. Fork the pulse_adjoint repo on GitHub (Note: if you are part of the team working
on pulse-adjoint then you can work on the original repo)

2. Clone your fork locally:

   .. code:: shell

      $ git clone git@github.com:your_name_here/pulse_adjoint.git

3. Install your local copy into a virtual environment.

   .. code:: shell

      $ cd pulse_adjoint/
      # Create virtual environment
      $ python -m venv venv
      $ source venv/bin/activate
      $ make dev

   This last command is a development install, which will install a lot of
   packages that are only used during development, as well as installing
   pulse-adjoint in editable mode (i.e the same as :code:`python -m pip install . -e`).
   The development install also installs a `pre-commit`_ hook which will run some
   tests every time you commit.


4. Create a branch for local development:

   .. code:: shell

      $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you’re done making changes, check that your changes pass flake8,
   static type checking and the tests:

   .. code:: shell

      $ make lint
      $ make type
      $ make test


6. Commit your changes and push your branch to GitHub:

   .. code:: shell

      $ git add .
      $ git commit -m "Your detailed description of your changes."
      $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.


.. _pre-commit: https://pre-commit.com


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add
   the feature to the list in README.rst.
3. The pull request should work for Python 3.7 and 3.8. Check circle CI
   at https://app.circleci.com/pipelines/github/ComputationalPhysiology/pulse_adjoint/

Tips
----

1. Use a code editor with linting enabled which helps you to catch typo bugs

2. Use type annotations - they can be a real pain, but will save you a lot of times
   and it makes it development so much better because your editor can use type hints to
   do autocompletion.

3. Commit often - it is always a pain to commit when lot of stuff is changed.

4. To run a subset of tests starting with ``test_something`` do:

.. code:: shell

   $ python -m pytest -k test_something

5. When pushing to the repo, try to increase the code coverage by writing one
   more test case. Check the code coverage at
   https://codecov.io/gh/ComputationalPhysiology/pulse_adjoint

Deploying
---------

A reminder for the maintainers on how to deploy. Make sure all your
changes are committed (including an entry in HISTORY.md). Then run:

.. code:: shell

   $ bump2version patch # possible: major / minor / patch
   $ git push
   $ git push --tags

Create a new pypi package using the :code:`make release` command.
