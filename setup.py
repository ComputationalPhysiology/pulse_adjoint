# System imports
import os
import sys
import platform

from setuptools import setup

# Version number
major = 1
minor = 0

on_rtd = os.environ.get('READTHEDOCS') == 'True'


if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % os.path.split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)



setup(name = "pulse_adjoint",
      version = "{0}.{1}".format(major, minor),
      description = """
      An adjointable cardiac mechanics data assimilator.
      """,
      author = "Henrik Finsberg",
      author_email = "henriknf@simula.no",
      license="LGPL version 3 or later",
      # install_requires=REQUIREMENTS,
      # dependency_links=dependency_links,
      packages = ["pulse_adjoint"],
      # package_data={'pulse_adjoint.example_meshes':  ["*.h5", "*.yml"]},
      package_dir = {"pulse_adjoint": "pulse_adjoint"},
      )
