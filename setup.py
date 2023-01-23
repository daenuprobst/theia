"""
    Setup file for theia.
    Use setup.cfg to configure your project.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()

# def local_scheme(version):
#     return ""


# if __name__ == "__main__":
#     try:
#         setup(
#             use_scm_version={
#                 "version_scheme": "python-simplified-semver",
#                 "local_scheme": local_scheme,
#             }
#         )
#     except:  # noqa
#         print(
#             "\n\nAn error occurred while building the project, "
#             "please ensure you have the most updated version of setuptools, "
#             "setuptools_scm and wheel with:\n"
#             "   pip install -U setuptools setuptools_scm wheel\n\n"
#         )
#         raise
