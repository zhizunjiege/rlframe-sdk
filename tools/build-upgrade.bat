@echo off

@REM build
python setup.py bdist_wheel

@REM upgrade
for %%w in (dist/*.whl) do pip install --upgrade --force-reinstall --no-deps dist/%%w
