@echo off

@REM build
python setup.py bdist_wheel

@REM install
for %%w in (dist/*.whl) do pip install dist/%%w
