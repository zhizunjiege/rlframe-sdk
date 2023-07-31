@echo off

@REM remember to run all services before running this script
python -m unittest discover -s src/tests
