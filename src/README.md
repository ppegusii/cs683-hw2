# Homework 2
## Building
### Dependencies
Programs are dependent upon Python 2.7.11, pandas 0.17.1, and numpy 1.10.4.
## Running
```
python ./sudoku2.py -h
usage: sudoku2.py [-h] [-p PUZZLEDIR] [-o OUTDIR] [-n NUMBER]

Solve Sudoku using search and inference. Written in Python 2.7.

optional arguments:
	-h, --help            show this help message and exit
	-p PUZZLEDIR, --puzzledir PUZZLEDIR
		Directory containing puzzles. (default:	../data/sudoku_puzzles/)
	-o OUTDIR, --outdir OUTDIR
		Directory to place results. (default: ../data/sudoku_out/)
	-n NUMBER, --number NUMBER
		Problem number in [2, 3, 4] (default: 3)
```
### Example part 2
`python ./sudoku2.py -n 2`
### Example part 3
`python ./sudoku2.py -n 3`
### Example part 4
`python ./sudoku2.py -n 4`
