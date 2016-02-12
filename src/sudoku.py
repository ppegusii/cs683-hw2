#!/usr/bin/env python
from __future__ import print_function
import argparse
import itertools as it
import numpy as np
import pandas as pd

import csp


def main():
    s = Sudoku()
    print(s.p)
    print(s.p.index)


class Sudoku(csp.CSP):
    def __init__(self):
        '''
        self.p = np.full((81, 9), True, dtype=bool)
        self.p = pd.DataFrame(
            np.full((81, 9), True, dtype=bool),
            columns = ['{}{}
        '''
        index = pd.MultiIndex.from_tuples(
            list(it.product(range(1, 10), repeat=2)),
            names=['row', 'col'],
        )
        self.p = pd.DataFrame(
            np.full((81, 9), True, dtype=bool),
            index=index,
            columns=range(1, 10),
        )

    def assignment(self):
        return [idx for idx in it.product(self.p.index, self.p.columns) if
                self.p.loc[idx]]

    def isCompleteAssignment(self):
        return self.p.sum().sum() == 9

    def unAssignedVariablesIndexes(self):
        return self.p[self.p.sum(axis=1) > 1].index


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description=('Solve Sudoku using search and inference. '
                     'Written in Python 2.7.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-p', '--puzzledir',
        default='../data/sudoku_puzzles/',
        help='Directory containing puzzles.',
    )
    parser.add_argument(
        '-o', '--outdir',
        default='../data/sudoku_out/',
        help='Directory to place results.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
