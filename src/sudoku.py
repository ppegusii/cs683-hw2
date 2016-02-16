#!/usr/bin/env python
from __future__ import print_function
import argparse
import itertools as it
import numpy as np
import pandas as pd

import csp


def main():
    domain = Sudoku.domain('../data/sudoku_puzzles/sudoku_ex.txt')
    s = Sudoku(domain)
    print(csp.simpleBacktrackSearch(s))


class Sudoku(csp.CSP):
    def __init__(self, domain):
        self.d = domain

    def oldassignments(self):
        return [idx for idx in it.product(self.d.index, self.d.columns) if
                self.d.loc[idx]]

    def assignments(self):
        # get row indices where domain size is 1
        rds1 = self.d[self.d.sum(axis=1) == 1]
        p = pd.DataFrame(
            np.full((9, 9), 0, dtype=np.uint8),
            index=range(1, 10),
            columns=range(1, 10),
        )
        asns = [idx for idx in it.product(rds1.index, rds1.columns) if
                rds1.loc[idx]]
        for a in asns:
            p.loc[a[0]] = a[1]
        return p

    def isCompleteAssignment(self):
        return self.d.sum().sum() == 81

    def unassignedVariablesIndexes(self):
        return self.d[self.d.sum(axis=1) > 1].index

    def selectUnassignedVariableIndex(self):
        return self.unassignedVariablesIndexes()[0]

    def orderDomainValues(self, varIdx):
        return self.d.loc[varIdx][self.d.loc[varIdx]].index

    def isConsistent(self, varIdx, val):
        oldAss = self.d.loc[varIdx].copy()
        self.setAssignment(varIdx, val)
        p = self.assignments()
        for i in range(1, 10):
            rowValCnts = p.loc[i, :].value_counts().drop(0, errors='ignore')
            colValCnts = p.loc[:, i].value_counts().drop(0, errors='ignore')
            if (
                # check row consistency
                (len(rowValCnts) > 0 and rowValCnts.iloc[0] > 1) or
                # check col consistency
                (len(colValCnts) > 0 and colValCnts.iloc[0] > 1)
            ):
                self.d.loc[varIdx] = oldAss
                return False
        for i in range(0, 3):
            for j in range(0, 3):
                # check submatrix consistency
                subP = pd.Series(
                    p.loc[3*i+1:3*i+3, 3*j+1:3*j+3].values.flatten())
                valCnts = subP.value_counts().drop(0, errors='ignore')
                if len(valCnts) > 0 and valCnts.iloc[0] > 1:
                    self.d.loc[varIdx] = oldAss
                    return False
        self.d.loc[varIdx] = oldAss
        return True

    def setAssignment(self, varIdx, val):
        self.d.loc[varIdx] = False
        self.d.loc[varIdx, val] = True

    def copy(self):
        return Sudoku(self.d.copy())

    @staticmethod
    def domain(fn):
        index = pd.MultiIndex.from_tuples(
            list(it.product(range(1, 10), repeat=2)),
            names=['row', 'col'],
        )
        d = pd.DataFrame(
            np.full((81, 9), True, dtype=bool),
            index=index,
            columns=range(1, 10),
        )
        p = pd.read_csv(
            fn,
            sep=' ',
            na_values='-',
            header=None,
            names=range(1, 10),
            index_col=False,
        )
        p.fillna(0, inplace=True)
        p.astype(np.uint8, copy=False)
        p.index = range(1, 10)
        for r in range(1, 10):
            for c in range(1, 10):
                if p.loc[r, c] != 0:
                    d.loc[(r, c)] = False
                    d.loc[(r, c), p.loc[r, c]] = True
        return d


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
