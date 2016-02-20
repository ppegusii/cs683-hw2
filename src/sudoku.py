#!/usr/bin/env python
from __future__ import print_function
import argparse
import datetime as dt
import itertools as it
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys

import csp


def main():
    args = parseArgs(sys.argv)
    problems = {
        2: p2,
        3: p3,
    }
    problems[args.number](args)


def p2(args):
    domain = Sudoku.domain('../data/sudoku_example/sudoku_ex.txt')
    s = Sudoku(domain)
    sol = csp.simpleBacktrackSearch(
        s,
        selectUnassignedVariableIndex=csp.noReorder,
    )
    print(sol)
    assert np.all(np.equal(
        sol[0].values,
        np.array([
            [4, 3, 5, 2, 6, 9, 7, 8, 1],
            [6, 8, 2, 5, 7, 1, 4, 9, 3],
            [1, 9, 7, 8, 3, 4, 5, 6, 2],
            [8, 2, 6, 1, 9, 5, 3, 4, 7],
            [3, 7, 4, 6, 8, 2, 9, 1, 5],
            [9, 5, 1, 7, 4, 3, 6, 2, 8],
            [5, 1, 9, 3, 2, 6, 8, 7, 4],
            [2, 4, 8, 9, 5, 7, 1, 3, 6],
            [7, 6, 3, 4, 1, 8, 2, 5, 9],
        ])
    ))


def p3(args):
    pool = mp.Pool(2)
    pool.map(
        p3each,
        list(it.product(
            [args],
            sorted(next(os.walk(args.puzzledir))[2]),
            [csp.noReorder.__name__, csp.mrv.__name__],
        )),
    )


def p3each(argsFnOrder):
    args, fn, order = argsFnOrder
    order = csp.noReorder if order == csp.noReorder.__name__ else csp.mrv
    outFile = os.path.join(
        args.outdir,
        'p3_{}_{}.csv'.format(
            os.path.splitext(fn)[0],
            order.__name__,
        ),
    )
    if os.path.isfile(outFile):
        return
    domain = Sudoku.domain(os.path.join(args.puzzledir, fn))
    start = dt.datetime.now()
    s = Sudoku(domain)
    sol, guessCnt = csp.simpleBacktrackSearch(
        s,
        selectUnassignedVariableIndex=order,
    )
    compTime = (dt.datetime.now()-start).total_seconds()
    with open(outFile, 'w') as f:
        f.write('puzzle, order, guessCnt, compTime, solution\n')
        f.write('"{}","{}","{}","{}","{}"'.format(
            os.path.splitext(fn)[0],
            order.__name__,
            guessCnt,
            compTime,
            sol.values.tolist(),
        ))
    print('{} {} {} {}'.format(fn, order.__name__, guessCnt, compTime))


class Sudoku(csp.CSP):
    def __init__(self, domain):
        self.d = domain

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

    def inferences(self):
        changed = 1
        while changed:
            changed = 0
            changed += self.ac3()

    def ac3(self):
        return False

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
    parser.add_argument(
        '-n', '--number',
        default=3,
        type=int,
        help='Problem number in {}'.format(range(2, 4)),
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
