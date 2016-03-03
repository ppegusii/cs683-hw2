#!/usr/bin/env python
from __future__ import print_function
import argparse
import copy
import datetime as dt
import itertools as it
import multiprocessing as mp
import numpy as np
import operator
import os
import pandas as pd
import sys

import csp2


def main():
    args = parseArgs(sys.argv)
    problems = {
        2: p2,
        3: p3,
        4: p4,
    }
    problems[args.number](args)


def p2(args):
    d, a = Sudoku.domainAssigned('../data/sudoku_example/sudoku_ex.txt')
    s = Sudoku(d, a)
    result, sol, guessCnt = csp2.simpleBacktrackSearch(
        s,
        selectUnassignedVariableIndex=csp2.noReorder,
        start=dt.datetime.now(),
        # inferences=[InferenceType.ac3],
        # inferences=[InferenceType.nc],
    )
    print(sol)
    assert np.all(np.equal(
        sol.values,
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
    argsFnOrder = [list(x) for x in list(it.product(
        [args],
        sorted(next(os.walk(args.puzzledir))[2]),
        [csp2.noReorder.__name__, csp2.mrv.__name__],
    ))]
    # add empty inferences
    map(lambda x: x.append([]), argsFnOrder)
    argsFnOrderInf = [tuple(x) for x in argsFnOrder]
    pool = mp.Pool()
    pool.map(pEach, argsFnOrderInf)


def p4(args):
    pool = mp.Pool()
    pool.map(
        pEach,
        # [(args, fn, csp2.mrv.__name__, [InferenceType.nc])
        # [(args, fn, csp2.mrv.__name__, [InferenceType.ac3, InferenceType.ht])
        # [(args, fn, csp2.mrv.__name__, [InferenceType.ac3, InferenceType.nc, InferenceType.ht])
        # [(args, fn, csp2.mrv.__name__, [InferenceType.ht])
        # [(args, fn, csp2.mrv.__name__, [InferenceType.nt])
        # [(args, fn, csp2.mrv.__name__, [InferenceType.ac3, InferenceType.nc, InferenceType.nt])
        # [(args, fn, csp2.mrv.__name__, [InferenceType.ac3])
        [(args, fn, csp2.mrv.__name__, [InferenceType.ac3, InferenceType.nc])
         for fn in sorted(next(os.walk(args.puzzledir))[2])],
    )


def pEach(argsFnOrderInf):
    args, fn, order, infTypes = argsFnOrderInf
    # if fn.find('015') == -1:
    #     return
    order = csp2.noReorder if order == csp2.noReorder.__name__ else csp2.mrv
    outFile = os.path.join(
        args.outdir,
        'p{}_{}_{}{}.csv'.format(
            args.number,
            os.path.splitext(fn)[0],
            order.__name__,
            '_{}'.format('-'.join(infTypes)) if len(infTypes) > 0 else '',
        ),
    )
    if os.path.isfile(outFile):
        return
    print('{} Starting: {} {}'.format(
        dt.datetime.now(), os.path.splitext(fn)[0], order.__name__))
    d, a = Sudoku.domainAssigned(os.path.join(args.puzzledir, fn))
    start = dt.datetime.now()
    s = Sudoku(d, a)
    s.ac3(repeat=False)
    # print('Product of domain sizes: {}'.format(
    #     reduce(operator.mul, [len(x)/2. for x in s.d.values()])
    # ))
    # return
    result, sol, guessCnt = csp2.simpleBacktrackSearch(
        s,
        selectUnassignedVariableIndex=order,
        inferences=infTypes,
        start=dt.datetime.now(),
    )
    compTime = (dt.datetime.now()-start).total_seconds()
    if result == csp2.SearchResult.failure:
        print('Error on {}'.format(fn))
        return
    with open(outFile, 'w') as f:
        f.write('puzzle, order, guessCnt, compTime, solution\n')
        f.write('"{}","{}","{}","{}","{}","{}"'.format(
            os.path.splitext(fn)[0],
            order.__name__,
            '-'.join(infTypes),
            guessCnt,
            compTime,
            sol.values.tolist() if sol is not None else 'timeout',
        ))
    print('{} Finished: {} {} {} {}'.format(
        dt.datetime.now(), os.path.splitext(fn)[0], order.__name__, guessCnt,
        compTime))
    # sys.exit(0)


class Sudoku(csp2.CSP):
    def __init__(self, domain, assignments):
        self.d = domain
        self.a = assignments

    def assignments(self):
        # get row indices where domain size is 1
        p = pd.DataFrame(
            np.full((9, 9), 0, dtype=np.uint8),
            index=range(1, 10),
            columns=range(1, 10),
        )
        for varIdx, val in self.a.viewitems():
            p.loc[varIdx[0], varIdx[1]] = val
        return p

    def isCompleteAssignment(self):
        return len(self.a) == 81

    def unassignedVariablesIndexes(self):
        return iter(self.d.viewkeys())

    def selectUnassignedVariableIndex(self):
        return next(self.unassignedVariablesIndexes())

    def orderDomainValues(self, varIdx):
        return self.d[varIdx]

    def isConsistent(self, varIdx, val):
        self.a[varIdx] = val
        # Row consistency
        rowVals = []
        for i in xrange(1, 10):
            rowVal = self.a.get((varIdx[0], i))
            if rowVal is not None:
                rowVals.append(rowVal)
        if len(rowVals) != len(set(rowVals)):
            del self.a[varIdx]
            return False
        # Column consistency
        colVals = []
        for i in xrange(1, 10):
            colVal = self.a.get((i, varIdx[1]))
            if colVal is not None:
                colVals.append(colVal)
        if len(colVals) != len(set(colVals)):
            del self.a[varIdx]
            return False
        # 3x3 consistency
        ulr, ulc = Sudoku.threeByThreeUpperLeft(varIdx)
        vals = []
        for i in xrange(ulr, ulr+3):
            for j in xrange(ulc, ulc+3):
                val = self.a.get((i, j))
                if val is not None:
                    vals.append(val)
        if len(vals) != len(set(vals)):
            del self.a[varIdx]
            return False
        del self.a[varIdx]
        return True

    @staticmethod
    def threeByThreeUpperLeft(varIdx):
        ulr = (varIdx[0] - 1) / 3 * 3 + 1
        ulc = (varIdx[1] - 1) / 3 * 3 + 1
        return ulr, ulc

    def domain(self, varIdx):
        return self.d[varIdx]

    def setDomain(self, varIdx, d):
        del self.a[varIdx]
        self.d[varIdx] = d

    def setAssignment(self, varIdx, val):
        del self.d[varIdx]
        self.a[varIdx] = val

    def copy(self):
        return Sudoku(copy.deepcopy(self.d), copy.copy(self.a))

    def neighbors(self, varIdx):
        n = set([(varIdx[0], i) for i in xrange(1, 10)])
        n.update([(i, varIdx[1]) for i in xrange(1, 10)])
        ulr, ulc = Sudoku.threeByThreeUpperLeft(varIdx)
        n.update([(i, j) for i in xrange(ulr, ulr+3)
                  for j in xrange(ulc, ulc+3)])
        n.remove(varIdx)
        return n

    def inferences(self, infTypes):
        anyChange = False
        changed = True
        while changed:
            changed = False
            for infT in infTypes:
                if infT == InferenceType.ac3:
                    result = self.ac3()
                elif infT == InferenceType.nc:
                    result = self.neighborsCant()
                elif infT == InferenceType.ht:
                    result = self.hiddenTwin()
                elif infT == InferenceType.nt:
                    result = self.nakedTwin()
                if result == csp2.InferenceResult.failure:
                    return result
                if result == csp2.InferenceResult.change:
                    anyChange = True
                    changed = True
        return (csp2.InferenceResult.change if anyChange else
                csp2.InferenceResult.noChange)

    def ac3(self, repeat=True):
        anyChange = False
        q = list(self.unassignedVariablesIndexes())
        while len(q) > 0:
            change = False
            varIdx = q.pop()
            domain = self.d.get(varIdx)
            if domain is None:
                continue
            # indices of assigned neighbors
            neighbors = self.neighbors(varIdx)
            aIdxs = self.a.viewkeys() & neighbors
            for aIdx in aIdxs:
                try:
                    domain.remove(self.a[aIdx])
                    anyChange = True
                    change = True
                    if len(domain) == 0:
                        return csp2.InferenceResult.failure
                except:
                    pass
            if change and len(domain) == 1:
                # create an assignment for this variable
                self.a[varIdx] = self.d[varIdx].pop()
                del self.d[varIdx]
            if change and repeat:
                # add unassigned neighbors back to q
                q += self.d.viewkeys() & neighbors
        return (csp2.InferenceResult.change if anyChange
                else csp2.InferenceResult.noChange)

    @staticmethod
    def domainAssigned(fn):
        d = {}
        a = {}
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
                if p.loc[r, c] == 0:
                    d[(r, c)] = set(range(1, 10))
                else:
                    a[(r, c)] = int(p.loc[r, c])
        return d, a

    def neighborsCant(self):
        anyChange = False
        q = list(self.unassignedVariablesIndexes())
        while len(q) > 0:
            varIdx = q.pop()
            domain = self.d.get(varIdx)
            if domain is None:
                continue
            domain = copy.copy(domain)
            # indices of assigned neighbors
            neighbors = self.neighbors(varIdx)
            aIdxs = self.a.viewkeys() & neighbors
            for aIdx in aIdxs:
                try:
                    domain.remove(self.a[aIdx])
                except:
                    pass
            # indices of unassigned neighbors
            uaIdxs = neighbors & self.d.viewkeys()
            for uaIdx in uaIdxs:
                domain = domain - self.d[uaIdx]
            if len(domain) == 1:
                anyChange = True
                self.a[varIdx] = domain.pop()
                del self.d[varIdx]
                q += list(uaIdxs)
            '''
            This should be unreachable.
            elif len(domain) > 1 and len(domain) != len(self.d[varIdx]):
                anyChange = True
                self.d[varIdx] = domain
                q += list(uaIdxs)
            '''
        return (csp2.InferenceResult.change if anyChange
                else csp2.InferenceResult.noChange)

    def hiddenTwin(self):
        anyChange = False
        q = list(self.unassignedVariablesIndexes())
        while len(q) > 1:
            varIdx = q.pop()
            domain = self.d.get(varIdx)
            if domain is None:
                continue
            neighbors = self.neighbors(varIdx)
            # indices of unassigned neighbors
            uaIdxs = neighbors & self.d.viewkeys()
            for uaIdx in uaIdxs:
                if len(domain) == 2 and len(self.d[uaIdx]) == 2:
                    continue
                comVals = domain & self.d[uaIdx]
                if len(comVals) == 2:
                    others = uaIdxs - set([uaIdx])
                    shared = False
                    for othIdx in others:
                        if len(comVals & self.d[othIdx]) > 0:
                            shared = True
                            break
                    if shared:
                        continue
                    anyChange = True
                    self.d[varIdx] = copy.copy(comVals)
                    self.d[uaIdx] = comVals
                    q.remove(uaIdx)
                    break
        return (csp2.InferenceResult.change if anyChange
                else csp2.InferenceResult.noChange)

    def nakedTwin(self):
        anyChange = False
        q = list(self.unassignedVariablesIndexes())
        while len(q) > 1:
            varIdx = q.pop()
            domain = self.d.get(varIdx)
            if domain is None or len(domain) != 2:
                continue
            neighbors = self.neighbors(varIdx)
            # indices of unassigned neighbors
            uaIdxs = neighbors & self.d.viewkeys()
            for uaIdx in uaIdxs:
                neighDom = self.d[uaIdx]
                if neighDom != 2 or len(neighDom & domain) != 2:
                    continue
                others = uaIdxs - set([uaIdx])
                for othIdx in others:
                    if len(self.d[othIdx] & domain) > 0:
                        anyChange = True
                        self.d[othIdx] = self.d[othIdx] - domain
        return (csp2.InferenceResult.change if anyChange
                else csp2.InferenceResult.noChange)


class InferenceType:
    ac3 = 'ac3'
    nc = 'nc'
    ht = 'ht'
    nt = 'nt'


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
        help='Problem number in {}'.format(range(2, 5)),
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
