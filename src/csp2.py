from __future__ import print_function
import datetime as dt


TIMEOUT = dt.timedelta(minutes=5)


def noReorder(csp):
    return csp.selectUnassignedVariableIndex()


def mrv(csp):
    vIdxs = csp.unassignedVariablesIndexes()
    vIdxValCnts = [(vIdx, len(csp.orderDomainValues(vIdx))) for vIdx in vIdxs]
    return min(vIdxValCnts, key=lambda x: x[1])[0]


def simpleBacktrackSearch(
    csp,
    guessCnt=0,
    selectUnassignedVariableIndex=noReorder,
    inferences=[],
    start=None,
):
    if dt.datetime.now()-start > TIMEOUT:
        return SearchResult.timeout, None, guessCnt
    if csp.isCompleteAssignment():
        return SearchResult.success, csp.assignments(), guessCnt
    result = csp.inferences(inferences)
    if result == InferenceResult.failure:
        return SearchResult.failure, None, guessCnt
    elif result == InferenceResult.change:
        if csp.isCompleteAssignment():
            return SearchResult.success, csp.assignments(), guessCnt
    varIdx = selectUnassignedVariableIndex(csp)
    values = csp.orderDomainValues(varIdx)
    guessCnt += len(values)-1
    for val in values:
        if csp.isConsistent(varIdx, val):
            varD = csp.domain(varIdx)
            csp.setAssignment(varIdx, val)
            result, sol, guessCnt = simpleBacktrackSearch(
                csp.copy(),
                guessCnt=guessCnt,
                selectUnassignedVariableIndex=selectUnassignedVariableIndex,
                inferences=inferences,
                start=start,
            )
            # if result is not None:
            if result == SearchResult.success:
                return result, sol, guessCnt
            elif result == SearchResult.timeout:
                return result, None, guessCnt
            csp.setDomain(varIdx, varD)
    return SearchResult.failure, None, guessCnt


class CSP:
    def assignments(self):
        pass

    def isCompleteAssignment(self):
        pass

    def selectUnassignedVariableIndex(self):
        pass

    def orderDomainValues(self, varIdx):
        pass

    def isConsistent(self, varIdx, val):
        pass

    def domain(self, varIdx):
        pass

    def setDomain(self, varIdx, d):
        pass

    def setAssignment(self, varIdx, val):
        pass

    def copy(self):
        pass

    def inferences(self):
        pass


class InferenceResult:
    noChange = 1
    change = 2
    failure = 3


class SearchResult:
    failure = 1
    success = 2
    timeout = 3
