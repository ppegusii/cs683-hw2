from __future__ import print_function


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
):
    if csp.isCompleteAssignment():
        return csp.assignments(), guessCnt
    csp.inferences()
    varIdx = selectUnassignedVariableIndex(csp)
    values = csp.orderDomainValues(varIdx)
    guessCnt += len(values)
    for val in values:
        if csp.isConsistent(varIdx, val):
            csp.setAssignment(varIdx, val)
            result, guessCnt = simpleBacktrackSearch(
                csp.copy(),
                guessCnt=guessCnt,
                selectUnassignedVariableIndex=selectUnassignedVariableIndex,
            )
            if result is not None:
                return result, guessCnt
    return None, guessCnt


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

    def setAssignment(self, varIdx, val):
        pass

    def copy(self):
        pass

    def inferences(self):
        pass
