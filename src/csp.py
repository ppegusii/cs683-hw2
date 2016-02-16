from __future__ import print_function


def simpleBacktrackSearch(csp):
    if csp.isCompleteAssignment():
        return csp.assignments()
    varIdx = csp.selectUnassignedVariableIndex()
    for val in csp.orderDomainValues(varIdx):
        if csp.isConsistent(varIdx, val):
            csp.setAssignment(varIdx, val)
            result = simpleBacktrackSearch(csp.copy())
            if result is not None:
                return result
    return None


class CSP:
    def assignment(self):
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
