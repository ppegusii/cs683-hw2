from __future__ import print_function


def backtrackSearch(csp):
    if csp.isCompleteAssignment():
        return csp.assignment()


class CSP:
    def assignment(self):
        pass

    def isCompleteAssignment(self):
        pass

    def unAssignedVariablesIndexes(self):
        pass
