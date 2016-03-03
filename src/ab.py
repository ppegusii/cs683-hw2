#!/usr/bin/env python
from __future__ import print_function

INF = float('inf')


def main():
    gt = GtNode('A', -INF, INF, [
        GtNode('B', -INF, INF, [
            GtNode('E', -INF, INF, [
                GtNode('L', 2, 2, None),
                GtNode('M', 3, 3, None),
            ]),
            GtNode('F', -INF, INF, [
                GtNode('N', 8, 8, None),
                GtNode('O', 5, 5, None),
            ]),
            GtNode('G', -INF, INF, [
                GtNode('P', 7, 7, None),
                GtNode('Q', 6, 6, None),
            ]),
        ]),
        GtNode('C', -INF, INF, [
            GtNode('H', -INF, INF, [
                GtNode('R', 0, 0, None),
                GtNode('S', 1, 1, None),
            ]),
            GtNode('I', -INF, INF, [
                GtNode('T', 5, 5, None),
                GtNode('U', 2, 2, None),
            ]),
        ]),
        GtNode('D', -INF, INF, [
            GtNode('J', -INF, INF, [
                GtNode('V', 8, 8, None),
                GtNode('W', 4, 4, None),
            ]),
            GtNode('K', -INF, INF, [
                GtNode('X', 10, 10, None),
                GtNode('Y', 2, 2, None),
            ]),
        ]),
    ])
    print(abSearch(gt))


def abSearch(node):
    v = mxVal(node, node.min, node.max)
    return v


def mxVal(node, a, b):
    print('start: {} ({},{})'.format(node.name, a, b))
    if node.children is None:
        return node.max
    v = -INF
    for c in node.children:
        v = max(v, mnVal(c, a, b))
        if v >= b:
            print('end: {} ({},{})'.format(node.name, a, b))
            print('pruning after: {}'.format(c.name))
            return v
        a = max(a, v)
    print('end: {} ({},{})'.format(node.name, a, b))
    return v


def mnVal(node, a, b):
    print('start: {} ({},{})'.format(node.name, a, b))
    if node.children is None:
        return node.max
    v = INF
    for c in node.children:
        v = min(v, mxVal(c, a, b))
        if v <= a:
            print('end: {} ({},{})'.format(node.name, a, b))
            print('pruning after: {}'.format(c.name))
            return v
        print('b <- {}'.format(min(b, v)))
        b = min(b, v)
    print('end: {} ({},{})'.format(node.name, a, b))
    return v


class GtNode:
    def __init__(self, name, mn, mx, children):
        self.name = name
        self.min = mn
        self.max = mx
        self.children = children


if __name__ == '__main__':
    main()
