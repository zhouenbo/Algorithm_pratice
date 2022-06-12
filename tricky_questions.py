#
# Leetcode 708 Insert into a Cyclic Sorted List
# Given a node from a cyclic linked list which has been sorted, write a function to insert a value into the list
# such that it remains a cyclic sorted list. The given node can be any single node in the list.
# Return the inserted new node.


"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param node: a list node in the list
    @param x: An integer
    @return: the inserted new list node
    """

    def insert(self, node, x):
        # write your code here
        if not node:
            n = ListNode(x)
            n.next = n
            return n

        if node.next == node or node.val == x:
            n = ListNode(x, node.next)
            node.next = n
            return n

        cur = node
        while cur.val <= cur.next.val and cur.next != node:
            cur = cur.next
        if cur.val <= x or cur.next.val >= x:
            n = ListNode(x, cur.next)
            cur.next = n
            return n

        while cur.next.val < x:
            cur = cur.next
        n = ListNode(x, cur.next)
        cur.next = n
        return n
