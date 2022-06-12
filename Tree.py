######################################################################################
# leetcode 426 Convert Binary Search Tree to Sorted Doubly Linked List
######################################################################################

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param root: root of a tree
    @return: head node of a doubly linked list
    """

    def treeToDoublyList(self, root):
        # Write your code here.

        def inorder_dfs(node):
            if not node:
                return None, None

            if not node.left and not node.right:
                return node, node

            left, right = node, node
            if node.right:
                s_r, right = inorder_dfs(node.right)
                node.right = s_r
                s_r.left = node
            if node.left:
                left, e_l = inorder_dfs(node.left)
                e_l.right = node
                node.left = e_l
            return left, right

        s, e = inorder_dfs(root)
        s.left = e
        e.right = s
        return s