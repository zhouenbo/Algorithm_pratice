import collections


# bfs by level for binary tree
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def bfs(root: Node):
    if not root:
        return []

    q = collections.deque([root])
    res = []
    while len(q)>0:
        l = len(q)
        for i in range(l):
            n = q.popleft()
            res.append(n.val)
            if n.left:
                q.append(n.left)
            if n.right:
                q.append(n.right)
    return res



# 1192. Critical Connections in a Network
from collections import defaultdict
# class Solution:
#     def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
#         def makeGraph(connections):
#             g = defaultdict(list)
#             for s, e in connections:
#                 g[s].append(e)
#                 g[e].append(s)
#             return g
#
#         def dfs(g, idx, par, ids, low, ans):
#             ids[idx] = self.time
#             low[idx] = self.time
#             self.time += 1
#
#             for child in g[idx]:
#                 if child == par:
#                     continue
#                 if ids[child] == -2:
#                     dfs(g, child, idx, ids, low, ans)
#                     low[idx] = min(low[idx], low[child])
#                     if ids[idx] < low[child]:
#                         ans.append([idx, child])
#                 else:
#                     low[idx] = min(low[idx], ids[child])
#
#         g = makeGraph(connections)
#         self.time = 0
#         idx = 0
#         par = -1
#         ids = [-2] * n
#         low = [-2] * n
#         ans = []
#
#         dfs(g, idx, par, ids, low, ans)
#         return ans


from collections import defaultdict


class Solution:
    def criticalConnections(self, n: int, connections):
        def makeGraph(connections):
            g = defaultdict(list)
            for s,e in connections:
                g[s].append(e)
            return g

        def dfs(g, idx, par, ids, low, est, st, ans):
            ids[idx] = self.time
            low[idx] = self.time
            self.time += 1
            est[idx] = True
            st.append(idx)

            for child in g[idx]:
                if child == par:
                    continue
                if ids[child] == -2:
                    dfs(g, child, idx, ids, low, est, st, ans)
                    low[idx] = min(low[idx], low[child])
                    if ids[idx] < low[child]:
                        ans.append([idx, child])
                elif est[child] == True:
                    low[idx] = min(low[idx], ids[child])
                else:
                    ans.append([idx, child])

            if low[idx] == ids[idx]:
                w = -1
                print("ssc: ")
                while w != idx:
                    w = st.pop()
                    print(w)
                    est[w] = False


        g = makeGraph(connections)

        self.time = 0
        ids = [-2] * n
        low = [-2] * n
        existStack = [False] * n
        st = []
        ans = []

        for i in range(n):
            if ids[i] == -2:
                print("idx: ",i)
                dfs(g, i, -1, ids, low, existStack, st, ans)
        return ans

if __name__ == "__main__":
    s = Solution()
    #print("critical: ",s.criticalConnections(5, [[1,0],[0,2],[2,1],[0,3],[3,4]]))
    #print("critical: ", s.criticalConnections(4, [[0,1], [1, 2], [2, 3]]))
    print("critical: ", s.criticalConnections(7, [[0, 1], [1, 2], [2, 0],[3,4],[4,5],[5,6],[6,4]]))