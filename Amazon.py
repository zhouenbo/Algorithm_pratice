# # Dictionary of words
# # i.e Cat Road Can Rock Crash
# # Want to build autocomplete library
# # Input C
# # Output Cat, Can, Crash
# # Input Ca
# # Output Cat, Can
#
#
# # {'c':{'a':{'t':{-1:-1}}}}
#
# class Trie:
#     def __init__(self):
#         self.hashmap = {}
#         self.end = -1
#
#     def insert(self, s):
#         hashmap = self.hashmap
#         for ch in s:
#             if ch not in hashmap:
#                 hashmap[ch] = {}
#             hashmap = hashmap[ch]
#         hashmap[self.end] = self.end
#
#     def autoComplete(self, s):
#         print(s)
#         hashmap = self.hashmap
#         for ch in s:
#             if ch not in hashmap:
#                 return []
#             hashmap = hashmap[ch]
#
#         res = []
#
#         def backtrack(hashmap, node, path):
#             path.append(node)
#             if hashmap[node] == -1:
#                 res.append(s + "".join(path[0:-1]))
#                 return
#             for nxt in hashmap[node]:
#                 backtrack(hashmap[node], nxt, path)
#             path.pop()
#
#         for key in hashmap:
#             backtrack(hashmap, key, [])
#
#         return res
#
#
# def auto_complete(vocal, prefix):
#     t = Trie()
#     for w in vocal:
#         t.insert(w)
#     return t.autoComplete(prefix)
#
# if __name__ == '__main__':
#     vocal = ['Cat', 'Road', 'Can', 'Rock', 'Crash']
#     prefix = 'Ca'
#     print(auto_complete(vocal, prefix))

#
# # Grids of 0s and 1s, 1 representing land 0 representing water
# # Replace all the islands with their size
# # E.g
# # Input:
# # 0 0 0 1 0
# # 1 1 0 1 0
# # 0 0 0 1 1
# # 1 1 0 0 0
# # 1 1 1 1 0
# # Output:
# # 0 0 0 4 0
# # 2 2 0 4 0
# # 0 0 0 4 4
# # 6 6 0 0 0
# # 6 6 6 6 0
#
#
# m= 5,n = 5

grid = [[0, 0, 0, 1, 0],
[1, 1, 0, 1, 0],
[0, 0, 0, 1, 1],
[1, 1, 0, 0, 0],
[1, 1, 1, 1, 0]]

m, n = len(grid), len(grid[-1])


def dfs_area(i, j, grid):
    grid[i][j] = -1
    area = 1
    for dx, dy in {(-1, 0), (1, 0), (0, 1), (0, -1)}:
        if 0 <= i + dx < m and 0 <= j + dy < n and grid[i + dx][j + dy] == 1:
            area += dfs_area(i + dx, j + dy, grid)
    return area


def dfs_calsize(i, j, grid, area):
    grid[i][j] = area
    for dx, dy in {(-1, 0), (1, 0), (0, 1), (0, -1)}:
        if 0 <= i + dx < m and 0 <= j + dy < n and grid[i + dx][j + dy] == -1:
            dfs_calsize(i + dx, j + dy, grid, area)


def cal_size(grid):
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                area = dfs_area(i, j, grid)
                dfs_calsize(i, j, grid, area)
    return grid

ans = cal_size(grid)
for row in ans:
    print(row)