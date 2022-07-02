from queue import Queue, LifoQueue

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    for i in range(10):
        print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class TreeNode(object):
    def __init__(self, value = 0, left_node = None, right_node = None):
        self.val = value
        self.leftNode = left_node
        self.rightNode = right_node

def bfs_tree(root):
    if not root:
        return []
    que = Queue()
    que.put(root)
    ans = list()
    while not que.empty():
        current = que.get()
        if current:
            ans.append(current.val)
            que.put(current.left)
            que.put(current.right)
    return ans

def inorder_dfs_tree(root):
    st = LifoQueue()
    #st.put(root)
    ans = list()
    current = root

    while current or not st.empty():
        while current:
            st.put(current)
            current = current.left

        current = st.get()
        ans.append(current.val)
        current = current.right
    return ans

def preorder_dfs_tree(root):
    st = LifoQueue()
    ans = list()
    current = root
    while current or not st.empty():
        while current:
            ans.append(current.val)
            st.put(current)
            current = current.left

        current = st.get()
        current = current.right
    return ans

def postorder_dfs_tree(root):
    st = LifoQueue()
    ans = list()
    current = root

    while current or not st.empty():
        while current:
            st.put((current,True))
            current = current.left

        current, fir = st.get()
        if fir == True:
            st.put((current, False))
            current = current.right
        else:
            ans.append(current.val)
            current = None
    return ans

def postorder_dfs_tree_traversal(root):
    if not root:
        return []
    if not root.left and not root.right:
        return [root.val]
    return postorder_dfs_tree(root.left)+postorder_dfs_tree(root.right)+[root.val]



def binary_search(lst, target):
    '''
    :param lst: sorted asending array
    :param target: target value
    :return: -1 represents no result. number greater than -1 represent the index of the value in the array
    '''
    if len(lst)==0:
        return -1
    left = 0
    right = len(lst)-1
    while left <= right:
        mid = left + (right - left)//2
        if lst[mid] > target:
            right = mid-1
        elif lst[mid] < target:
            left = mid+1
        else:
            return mid
    return -1

####################################################################################################
#  Mergesort
#
#
####################################################################################################


def mergeSort(arr, low, high):

    def mergeArr(ar, l, h):
        tmp = [-1]*(h-l+1)
        mid = l + (h - l) // 2
        lowEnd = mid
        highStart = mid+1

        index = l
        i = l
        j = highStart
        while i <= lowEnd and j <= h:
            if ar[i] <= ar[j]:
                tmp[index] = ar[i]
                i += 1
            else:
                tmp[index] = ar[j]
                j += 1
            index += 1

        if i < lowEnd:
            tmp[len(ar) - (lowEnd-i+1):] = ar[i:lowEnd+1]
        else:
            tmp[len(ar) - (h - j + 1):] = ar[j:h+1]

    if low >= high:
        return
    mid = low + (high - low)//2
    mergeSort(arr,low, mid)
    mergeSort(arr,mid+1, high)
    mergeArr(arr,low,high)

####################################################################################################
# QuickSort
#
#
####################################################################################################


def qsort(arr, low, high):
    if low >= high:
        return

    i = low
    j = high
    tar = i
    while i < j:
        if i == tar:
            if arr[j] < arr[tar]:
                tmp = arr[j]
                arr[j] = arr[tar]
                arr[tar] = tmp
                tar = j
                i += 1
            else:
                j -= 1
        else: #j == tar
            if arr[i] > arr[tar]:
                tmp = arr[i]
                arr[i] = arr[tar]
                arr[tar] = tmp
                tar = i
                j -= 1
            else:
                i += 1
    qsort(arr, low, tar-1)
    qsort(arr, tar+1, high)
    return

##########################################################
#insertion sort
############################################################
def insertionSort(arr, l, h):
    for i in range(l+1,h+1):
        key = arr[i]
        j = i - 1
        while j>=l and arr[j]>key:
            arr[j+1] = arr[j]
            j-=1
        arr[j+1] = key

###################################################################################################
# Heap Sort
#
###################################################################################################
def heapSort(arr, l, r):
    def heapify(arr, s, l, r):
        m = s
        left = l + 2*(s-l)+1
        right = l + 2*(s-l)+2

        if left <= r and arr[left]>arr[m]:
            m = left
        if right <= r and arr[right]>arr[m]:
            m = right

        if m != s:
            arr[m], arr[s] = arr[s], arr[m]
            heapify(arr, m, l, r)

    for i in range(l + (r+1-l)//2 - 1,l-1,-1):
        heapify(arr, i, l, r)

    for i in range(r, l, -1):
        arr[i], arr[l] = arr[l], arr[i]
        heapify(arr, l, l, i-1)


####################################################################################################
#  Leetcode 621: task scheduler
#
#
####################################################################################################
import string
from collections import Counter

class Solution:
    def leastInterval(self, tasks, n):
        fre = Counter(tasks)
        fre_sort = sorted(fre.items(), key=lambda x: x[1], reverse=True)
        mx = fre_sort[0][1]
        i = 0
        while i < len(fre_sort) and fre_sort[i][1] == mx:
            i += 1
        return max(len(tasks), (mx - 1) * (n + 1) + i)


####################################################################################################
#  Leetcode 53: Maximum Subarray
#
#
####################################################################################################
def maxSubArray(self, nums):
    max_sum = nums[0]
    sum_tmp = nums[0]
    for i in range(1,len(nums)):
        sum_tmp = max(sum_tmp+nums[i],nums[i])
        max_sum = max(sum_tmp,max_sum)
    return max_sum

####################################################
# 2 ByteDance
#
#####################################################
import re
def versionControl(version1, version2):
    v1 = re.split(r'[\.\-]+', version1)
    v2 = re.split(r'[\.\-]+', version2)
    i = 0
    while i<min(len(v1),len(v2),3):
        if int(v1[i])!=int(v2[i]):
            return 1 if int(v1[i])>int(v2[i]) else -1
        i+=1
    if i == 3:
        if len(v1)==len(v2):
            return 0
        else:
            return 1 if len(v1)<len(v2) else -1
    else:
        return 1 if len(v1)>len(v2) else -1 if len(v1)<len(v2) else 0

##################################################################
# 3 Select Pairs
#
###################################################################
from queue import Queue
def minLastNumber(arr):
    if len(arr) ==0:
        return 0
    elif len(arr)==1:
        return arr[0]

    total = sum(arr)
    arr.sort(reverse = True)
    q = Queue()
    q.put((0,arr[0]))
    minn = abs(total - arr[0] - arr[0])

    while not q.empty():
        ind, tmp = q.get()
        if ind+1<len(arr):
            q.put((ind+1, tmp))
            tmp = tmp + arr[ind+1]
            if abs(total-2*tmp)<=minn:
                minn = abs(total-2*tmp)
                q.put((ind+1,tmp))

    return minn


##################################################################
# 4 Smallest Rectangle
#
###################################################################
from queue import Queue
import copy
def smallestRectangle(n, k, arr):
    def isOverlap(rect1, rect2):
        if len(rect1)==0 or len(rect2)==0:
            return False
        if max(rect1[0],rect2[0])>min(rect1[2],rect2[2]):
            return False
        else:
            if max(rect1[1],rect2[1])>min(rect1[3],rect2[3]):
                return False
            else:
                return True

    if n <= 1 or k>=n:
        return 0
    if k == 1:
        x_min = y_min = float('inf')
        x_max = y_max = -float('inf')
        for x, y in arr:
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
        return (y_max - y_min)*(x_max - x_min)
    else:
        ans = float('inf')
        arr.sort()
        q = Queue()

        tmp = [[] for _ in range(k)]
        tmp[0] = [arr[0][0],arr[0][1],arr[0][0],arr[0][1]]
        q.put((0,tmp))

        while not q.empty():
            ind, kRect = q.get()
            if ind == len(arr) - 1:
                area = 0
                for i in range(len(kRect)):
                    if len(kRect[i]) == 0:
                        continue
                    area += (kRect[i][2]-kRect[i][0])*(kRect[i][3]-kRect[i][1])
                ans = min(ans, area)
            else:
                for i in range(len(kRect)):
                    tmp = copy.deepcopy(kRect)
                    if len(tmp[i])==0:
                        tmp[i] = [arr[ind+1][0],arr[ind+1][1],arr[ind+1][0],arr[ind+1][1]]
                    else:
                        tmp[i][0] = min(tmp[i][0], arr[ind+1][0])
                        tmp[i][1] = min(tmp[i][1], arr[ind+1][1])
                        tmp[i][2] = max(tmp[i][2], arr[ind+1][0])
                        tmp[i][3] = max(tmp[i][3], arr[ind+1][1])

                    flag = False
                    for j in range(len(tmp)):
                        if i != j and isOverlap(tmp[i], tmp[j]):
                            flag = True
                    if not flag:
                        q.put((ind+1,tmp))
        return ans

#########################################################################################################################
#leetcode 902
#Numbers At Most N Given Digit Set
#########################################################################################################################
import math
def atMostNGivenDigitSet(self, digits, n):
    t = int(math.log10(n))
    l = len(digits)
    self.ans = int(l * (1 - pow(l, t)) / (1 - l)) if l > 1 else t

    def backTracking(i, val):
        if i >= t + 1:
            self.ans += 1
        else:
            for ch in digits:
                if ch == str(n)[i]:
                    v_incre = int(ch) * pow(10, t - i)
                    val += v_incre
                    backTracking(i + 1, val)
                    val -= v_incre
                elif ch < str(n)[i]:
                    self.ans += pow(l, t - i)

    backTracking(0, 0)
    return self.ans

################################################
#ByteDance minimum character transformation
#
###############################################
#from collections import defaultdict
def minTrans(source, target):
    map = {}
    for i in range(len(source)):
        if source[i] not in map:
            map[source[i]] = target[i]
        elif map[source[i]] != target[i]:
            return -1

    #newmap = {}
    for k, v in list(map.items()):
        if k == v:
            del map[k]
            #newmap[k] = v

    ans = 0
    flag = True
    while flag:
        flag = False
        for key, val in list(map.items()):
            if val not in map:
                flag = True
                ans += 1
                del map[key]

    while len(map)>0:
        s = set()
        tmp = list(map.items())[0]
        s.add(tmp[0])
        cur = tmp[1]
        del map[tmp[0]]
        while cur not in s:
            s.add(cur)
            t = cur
            cur = map[cur]
            del map[t]
        ans += len(s)+1

    return ans


#################################
#1
##################################
def commonEle(arr1, arr2):
    arr1.sort()
    arr2.sort()
    i = j = 0
    ans = []
    while i<len(arr1) and j<len(arr2):
        if arr1[i] == arr2[j]:
            ans.append(arr1[i])
            i+=1
            j+=1
        elif arr1[i]>arr2[j]:
            j+=1
        else:
            i+=1
    return ans

#####################################
#2

##########################################
class MyClass_1:
    def realGame(self, N, M, K):
        self.cnt = 0
        def backTracking(ind, s):
            if ind == K:
                if s >= N:
                    self.cnt += 1
            else:
                for i in range(M+1):
                    s += i
                    if s >= N:
                        self.cnt += pow(1+M, K-ind-1)
                    else:
                        backTracking(ind+1, s)
                    s -= i
        backTracking(0,0)
        return self.cnt/pow(1+M,K)

#################################################################
#3

##################################################################
def memoLeak(n, arr):
    def mleak(m1, m2):
        pre = [0, m1, m2]
        #print(pre)
        while True:
            if pre[1] >= pre[2]:
                if pre[0]+1>pre[1]:
                    return pre[0] + 1, pre[1], pre[2]
                else:
                    pre[0] += 1
                    pre[1] -= pre[0]
            else:
                if pre[0]+1>pre[2]:
                    return pre[0] + 1, pre[1], pre[2]
                else:
                    pre[0] += 1
                    pre[2] -= pre[0]
            #print(pre)

    for i in range(n):
        ans = mleak(arr[i][0],arr[i][1])
        print(" ".join([str(num) for num in ans]))


####################################################################################
#721. Accounts Merge
#
#####################################################################################
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        self.p[self.find(x)] = self.find(y)

import collections
class Solution:
    def accountsMerge(self, accounts):
        em_to_name = {}
        em_to_id = {}
        dsu = DSU(100001)

        i = 0
        for acc in accounts:
            name = acc[0]
            for email in acc[1:]:
                em_to_name[email] = name
                if email not in em_to_id:
                    em_to_id[email] = i
                    i += 1
                dsu.union(em_to_id[acc[1]], em_to_id[email])

        ans = collections.defaultdict(list)
        for email in em_to_name:
            ans[dsu.find(em_to_id[email])].append(email)

        return [[em_to_name[v[0]]] + sorted(v) for v in ans.values()]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #memoLeak(2,[[2,2],[8,11]])
    # c = MyClass_1()
    # print(c.realGame(2,1,3))
    #print(realGame(2,1,3))
    #print(minTrans("abcdef","badecg"))

    #print(smallestRectangle(4,2,[[1,1],[2,2],[3,6],[0,7]]))
    #print(minLastNumber([1,4,8]))#[2,7,4,1,8,1]
    # version1 = "1.0.0-alpha"
    # version2 = "1.0.0"
    # print(versionControl(version1,version2))

    #print_hi('PyCharm')
    test_list = [2,3,4,5,6,8,11,24,65,3,4,5,7765,986]
    print(test_list)

    heapSort(test_list, 0, len(test_list)-1)
    print(test_list)

    test_list_1 = [2, 3, 4, 5, 6, 8, 11, 24, 65, 3, 4, 5, 7765, 986]
    qsort(test_list_1, 0, len(test_list_1) - 1)
    print(test_list_1)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
