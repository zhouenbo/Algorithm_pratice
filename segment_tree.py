import math

class SegmentTree:
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.seg = [0] * (2 * 2 ** math.ceil(math.log(self.n, 2)) - 1)
        self.build(0, 0, self.n - 1, nums)

    def build(self, treeIndex: int, l: int, r: int, nums: List[int]) -> None:
        if l == r:
            self.seg[treeIndex] = nums[l]
            return

        mid = l + (r - l) // 2
        self.build(2 * treeIndex + 1, l, mid, nums)
        self.build(2 * treeIndex + 2, mid + 1, r, nums)
        self.seg[treeIndex] = self.seg[2 * treeIndex + 1] + self.seg[2 * treeIndex + 2]

    def update(self, treeIndex: int, l: int, r: int, index: int, val: int) -> None:
        if l == r and l == index:
            self.seg[treeIndex] = val
            return

        if l > index or r < index:
            return

        mid = l + (r - l) // 2
        if index > mid:
            self.update(2 * treeIndex + 2, mid + 1, r, index, val)
        else:
            self.update(2 * treeIndex + 1, l, mid, index, val)
        self.seg[treeIndex] = self.seg[2 * treeIndex + 1] + self.seg[2 * treeIndex + 2]

    def rangeQuery(self, treeIndex, l, r, left, right):
        if l >= left and r <= right:
            return self.seg[treeIndex]

        if l > right or r < left:
            return 0

        mid = l + (r - l) // 2
        if mid < left:
            return self.rangeQuery(2 * treeIndex + 2, mid + 1, r, left, right)
        if mid >= right:
            return self.rangeQuery(2 * treeIndex + 1, l, mid, left, right)

        return self.rangeQuery(2 * treeIndex + 1, l, mid, left, right) + self.rangeQuery(2 * treeIndex + 2, mid + 1, r,
                                                                                         left, right)




class SegmentTreeMax:
    def __init__(self, arr):
        self.n = len(arr)
        l = 1
        while l < self.n:
            l *= 2
        self.st = [0]*(2*l-1)
        self.build(0, 0, self.n-1, arr)

    def build(self, treeIndex, l, r, arr):
        if l == r:
            self.st[treeIndex] = arr[l]
            return

        mid = l + (r - l)//2
        self.build(2*treeIndex+1, l, mid, arr)
        self.build(2 * treeIndex + 2, mid+1, r, arr)
        self.st[treeIndex] = max(self.st[2*treeIndex+1], self.st[2*treeIndex+2])

    def update(self, treeIndex, l, r, index, val):
        if l==index and l == r:
            self.st[treeIndex] = val
            return

        if index<l or index>r:
            return

        mid = l + (r-l)//2
        if mid>=index:
            self.update(2*treeIndex+1, l, mid, index, val)
        else:
            self.update(2*treeIndex+2, mid+1, r, index, val)
        self.st[treeIndex] = max(self.st[2*treeIndex+1], self.st[2*treeIndex+2])

    def rangeQuery(self, treeIndex, l, r, left, right):
        if l>right or r<left:
            return float('-inf')

        if l>=left and r<=right:
            return self.st[treeIndex]

        mid = l + (r-l)//2
        if mid>=right:
            return self.rangeQuery(2*treeIndex, l, mid, left, right)
        elif mid<left:
            return self.rangeQuery(2*treeIndex, mid+1, r, left, right)

        return max(self.rangeQuery(2*treeIndex+1, l, mid, left, mid), self.rangeQuery((2*treeIndex+2, mid+1, r, mid+1, r)))

