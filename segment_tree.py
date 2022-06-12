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