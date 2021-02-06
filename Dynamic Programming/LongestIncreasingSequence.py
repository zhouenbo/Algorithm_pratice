# 354. Russian Doll Envelopes
# You have a number of envelopes with widths and heights given as a pair of integers (w, h). One envelope can fit into another if and only if both the width and height of one envelope is greater than the width and height of the other envelope.
#
# What is the maximum number of envelopes can you Russian doll? (put one inside other)
#
# Note:
# Rotation is not allowed.
#
# Example:
#
# Input: [[5,4],[6,4],[6,7],[2,3]]
# Output: 3
# Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).

##########################################################################################
# 转化成LIS问题
# LIS 的两种递归思考
# 一种类似于单调栈
# 二分法查找
# bisect.bisect_left函数的使用 bisect.bisect_left(a, x, lo=0, hi=len(a))
##########################################################################################



# 300. Longest Increasing Subsequence

https://www.geeksforgeeks.org/topological-sorting/
https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/
https://www.1point3acres.com/bbs/thread-700525-1-1.html
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=700840&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=691027&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26sortid%3D311


logo
Explore
Problems
Mock
Contest
Discuss
Store

Premium
0
5639.
Find
Minimum
Time
to
Finish
All
Jobs
User
Accepted: 339
User
Tried: 1342
Total
Accepted: 383
Total
Submissions: 2835
Difficulty: Hard
You
are
given
an
integer
array
jobs, where
jobs[i] is the
amount
of
time
it
takes
to
complete
the
ith
job.

There
are
k
workers
that
you
can
assign
jobs
to.Each
job
should
be
assigned
to
exactly
one
worker.The
working
time
of
a
worker is the
sum
of
the
time
it
takes
to
complete
all
jobs
assigned
to
them.Your
goal is to
devise
an
optimal
assignment
such
that
the
maximum
working
time
of
any
worker is minimized.

Return
the
minimum
possible
maximum
working
time
of
any
assignment.

Example
1:

Input: jobs = [3, 2, 3], k = 3
Output: 3
Explanation: By
assigning
each
person
one
job, the
maximum
time is 3.
Example
2:

Input: jobs = [1, 2, 4, 7, 8], k = 2
Output: 11
Explanation: Assign
the
jobs
the
following
way:
Worker
1: 1, 2, 8(working
time = 1 + 2 + 8 = 11)
Worker
2: 4, 7(working
time = 4 + 7 = 11)
The
maximum
working
time is 11.

Constraints:

1 <= k <= jobs.length <= 12
1 <= jobs[i] <= 107
Python3
1


class Solution:
    2


def minimumTimeRequired(self, jobs: List[int], k: int) -> int:


    3

Custom
Testcase
Copyright © 2021
LeetCode
Help
Center
Jobs
Bug
Bounty
Students
Terms
Privacy
Policy
United
StatesUnited
States