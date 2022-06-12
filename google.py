#import Exception
from collections import defaultdict
def tranf(A, B, C):
    if len(A) != len(C):
        raise ValueError("Error")

    d = defaultdict(list)
    for i in range(len(A)):
        d[A[i]].append(C[i])

    ans = []
    for item in B:
        if len(d[item]) > 0:
            ans.append(d[item][0])
            d[item].pop(0)
        else:
            raise ValueError("Error")
    return ans


def searchVal(arr, s, e, tar):
    if len(arr) == 0 or s == -1 or e == -1:
        return -1
    n = e - s + 1
    print(arr[s:e+1])
    i = s + n // 2
    print(arr[i])
    print(s,e,i)
    if arr[i] == " ":
        right = i
    else:
        while i < e + 1:
            if arr[i] != " ":
                i += 1
            else:
                break
        right = i
    #print(right)

    j = s + n // 2
    if arr[j] == " ":
        j -= 1
    while j >= s:
        if arr[j] != " ":
            j -= 1
        else:
            break
    left = j

    print(left, right)
    if left + 1 < right:
        tmp_num = int(arr[left + 1:right])
        print(tmp_num)
    else:
        return -1

    if tmp_num > tar:
        return searchVal(arr, s, left - 1, tar)
    elif tmp_num < tar:
        return searchVal(arr, right + 1, e, tar)
    else:
        return left + 1

if __name__=="__main__":
    # a = [3, 2, 1, 4, 5, 3]
    # b = [5, 3, 1, 2, 3, 4]
    # c = ['a', 'b', 'c', 'd', 'e', 'f']

    a = "12 35 60 200 10204 532566"
    print(searchVal(a,0,len(a)-1,60))
