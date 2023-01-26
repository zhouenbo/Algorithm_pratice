num = input()
for i in range(int(num)):
    s = input()
    if s[1] == 'a':
        print("{0} {1} {2}".format(s[0], s[1], s[2:]))
    else:
        print("{0} {1} {2}".format(s[0], s[1:-1], s[-1]))