import time

s = time.time()
h = 0
for i in range(10000):
    h += i
print(time.time() - s)