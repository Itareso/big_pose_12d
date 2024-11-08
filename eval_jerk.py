import numpy as np
import sys
import matplotlib.pyplot as plt

f = sys.argv[1]

with open(f, 'r') as file:
    lines = file.readlines()

acc_all = []
jerk_all = []
for line in lines:
    acc = eval(line)
    acc_all.extend(acc)
    acc = np.array(acc)
    jerk_all.extend(list(np.diff(acc, axis=0) * 30))

acc_all = np.array(acc_all)
# jerk = np.diff(acc_all, axis=0) * 30
# jerk = np.linalg.norm(jerk, axis=1)
acc = np.linalg.norm(acc_all, axis=1)
jerk_all = np.array(jerk_all)
jerk = np.linalg.norm(jerk_all, axis=1)


plt.hist(jerk, bins=100)
plt.title("histogram of jerk")
plt.savefig(f"./jerk_gt.png")
plt.cla()
print("jerk:", max(jerk), min(jerk), np.mean(jerk), np.std(jerk))
plt.hist(acc, bins=100)
plt.title("histogram of acc")
plt.savefig(f"./acc_gt.png")
plt.cla()
print("acc:", max(acc), min(acc), np.mean(acc), np.std(acc))