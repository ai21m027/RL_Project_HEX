import matplotlib.pyplot as plt
import numpy as np

n = [0, 1, 2, 3]
v = [0, 0, 1, 1]

print('sorted then random choice')
for i in range(10):
    p = sorted(n, key=lambda x: v[x])
    print(np.random.choice(len(p), p=p / np.sum(p)), ',', end=' ')

print()

print('shuffled first')
for i in range(10):
    np.random.shuffle(n)
    p = sorted(n, key=lambda x: v[x])
    print(np.argmax(p), ',', end=' ')


# for i in range(len(n)):
#     for j in range(len(v)):

#         q = 0


#         q_values = []

#         for k in range(10):
#             q = (n[i] * q + v[j])  / (n[i] + 1)
#             q_values.append(q)


#         # plot the values

#         plt.scatter(x=list(range(10)), y=q_values)

#         plt.show()


#         print('')
