import importlib_resources
import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt

filename = importlib_resources.files("dtuimldmtools").joinpath("data/abalone.xls")
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# for the continuous attributes
attributeNames = doc.row_values(0, 0, 9)
print(attributeNames)
classLabels = doc.col_values(8, 1, 4177)
print(classLabels)
classNames = sorted(set(classLabels))
print(classNames)
classDict = dict(zip(classNames, range(29)))
print(classDict)

# for the nominal attribute sex
Z = np.asarray(doc.col_values(0, 1, 4177))
sex_types = sorted(set(doc.col_values(0, 1, 4177)))
print(sex_types)
counts_sex = np.unique(Z, return_counts=True)
values = counts_sex[0]
count_values = counts_sex[1]
labels = {'M': 'Male', 'F': 'Female', 'I': 'Infant'}
labels = [labels[val] for val in values]


y = np.asarray([classDict[value] for value in classLabels])

X = np.empty((4176, 8))
for i, col_id in enumerate(range(1, 9)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 4177))

N = len(y)
M = len(attributeNames)
C = len(classNames)
print(N, M, C)

df = pd.DataFrame(X, columns=attributeNames[1:])
summary_stats = df.describe()
summary_stats = summary_stats.drop(index="count")
print(summary_stats)

with open("summary_stats.tex", "w") as f:
    f.write(summary_stats.to_latex())

# plots


num_cols = 3
num_rows = 3
fig, axs = plt.subplots(num_rows, num_cols)

axs = axs.flatten()

for i in range(8):
    ax = axs[i+1]
    ax.hist(X[:, i], bins=20, color="lightblue", ec="black")
    ax.set_title('Histogram of Attribute ' + "\n" + attributeNames[i+1])
    ax.title.set_size(8)
    ax.set_xlabel('Value', fontsize=6)
    ax.set_ylabel('Frequency', fontsize=6)


ax_sex = axs[0]
ax_sex.bar(labels, count_values, color="lightblue", ec="black")
ax_sex.set_ylabel('Frequency', fontsize=6)
ax_sex.set_title('Histogram of Attribute ' + "\n" + "gender")
ax_sex.title.set_size(8)


plt.tight_layout()

plt.show()
