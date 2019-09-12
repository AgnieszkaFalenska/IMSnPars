import cyEisnerO2g
import numpy as np

l = [ ]
nxt = 0.0
allElem = [] 
for i in range(5):
    row = [ ]
    for j in range(5):
        rowrow = [ ]
        for k in range(5):
            rowrow.append(nxt)
            allElem.append(nxt)
            nxt += 1.0
        row.append(rowrow)
    l.append(row)


print(l)
print(allElem)

print(cyEisnerO2g.decodeProjective(5, np.array(allElem)))


