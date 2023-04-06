from itertools import permutations
  
# Get all combination of [1, 2, 3]
# of length 3
comb = permutations([5,3,3], 3)
  
for i in comb:
    print(i)