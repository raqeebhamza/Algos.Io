
from typing import List
#---------------------------------------------META Questions--------------------------------------------------------------

#------------------------------------Cafeteria Dinners Problem------------------------------------------------------------
def getMaxAdditionalDinersCount(N: int, K: int, M: int, S: List[int]) -> int: #O(Mlog(M))
                            #[1,2_,3,4,5,6_,7,8,9,10]
    dinners=0
    distance=K+1
    S.sort()
    dinners+=(S[0]-1)//distance # count from left most
    dinners+=(N-S[-1])//distance# count from right most
    for i in range(M-1):
        dinners+= (S[i+1]-S[i]-K-1)//distance # count between the already seated
    return dinners
# print(getMaxAdditionalDinersCount(10,1,2,[2,6]))

