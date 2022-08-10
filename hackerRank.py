#--------------------------------------------------flipping bits -----------------------------------------------------------
def numToBin(n):
    res=[0]*32
    i=-1
    while n>=1 and i>=-32:
        res[i]=n%2
        n=n//2
        i-=1
    return res
def flippingBits(n):
    binaryStr=numToBin(n)
    for i in range(len(binaryStr)):
        if binaryStr[i]==0:
            binaryStr[i]=1
        else: binaryStr[i]=0
    strBin=''.join([str(e) for e in binaryStr])
    return int(strBin,2)
# print(flippingBits(8))

#--------------------------------------------------diagonal difference-----------------------------------------------------------
def diagonalDifference(arr):
    primaryDiagonal=0
    for i in range(len(arr)):
        primaryDiagonal+=arr[i][i]
    j=len(arr)-1
    secondaryDiagonal=0
    for i in range(len(arr)):
        secondaryDiagonal+=arr[i][j]
        j-=1
    return abs(primaryDiagonal-secondaryDiagonal)
#--------------------------------------------------Counting Sort-----------------------------------------------------------
def countingSort(arr):
    minEle=min(arr)
    rangeOfElement=max(arr)-minEle+1
    countArr=[0 for _ in range(rangeOfElement)]
    outputArr=[0 for _ in range(len(arr))]
    for i in range(len(arr)):
        countArr[arr[i]-minEle]+=1
    for i in range(1,len(countArr)):
        countArr[i]+=countArr[i-1]
    for i in range(len(arr)):
        outputArr[countArr[arr[i]-minEle]-1]=arr[i]
        countArr[arr[i]-minEle]-=1
    for i in range(len(arr)):
        arr[i]=outputArr[i]
    return arr
# print(countingSort([-5, -10, 0, -3, 8, 5, -1, 10]))
#--------------------------------------------------Pangram-----------------------------------------------------------
def pangrams(s):
    strSet=set(s.lower())
    if ' ' in strSet:
        strSet.remove(' ')
    if len(set('abcdefghijklmnopqrstuvwxyz'))-len(strSet)==0:
        return "pangram"
    return "not pangram" 

# print(pangrams("We promptly judged antique ivory buckles for the next prize"))
#--------------------------------------------------Permutation of array-----------------------------------------------------------
# A'[i] +B'[i]>=k
def twoArrays(k, A, B):
    # Write your code here
    A.sort()
    B.sort(reverse=True)
    for i  in range(len(A)):
        if A[i]+B[i]<k:
            return "No"
    return "Yes"
# print(twoArrays(10,[2, 1, 3],[7, 8, 9]))
#--------------------------------------------------SubArray Division-----------------------------------------------------------
def birthday(s, d, m):
    numWays=0
    for i in range(len(s)):
        currSum=sum(s[i:i+m])
        if currSum==d:
            numWays+=1
    return numWays
# print(birthday([2,2,1,3,2],4,2))
#--------------------------------------------------Sale By Match-----------------------------------------------------------
def sockMerchant(n, ar):
    hm={}
    for e in ar:
        if e not in hm:
            hm[e]=1
        else:
            hm[e]+=1
    numPair=0
    for key,val in hm.items():
        numPair+=val//2
    return numPair
# print(sockMerchant(6,[10, 20, 20, 10, 10, 30, 50, 10, 20]))

#--------------------------------------------------Counter Game-----------------------------------------------------------



def getPowerOfTwo(v):
    v-=1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v+=1
    return v
def isPowerOfTwo(x):
    return (x != 0) and ((x & (x - 1)) == 0)

def counterGame(n):
    turns=0
    while n>1:
        if isPowerOfTwo(n):
            n//=2
        else:
            p2=getPowerOfTwo(n)//2
            n-=p2
        turns+=1
    if turns%2==0:
        print("Louise")
    else:
        print("Richard")
print(counterGame(6))