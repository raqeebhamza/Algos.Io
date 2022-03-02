#``````````````````````````````````````````----------------Algo Expert-------------------``````````````````````````````````````````

#Q.1: Two Number SUM -----------------------------------------------------------------------------------------------
#solution:1
# def twoNumberSum(arr,targetSum): #O(n^2) time | O(1) space
#     for i in range(len(arr)-1):
#         firstNum=arr[i]
#         for j in range(i+1,len(arr)):
#             secondNum=arr[j]
#             if firstNum+secondNum==targetSum:
#                 return [firstNum,secondNum]
#     return []

#solution:2 (fastest solution) by using hashmaps
# def twoNumberSum(arr,targetSum): # O(n) time | O(n) space   
#     hashMapOfNum={}
#     for num in arr:
#         if targetSum-num in hashMapOfNum:
#             return [num,targetSum-num]
#         else:
#             hashMapOfNum[num]=True
#     return []

#solution:3  using l pointer and R pointer

def twoNumberSum(arr,targetSum): # O(nlogn) time | O(1) space
    arr.sort()
    left=0
    right=len(arr)-1
    while left<right:
        currSum=arr[left]+arr[right]
        if currSum==targetSum:
            return [arr[left],arr[right]]
        elif currSum<targetSum:
            left+=1
        elif currSum>targetSum:
            right-=1
    return []                

#Q.2: Validate SubSequence -----------------------------------------------------------------------------------------------
#solution:1 using while loop
# def validateSubSequence(array,sequence): # O(nlogn) time | O(1) space
#     arrIdx=0
#     seqIdx=0
#     while arrIdx <len(array) and seqIdx<len(sequence):
#         if array[arrIdx]==sequence[seqIdx]:
#             seqIdx+=1
#         arrIdx+=1
#     return seqIdx==len(sequence) 
       
#solution:2 using for loop
def validateSubSequence(array,sequence): # O(nlogn) time | O(1) space
    seqIdx=0
    for value in array:
        if seqIdx==len(sequence):
            break
        if sequence[seqIdx]==value:
            seqIdx+=1
    return seqIdx==len(sequence)

#Q.3: SortedSquareArray -----------------------------------------------------------------------------------------------

#solution:1 using broot force
def sortedSquaredArray(array): # O(nlogn) time | O(n) space
    sortedSquares=[0 for _ in array]
    for idx in range(len(array)):
        value=array[idx]
        sortedSquares[idx]=value*value
    sortedSquares.sort()
    return sortedSquares    

#Q.4: fourNumbersum--------------------------------------------------------------------------------------
def fourNumberSum(array,targetSum): # O(n^2) time | O(n^2) space
    allPairs={}
    ansPairs=[]
    for i in range(1,len(array)-1):
        currNum=array[i]
        for c in range(i+1,len(array)):
            currSum=array[c]+currNum
            if (targetSum-currSum) in allPairs.keys():
                for e in allPairs[targetSum-currSum]:
                    ansPairs.append(e+[array[c],currNum])
        for j in range(0,i):
            pairSum=currNum+array[j]
            if pairSum in allPairs.keys():
                allPairs[pairSum].append([currNum,array[j]])
            else:
                allPairs[pairSum]=list()
                allPairs[pairSum].append([currNum,array[j]])    
    return ansPairs


#Q.5: SubArraySort find the unsort subarray indecies -----------------------------------------------------
def subArraySort(array): # O(n) time | O(1) space 
    minOutOfOrder=float("inf")
    maxOutOfOrder=float("-inf")
    for i in range(0,len(array)):
        currNum=array[i]
        if isOutOfOrder(i,currNum,array):
            minOutOfOrder=min(array[i],minOutOfOrder)
            maxOutOfOrder=max(array[i],maxOutOfOrder)        
    if minOutOfOrder == float("inf"):
        return [-1,-1]
    smallestIdx=0
    largertIdx=len(array)-1
    while array[smallestIdx]<= minOutOfOrder:
        smallestIdx+=1
    while(array[largertIdx] >= maxOutOfOrder):
        largertIdx-=1    
    return [smallestIdx,largertIdx]

def isOutOfOrder(i,num,array):
    if i==0:
        return num > array[i+1]
    elif  i==len(array)-1:
        return num < array[i-1]     
    return num<array[i-1] or num> array[i+1]

#Q.6: Largest Range present in the array -------------------------------------------------------------------
def largestRange(array): #O(N) time | O(N) space
    notVisited={}
    bestRange=[]
    maxLength=0
    for i in range(0,len(array)):
        notVisited[array[i]]=True
    for i in range(0,len(array)):
        num=array[i]
        if not notVisited[num]:
            continue
        currLength=1
        left=num-1
        right=num+1
        while left in notVisited:
            notVisited[left]=False
            currLength+=1
            left-=1
        while right in notVisited:
            notVisited[right]=False
            currLength+=1
            right+=1
        if currLength> maxLength:
            maxLength=currLength
            bestRange=[left+1,right-1] 
    return bestRange           
                
# Q.7: Min Rewards -----------------------------------------------------------------------------------------
def minRewards(scores):  # O(N) time | O(N) Space.....
    rewards=[1 for _ in scores]
    localMinIdxs=getlocalMinIdx(scores)
    for localMinIdx in localMinIdxs:
        expandFromLocalMin(localMinIdx,scores,rewards)
    return sum(rewards)    

def getlocalMinIdx(array):
    if len(array)==1:
        return [0]
    localMinIdx=[]
    for i in range(0,len(array)):
        if i==0 and array[i]<array[i+1]:
            localMinIdx.append(i)
        if i==len(array)-1  and array[i]<array[i-1]:
            localMinIdx.append(i)
        if i==0 or i==len(array)-1:
            continue       
        if array[i]<array[i+1] and array[i]<array[i-1]:
            localMinIdx.append(i)
    return localMinIdx    

def expandFromLocalMin(localMinIdx,scores,rewards):
    leftIdx=localMinIdx-1
    while leftIdx >=0 and scores[leftIdx]>scores[leftIdx+1]:
        rewards[leftIdx]=max(rewards[leftIdx],rewards[leftIdx+1]+1)
        leftIdx-=1
    rightIdx=localMinIdx+1
    while rightIdx <=len(scores)-1 and scores[rightIdx]>scores[rightIdx-1]:
        rewards[rightIdx]=rewards[rightIdx-1]+1
        rightIdx+=1

# Q.8: ZigZagTravers----------------------------------------------------------------------------------------
def zigZagTraverse(array): #O(N) time | O(N) space
    height=len(array)-1
    width=len(array[0])-1
    goingDown=True
    row,col=0,0
    result=[]
    while not isOutOfBound(row,col,height,width):
        result.append(array[row][col])
        if goingDown:
            if col==0 or row==height:
                goingDown=False
                if row==height:
                    col+=1
                else:
                    row+=1
            else:
                col-=1
                row+=1            
        else:    
            if row==0 or col==width:
                goingDown=True
                if col==width:
                    row+=1
                else:
                    col+=1
            else:
                row-=1
                col+=1            

    return result

def isOutOfBound(row,col,height,width):
    return row<0 or col<0 or col>width or row>height

#Q.9: SpiralTraverse---------------------------------------------------------------------------------------- 
def spiralTraverse(array): #O(n) time | O(n) space 
    result=[]
    startRow,endRow=0,len(array)-1
    startCol,endCol=0,len(array[0])-1
    while startRow<=endRow and startCol<=endCol:
        if startRow == endRow:
            for i in range(startCol,endCol+1):
                result.append(array[startRow][i])
            return result
        if endCol == startCol:
            for i in range(startRow,endRow+1):
                result.append(array[i][endCol])
            return result    
        for col in range(startCol,endCol+1):
            result.append(array[startRow][col])
        for row in range(startRow+1,endRow+1):
            result.append(array[row][endCol])   
        for col in reversed(range(startCol,endCol)):
            result.append(array[endRow][col])
        for row in reversed(range(startRow+1,endRow)):
            result.append(array[row][startCol])
        startRow+=1
        endRow-=1
        startCol+=1
        endCol-=1         
    return result        

#Q.10 sameBsts--------------------------------------------------------------------------------------------
def sameBsts(arrayOne,arrayTwo): # O(n^2) time | O(n^2) space
    if len(arrayOne)!=len(arrayTwo):
        return False
    if len(arrayOne)==0 and len(arrayTwo)==0:
        return True
    if arrayOne[0]!=arrayTwo[0]:
        return False 

    leftOne=getSmaller(arrayOne)
    leftTwo=getSmaller(arrayTwo)
    rightOne=getBiggerOrEqual(arrayOne)
    rightTwo=getBiggerOrEqual(arrayTwo)

    return sameBsts(leftOne,leftTwo) and sameBsts(rightOne,rightTwo)

def getSmaller(array):
    smaller=[]
    for i in range(1,len(array)):
        if array[i]< array[0]:
            smaller.append(array[i])
    return smaller        
def getBiggerOrEqual(array):
    biggerOrEqual=[]
    for i in range(1,len(array)):
        if array[i]>= array[0]:
            biggerOrEqual.append(array[i])
    return biggerOrEqual  

#Q.11 KnapSackProblem........................................................................
def knapsackProblem(items,capacity): # O(NC) time | O(NC) space  => N=number of items, C= total Capacity
    knapsackValues=[[0 for x in range(0,capacity+1)] for y in range(0,len(items)+1)]
    for i in range(1,len(items)+1):
        w=items[i-1][1]
        v=items[i-1][0]
        for c in range(0,capacity+1):
            if w>c:
                knapsackValues[i][c]=knapsackValues[i-1][c]
            else:
                knapsackValues[i][c]=max(knapsackValues[i-1][c],knapsackValues[i-1][c-w]+v)    
    return [knapsackValues[-1][-1],getKnapSackItems(knapsackValues,items)]

def getKnapSackItems(knapsackValues,items):
    sequence=[]
    i=len(knapsackValues)-1
    c=len(knapsackValues[0])-1
    while i>0:
        if knapsackValues[i][c]==knapsackValues[i-1][c]:
            i-=1
        else:
            sequence.append(i-1)
            c-=items[i-1][1]    
            i-=1
        if c <= 0:
            break
    return list(reversed(sequence))    


#----------------function calling---------------

# print(twoNumberSum([3,5,-4,8,11,1,-1,6],10))

# print(validateSubSequence([5, 1, 22, 25, 6, -1, 8,10],[1, 6, -1, 10]))

# print(sortedSquaredArray([1, 2, -7, 5, 6, 8, 9]))

# print(fourNumberSum([7, 6, 4, -1, 1, 2],16))

# print(subArraySort([1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19]))

# print(largestRange([1, 11, 3, 0, 15, 5, 2, 4, 10, 7, 12, 6]))

# print(minRewards([8, 4, 2, 1, 3, 6, 7, 9, 5]))

# print(zigZagTraverse([
#   [1, 3, 4, 10],
#   [2, 5, 9, 11],
#   [6, 8, 12, 15],
#   [7, 13, 14, 16]
# ]))

# print(spiralTraverse([
#   [1,  2,  3],
#   [12, 13, 4],
#   [11, 14, 5],
#   [10, 15, 6],
#   [9,  8,  7],
# ]))

# print(sameBsts([10, 8, 5, 15, 2, 12, 11, 94, 81],[10, 15, 8, 12, 94, 81, 5, 2, 11]))

print(knapsackProblem([
  [1, 2],
  [4, 3],
  [5, 6],
  [6, 7]
],10))
