#``````````````````````````````````````````----------------Algo Expert-------------------``````````````````````````````````````````

#Q.1:----------------------------------------------------------------Two Number SUM-----------------------------------------------------------
#solution:1
def twoNumberSum(arr,targetSum): #O(n^2) time | O(1) space
    for i in range(len(arr)-1):
        firstNum=arr[i]
        for j in range(i+1,len(arr)):
            secondNum=arr[j]
            if firstNum+secondNum==targetSum:
                return [firstNum,secondNum]
    return []

#solution:2 (fastest solution) by using hashmaps
def twoNumberSum(arr,targetSum): # O(n) time | O(n) space   
    hashMapOfNum={}
    for num in arr:
        if targetSum-num in hashMapOfNum:
            return [num,targetSum-num]
        else:
            hashMapOfNum[num]=True
    return []

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

# print(twoNumberSum([3,5,-4,8,11,1,-1,6],10))
#Q.2:-------------------------------------------------Validate SubSequence ------------------------------------------------------------------
#solution:1 using while loop
def validateSubSequence(array,sequence): # O(nlogn) time | O(1) space
    arrIdx=0
    seqIdx=0
    while arrIdx <len(array) and seqIdx<len(sequence):
        if array[arrIdx]==sequence[seqIdx]:
            seqIdx+=1
        arrIdx+=1
    return seqIdx==len(sequence) 
       
#solution:2 using for loop
def validateSubSequence(array,sequence): # O(nlogn) time | O(1) space
    seqIdx=0
    for value in array:
        if seqIdx==len(sequence):
            break
        if sequence[seqIdx]==value:
            seqIdx+=1
    return seqIdx==len(sequence)

# print(validateSubSequence([5, 1, 22, 25, 6, -1, 8,10],[1, 6, -1, 10]))
#Q.3:----------------------------------------SortedSquareArray-----------------------------------------------------------------------------------------

#solution:1 using broot force
def sortedSquaredArray(array): # O(nlogn) time | O(n) space
    sortedSquares=[0 for _ in array]
    for idx in range(len(array)):
        value=array[idx]
        sortedSquares[idx]=value*value
    sortedSquares.sort()
    return sortedSquares    

# print(sortedSquaredArray([1, 2, -7, 5, 6, 8, 9]))
#Q.4: fourNumbersum-------------------------------------------------------------------------------------------------------------------------
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


# print(fourNumberSum([7, 6, 4, -1, 1, 2],16))

#Q.5: ---------------------------------------------------SubArraySort find the unsort subarray indecies-------------------------------------
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

# print(subArraySort([1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19]))

#Q.6: ----------------------------------------Largest Range present in the array -----------------------------------------------------------
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

# print(largestRange([1, 11, 3, 0, 15, 5, 2, 4, 10, 7, 12, 6]))

# Q.7:---------------------------------------------------------Min Rewards------------------------------------------------------------------
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

# print(minRewards([8, 4, 2, 1, 3, 6, 7, 9, 5]))

# Q.8: -------------------------------------------------------ZigZagTravers-------------------------------------------------------------------
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


# print(zigZagTraverse([
#   [1, 3, 4, 10],
#   [2, 5, 9, 11],
#   [6, 8, 12, 15],
#   [7, 13, 14, 16]
# ]))
#Q.9:--------------------------------------------------------SpiralTraverse--------------------------------------------------------------------- 
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

# print(spiralTraverse([
#   [1,  2,  3],
#   [12, 13, 4],
#   [11, 14, 5],
#   [10, 15, 6],
#   [9,  8,  7],
# ]))


#Q.11------------------------------------------------------------KnapSackProblem--------------------------------------------------------------
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


# print(knapsackProblem([
#   [1, 2],
#   [4, 3],
#   [5, 6],
#   [6, 7]
# ],10))

#Q.11 -----------------------------------------------------------trapwater rain---------------------------------------------------------------------------
def trap(height): # O(N) time | O(1) space
    leftMax,rightMax=0,0
    left,right=0,len(height)-1
    trappedWater=0
    while left<right:
        if height[left] > leftMax:
            leftMax=height[left]
        if height[right]> rightMax:
            rightMax=height[right]
        if leftMax<rightMax:
            trappedWater+=max(0,leftMax-height[left])
            left+=1
        else:
            trappedWater+=max(0,rightMax-height[right])
            right-=1
    return trappedWater  

# print(trap([4,2,0,3,2,5]))
#Q.12 ----------------------------------------------------------levenshteinDistance----------------------------------------------------------
#1st Solution
def levenshteinDistance(str1,str2): # O(nm) Time | O(nm) space
    edits=[[ x for x in range(len(str1)+1)] for y in range(len(str2)+1)] 
    for i in range(1,len(str2)+1):
        edits[i][0]=edits[i-1][0]+1
    for i in range(1,len(str2)+1):
        for j in range(1,len(str1)+1):
            if str1[j-1] == str2[i-1]:
                edits[i][j]=edits[i-1][j-1]
            else:
                edits[i][j]=1+min(edits[i][j-1],edits[i-1][j],edits[i-1][j-1]) 
    return edits[-1][-1]
# 2nd solution 
def levenshteinDistance(str1,str2): # O(nm) time | O(min(n,m)) Space
    small=str1 if len(str1)<len(str2) else str2
    big=str1 if len(str1)>=len(str2) else str2
    evenEdits=[x for x in range(len(small)+1)]
    oddEdits=[None for  x in range(len(small)+1)]
    for i in range(1,len(big)+1):
        if i%2==1:
            currentEdits=oddEdits
            previousEdits=evenEdits
        else:
            currentEdits=evenEdits
            previousEdits=oddEdits
        currentEdits[0]=i 
        for j in range(1,len(small)+1):
            if big[i-1]==small[j-1]:
                currentEdits[j]=previousEdits[j-1]
            else:
                currentEdits[j]=1+min(currentEdits[j-1],previousEdits[j],previousEdits[j-1])
    return evenEdits[-1] if len(big)%2==0 else oddEdits[-1]

# print(levenshteinDistance("abc","yabd"))
#Q.14------------------------------------------------------------RiverSizes------------------------------------------------------------------
def riverSizes(matrix):  # O(wh) time | O(wh) space
    sizes=[]
    visited=[[False for value in row] for row in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if visited[i][j]:
                continue
            else:
                traverseNode(i,j,matrix,visited,sizes)
    return sizes
def traverseNode(i,j,matrix,visited,sizes):
    currnentSize=0
    nodesToExplore=[[i,j]] # stack
    while len(nodesToExplore):
        currnentNode=nodesToExplore.pop()
        i=currnentNode[0]
        j=currnentNode[1]
        if visited[i][j]:
            continue
        visited[i][j]=True
        if matrix[i][j]==0:
            continue
        currnentSize+=1
        unvisitedNeighbors=getUnVisitedNeighbors(i,j,matrix,visited)
        for neigbors in unvisitedNeighbors:
            nodesToExplore.append(neigbors)
    if currnentSize>0:
        sizes.append(currnentSize)

def getUnVisitedNeighbors(i,j,matrix,visited):
    neighbors=[]
    if i!=0:  # up neighbor
        if not visited[i-1][j]:
            neighbors.append([i-1,j])
    if j!=0: # left neighbor
        if not visited[i][j-1]: 
            neighbors.append([i,j-1])
    if i!=len(matrix)-1: # down neighbor
        if not visited[i+1][j]:
            neighbors.append([i+1,j])
    if j!= len(matrix[i])-1: # right neighbor
        if not visited[i][j+1]:        
            neighbors.append([i,j+1])
    return neighbors

# print(riverSizes([
#   [1, 0, 0, 1, 0],
#   [1, 0, 1, 0, 0],
#   [0, 0, 1, 0, 1],
#   [1, 0, 1, 0, 1],
#   [1, 0, 1, 1, 0]
# ]))
#Q.15--------------------------------------------------------------Apartment Hunting----------------------------------------------------------
def apartmentHunting(blocks,req):
    minDistancesFromBlocks=list(map(lambda req:getMinDistances(blocks,req),req))
    print(minDistancesFromBlocks)
    maxDistancesAtBlocks=getMaxDistanceAtBlocks(blocks,minDistancesFromBlocks)
    return getIdxOfMinValue(maxDistancesAtBlocks)
def getMaxDistanceAtBlocks(blocks,minDistanceFromBlocks):
    maxDistanceAtBlocks=[0 for block in blocks]
    for i in range(len(blocks)):
        minDistancesAtBlock=list(map(lambda distances:distances[i],minDistanceFromBlocks))
        maxDistanceAtBlocks[i]=max(minDistancesAtBlock)
    return maxDistanceAtBlocks
def getMinDistances(blocks,req):
    minDistances=[0 for block in blocks]
    closestReqIdx=float("inf")
    for i in range(len(blocks)):
        if blocks[i][req]:
            closestReqIdx=i
        minDistances[i]=distanceBetween(i,closestReqIdx)
    for i in reversed(range(0,len(blocks))):
        if blocks[i][req]:
            closestReqIdx=i
        minDistances[i]=min(minDistances[i],distanceBetween(i,closestReqIdx))    
    return minDistances
def distanceBetween(a,b):
    return abs(a-b)        
def getIdxOfMinValue(array):
    idxAtMinValue=0
    minValue=float("inf")
    for i in range(len(array)):
        currVal=array[i]
        if currVal<minValue:
            minValue=currVal
            idxAtMinValue=i
    return idxAtMinValue    

# print(apartmentHunting([
#   {
#     "gym":False ,
#     "school": True,
#     "store": False
#   },
#   {
#     "gym": True,
#     "school": False,
#     "store": False
#   },
#   {
#     "gym": True,
#     "school": True,
#     "store": False
#   },
#   {
#     "gym": False,
#     "school": True,
#     "store": False
#   },
#   {
#     "gym": False,
#     "school": True,
#     "store": True
#   }],["gym", "school", "store"]))
#Q.16 ------------------------------------------------------------Cycle in Graph -------------------------------------------------------------------
# Solution:1
def cycleInGraph(edges): # O(v+e) Time | O(v) space
    numberOfNodes=len(edges)
    nodesInStack=[False for _ in range(numberOfNodes)]
    isVisited=[False for _ in range(numberOfNodes)]
    for node in range(numberOfNodes):
        if isVisited[node]:
            continue
        containsCycle=isNodeContainsCycle(edges,node,isVisited,nodesInStack)
        if containsCycle:
            return True
    return False
def isNodeContainsCycle(edges,node,isVisited,nodesInStack):
    nodesInStack[node]=True
    isVisited[node]=True
    neigbours=edges[node]
    for neighbor in neigbours:
        if not isVisited[neighbor]:
            containsCycle=isNodeContainsCycle(edges,neighbor,isVisited,nodesInStack)
            if containsCycle:
                return True
        elif nodesInStack[neighbor]:
            return True   
    nodesInStack[node]=False         
    return False
# Solution:2
WHITE, GREY,BLACK=0,1,2
def cycleInGraph(edges): # O(v+e) Time | O(v) space 
    numberOfNodes=len(edges)
    colors=[WHITE for _ in range(numberOfNodes)]
    for node in range(numberOfNodes):
        if colors[node]!=WHITE:
            continue
        containsCycle=traverseAndColorNode(edges,node,colors)
        if containsCycle:
            return True
    return False
def traverseAndColorNode(edges,node,colors):
    colors[node]=GREY
    neighbors=edges[node]
    for neighbor in neighbors:
        if colors[neighbor]==GREY:
            return True
        if colors[neighbor]!=WHITE:
            continue
        containsCycle=traverseAndColorNode(edges,neighbor,colors)
        if containsCycle:
            return True
    colors[node]=BLACK
    return False        


# print(cycleInGraph([
#   [1, 3],
#   [2, 3, 4],
#   [0],
#   [],
#   [2, 5],
#   []
# ]))

#Q.16-------------------------------------------------------------Loop in the linkedList------------------------------------------------------
class LinkedList:
    def __init__(self,value) -> None:
        self.value=value
        self.next=None
def findLoop(head:LinkedList): # Find loop and return loop head of the linkedList
    if not head:
        return head
    first=head.next
    second=head.next.next
    while first!=second:
        first=first.next
        second=second.next.next
    first=head
    while first!=second:
        first=first.next
        second=second.next
    return first
# print(findLoop({
#   "head": "0",
#   "nodes": [
#     {"id": "0", "next": "1", "value": 0},
#     {"id": "1", "next": "2", "value": 1},
#     {"id": "2", "next": "3", "value": 2},
#     {"id": "3", "next": "4", "value": 3},
#     {"id": "4", "next": "5", "value": 4},
#     {"id": "5", "next": "6", "value": 5},
#     {"id": "6", "next": "7", "value": 6},
#     {"id": "7", "next": "8", "value": 7},
#     {"id": "8", "next": "9", "value": 8}, 
#     {"id": "9", "next": "4", "value": 9}
#   ]
# }))
#Q.17 ----------------------------------------------------------Depth first Search questions--------------------------------------------------------------------------------------------
class Node:       #O(v+e) Time | O(v) space
    def __init__(self, name):
        self.children = []
        self.name = name

    def addChild(self, name):
        self.children.append(Node(name))
        return self

    def depthFirstSearch(self, array):
        array.append(self.name)
        for child in self.children:
            child.depthFirstSearch(array)
        return array

#Q18:------------------------------------------------------------single cycle check----------------------------------------------------------
def hasSingleCycle(array): #O(N) Time | O(1) space
    numOfElementvisited=0
    currIdx=0
    while numOfElementvisited<len(array):
        if numOfElementvisited>0 and currIdx==0:
            return False
        numOfElementvisited+=1
        currIdx=getNextIdx(currIdx,array)
    return currIdx==0
def getNextIdx(currIdx,array):
    jump=array[currIdx]
    nextIdx=(currIdx+jump)%len(array)    
    return nextIdx if nextIdx>=0 else nextIdx+len(array)

#Q19: ----------------------------------------------------------YoungestCommonAncestor--------------------------------------------------------------------------------------------
def getYoungestCommonAncestor(root,descendantOne,descendantTwo): #O(d) time  | O(1) space
    depthOne=getDescendantDepth(descendantOne,root)
    depthTwo=getDescendantDepth(descendantTwo,root)
    if depthOne>depthTwo:
        return backtrackAncestralTree(descendantOne,descendantTwo,depthOne-depthTwo)
    else:
        return backtrackAncestralTree(descendantTwo,descendantOne,depthTwo-depthOne)
def getDescendantDepth(descendant,root):
    depth=0
    while descendant!=root:
        depth+=1
        descendant=descendant.ancestor
    return depth
def backtrackAncestralTree(lowerDescendant,higherDescendant,diff):
    while diff>0:
        lowerDescendant=lowerDescendant.ancestor
        diff-=1
    while lowerDescendant!=higherDescendant:
        lowerDescendant=lowerDescendant.ancestor
        higherDescendant=higherDescendant.ancestor
    return lowerDescendant    

#Q20: --------------------------------------------------------minNumberOfCoinsForChange--------------------------------------------------------------------------------------------
def minNumberOfCoinsForChange(n,denoms): #O(nd) time | O(n) space
    numOfCoins=[float("inf") for  amount in range(n+1)]
    numOfCoins[0]=0
    for denom in denoms:
        for amount  in range(len(numOfCoins)):
            if denom<=amount:
                numOfCoins[amount]=min(numOfCoins[amount],1+numOfCoins[amount-denom])
    return numOfCoins[n] if numOfCoins[n]!=float("inf") else -1

# print(minNumberOfCoinsForChange(7,[1, 5, 10]))

#Q21: -----------------------------------------------------------threeNumberSum--------------------------------------------------------------------------------------------
def threeNumberSum(nums,target): #O(n^2) time | O(n) space
    nums.sort()
    res=[]
    for i in range(len(nums)):
        l=i+1
        r=len(nums)-1
        target2=target-nums[i]
        while l<r:
            if nums[l]+nums[r]<target2:
                l+=1
            elif nums[l]+nums[r]>target2:
                r-=1
            else:
                res.append([nums[i],nums[l],nums[r]])
                r-=1
                l+=1
    return res
# print(threeNumberSum([12, 3, 1, 2, -6, 5, -8, 6],0))
#Q22: ----------------------------------------------------smallestDifference---------------------------------------------------------
def smallestDifference(arrayOne, arrayTwo): #O(nlogn)+O(mlogm) time | O(1) space
    arrayOne.sort()
    arrayTwo.sort()
    print(arrayOne)
    print(arrayTwo)
    i=j=0
    ans=[arrayOne[0],arrayTwo[0]]
    while i<len(arrayOne) and j<len(arrayTwo):
        if abs(arrayOne[i]-arrayTwo[j])<abs(ans[0]-ans[1]):
            ans=arrayOne[i],arrayTwo[j]
        if arrayOne[i]>arrayTwo[j]:
            j+=1
        else:
            i+=1
    return ans
# print(smallestDifference([-1, 5, 10, 20, 28, 3],[26, 134, 135, 15, 17]))

#Q23: --------------------------------------------------moveElementToEnd------------------------------------------------------------------
def moveElementToEnd(array, toMove): #O(N) time | O(1) space
    i,j=0,len(array)-1
    while i<j:
        while i<j and array[j]==toMove:
            j-=1
        if array[i]==toMove:
            array[i],array[j]=array[j],array[i]
        i+=1
    return array
# print(moveElementToEnd([2, 1, 2, 2, 2, 3, 4, 2],2))

#Q24: ----------------------------------------------------isMonotonic-----------------------------------------------------------------------
def isMonotonic(array): #O(N) time | O(1) space
    isNonDecreasing=True
    isNonIncreasing=True
    for i in range(1,len(array)):
        if array[i]<array[i-1]:
            isNonDecreasing=False
        if array[i]>array[i-1]:
            isNonIncreasing=False
    return isNonIncreasing or isNonDecreasing
# print(isMonotonic([-1, -5, -10, -1100, -1100, -1101, -1102, -9001]))

#Q25: ---------------------------------------------------longestPeak-----------------------------------------------------------------------   
def longestPeak(array): #O(n) time | O(1) space
    i=1
    longestPeakLength=0
    while i<len(array)-1:
        isPeak=array[i-1]<array[i] and array[i]>array[i+1]
        if not isPeak:
            i+=1
            continue
        leftIdx=i-2
        while leftIdx>=0 and array[leftIdx]<array[leftIdx+1]:
            leftIdx-=1
        rightIdx=i+2
        while rightIdx<len(array) and array[rightIdx]<array[rightIdx-1]:
            rightIdx+=1
        currPeakLen=rightIdx-leftIdx-1
        longestPeakLength=max(currPeakLen,longestPeakLength)
        i=rightIdx
    return longestPeakLength

# print(longestPeak([1, 2, 3, 3, 4, 0, 10, 6, 5, -1, -3, 2, 3]))
#Q26: --------------------------------------arrayOfProducts without using division----------------------------------------------------------------------   
def arrayOfProducts(array): #O(N) time | O(N) space
    right,left=[1]*len(array),[1]*len(array)
    currProd=1
    for i in range(len(array)):
        left[i]=currProd
        currProd*=array[i]
    currProd=1
    print(left)
    for i in range(len(array)-1,-1,-1):
        right[i]=currProd
        currProd*=array[i]
    
    print(right)
    for i in range(len(right)):
        left[i]*=right[i]
    return left
# print(arrayOfProducts([5, 1, 4, 2]))
#Q28: --------------------------------------------------firstDuplicateValue-------------------------------------------   
def firstDuplicateValue(array): #O(N) time | O(1) space
    for value in array:
        absVal=abs(value)
        if array[absVal-1]<0:
            return absVal
        array[absVal-1]*=-1
    return -1

#Q28: --------------------------------------------------mergeOverlappingIntervals--------------------------------------------   
def mergeOverlappingIntervals(intervals): #O(nlogn) time | O(n) space
    intervals.sort(key= lambda x: x[0])
    mergedIntevals=[]
    for interval in intervals:
        if len(mergedIntevals)==0:
            mergedIntevals.append(interval)
        else:
            if mergedIntevals[-1][1]>=interval[0]:
                mergedIntevals[-1][1]=max(mergedIntevals[-1][1],interval[1])
            else:
                mergedIntevals.append(interval)
    return mergedIntevals




















#---------------------------------------------function calling-------------------------------------------------------




# print(tournamentWinner([
#   ["HTML", "C#"],
#   ["C#", "Python"],
#   ["Python", "HTML"]
# ], [0, 0, 1]))

# print(nonConstructibleChange([5, 7, 1, 1, 2, 3, 22]))



