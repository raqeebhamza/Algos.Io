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

from array import ArrayType
from operator import contains
from platform import node
from sre_constants import JUMP
import turtle


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

#Q.11 trapwater rain...........................................................................
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

#Q.12 levenshteinDistance........................................................................
# def levenshteinDistance(str1,str2): # O(nm) Time | O(nm) space
#     edits=[[ x for x in range(len(str1)+1)] for y in range(len(str2)+1)] 
#     for i in range(1,len(str2)+1):
#         edits[i][0]=edits[i-1][0]+1
#     for i in range(1,len(str2)+1):
#         for j in range(1,len(str1)+1):
#             if str1[j-1] == str2[i-1]:
#                 edits[i][j]=edits[i-1][j-1]
#             else:
#                 edits[i][j]=1+min(edits[i][j-1],edits[i-1][j],edits[i-1][j-1]) 
#     return edits[-1][-1]
# 2nd solution with better space complexity------------------------------------------------------------
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

#Q.14 RiverSizes ---------------------------------------------------------------------
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

#Q.15 Apartment Hunting ----------------------------------------------------------------
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


#Q.16 Cycle in Graph -------------------------------------------------------------------
# def cycleInGraph(edges): # O(v+e) Time | O(v) space
#     numberOfNodes=len(edges)
#     nodesInStack=[False for _ in range(numberOfNodes)]
#     isVisited=[False for _ in range(numberOfNodes)]
#     for node in range(numberOfNodes):
#         if isVisited[node]:
#             continue
#         containsCycle=isNodeContainsCycle(edges,node,isVisited,nodesInStack)
#         if containsCycle:
#             return True
#     return False
# def isNodeContainsCycle(edges,node,isVisited,nodesInStack):
#     nodesInStack[node]=True
#     isVisited[node]=True
#     neigbours=edges[node]
#     for neighbor in neigbours:
#         if not isVisited[neighbor]:
#             containsCycle=isNodeContainsCycle(edges,neighbor,isVisited,nodesInStack)
#             if containsCycle:
#                 return True
#         elif nodesInStack[neighbor]:
#             return True   
#     nodesInStack[node]=False         
#     return False
#Solution:2.....................................................................
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

#Q.16 Loop in the linkedList -----------------------------------------------------------
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

#Q.17 Depth first Search questions--------------------------------------------------------------------------------------------
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
#Q18: single cycle check --------------------------------------------------------------------------------------------
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
#Q19: --------------------------------------------------------------------------------------------
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

#Q20: --------------------------------------------------------------------------------------------
def minNumberOfCoinsForChange(n,denoms): #O(nd) time | O(n) space
    numOfCoins=[float("inf") for  amount in range(n+1)]
    numOfCoins[0]=0
    for denom in denoms:
        for amount  in range(len(numOfCoins)):
            if denom<=amount:
                numOfCoins[amount]=min(numOfCoins[amount],1+numOfCoins[amount-denom])
    return numOfCoins[n] if numOfCoins[n]!=float("inf") else -1

#Interviews Questions--------------------------------------------------------------------

# Zalando Questions---------------------------------------------------------------------
# B=['X.....>', '..v..X.', '.>..X..', 'A......']
# B=['...', '>.A']
# B=['A.v', '...']
def AssasinVsGuards(B):
    matrix=[]
    for i in range(0,len(B)):
        matrix.append(list(B[i]))   
    up='^'
    down='v'
    right='>'
    left='<'
    ai,aj=0,0
    visited=[[False for value in row] for row in matrix]
    for i in range(0,len(matrix)):
        for j in range(0,len(matrix[i])):
            if matrix[i][j]=='A':
                ai=i
                aj=j
            if matrix[i][j]==up:
                for k in reversed(range(0,i-1)):
                    if matrix[k][j]=='.':
                        matrix[k][j]=up
                    else:
                        break
            if matrix[i][j]==down:
                for k in range(i+1,len(matrix)):
                    if matrix[k][j]=='.':
                        matrix[k][j]=down
                    else:
                        break
            if matrix[i][j]==right:
                for k in range(j+1,len(matrix[i])):
                    if matrix[i][k]=='.':
                        matrix[i][k]=right
                    else:
                        break
    q=[[ai,aj]]
    if matrix[len(matrix)-1][len(matrix[0])-2]==right:
        return False
    if matrix[len(matrix)-2][len(matrix[0])-1]==down:
        return False    
    count=1
    while len(q):
        curr=q.pop(0)
        visited[curr[0]][curr[1]]=True
        if curr[0]==len(matrix)-1 and curr[1]==len(matrix[0])-1:
            return True
        neigbors=getneighbors(curr[0],curr[1],matrix,visited)
        for n in neigbors:    
            q.append(n)  
    return False    

def getneighbors(i,j,matrix,visited):
    neigbors=[]
    if i!=0:  # up neighbor
        if matrix[i-1][j]=='.' and visited[i-1][j]:
            neigbors.append([i-1,j])
    if j!=0: # left neighbor
        if  matrix[i][j-1]=='.' and  visited[i][j-1]: 
            neigbors.append([i,j-1])
    if i!=len(matrix)-1: # down neighbor
        if  matrix[i+1][j]=='.' and visited[i+1][j]:
            neigbors.append([i+1,j])
    if j!= len(matrix[i])-1: # right neighbor
        if  matrix[i][j+1]=='.' and visited[i][j+1]:        
            neigbors.append([i,j+1])
    return neigbors



#---------------------------------------------function calling--------------------------------------------------------

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

# print(knapsackProblem([
#   [1, 2],
#   [4, 3],
#   [5, 6],
#   [6, 7]
# ],10))

# print(trap([4,2,0,3,2,5]))

# print(levenshteinDistance("abc","yabd"))

# print(riverSizes([
#   [1, 0, 0, 1, 0],
#   [1, 0, 1, 0, 0],
#   [0, 0, 1, 0, 1],
#   [1, 0, 1, 0, 1],
#   [1, 0, 1, 1, 0]
# ]))
 
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


# print(cycleInGraph([
#   [1, 3],
#   [2, 3, 4],
#   [0],
#   [],
#   [2, 5],
#   []
# ]))

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

# print(minNumberOfCoinsForChange(7,[1, 5, 10]))




# Interview Question Calls
# print(AssasinVsGuards(['X.....>', '..v..X.', '....X..', 'A......']))               
