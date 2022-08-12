from syslog import closelog


class BST:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
#Question---------------------------------------------------------sameBsts----------------------------------------------------------------------
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


# print(sameBsts([10, 8, 5, 15, 2, 12, 11, 94, 81],[10, 15, 8, 12, 94, 81, 5, 2, 11]))

#Question----------------------------------------------------findClosestValueInBst----------------------------------------------------------------------
#Solution => recursive
def findClosestValueInBst(tree, target):
    return findClosestValueInBstHelper(tree,target,float("inf"))
def findClosestValueInBstHelper(node,target,closest):
    if not node:
        return closest
    if abs(target-closest)>abs(target-node.value):
        closest=node.value
    if target<node.value:
        return findClosestValueInBstHelper(node.left,target,closest)
    elif target>node.value:
        return findClosestValueInBstHelper(node.right,target,closest)
    else:
        return closest

#Solution => iterative
def findClosestValueInBst(tree, target):
    curr=tree
    closest=float("inf")
    while curr:
        if abs(target-closest)>abs(target-curr.value):
            closest=curr.value
        if target<curr.value:
            curr=curr.left
        elif target>curr.value:
            curr=curr.right
        else:
            break
    return closest

#Question----------------------------------------------------validateBst----------------------------------------------------------------------
def validateBst(tree): #O(n) time | O(1) space
    return validateBstHelper(tree,float("-inf"),float("inf"))

def validateBstHelper(node,left,right):
    if not node:
        return True
    if not left<=node.value<right:
        return False
    return validateBstHelper(node.left,left,node.value) and validateBstHelper(node.right,node.value,right) 

#Question----------------------------------------------------BST traversal----------------------------------------------------------------------
#Solution Recursive-----------
def inOrderTraverse(tree, array):
    if tree.left:
        inOrderTraverse(tree.left,array)
    array.append(tree.value)
    if tree.right:
        inOrderTraverse(tree.right,array)
    return array

def preOrderTraverse(tree, array):
    array.append(tree.value)
    if tree.left:
        preOrderTraverse(tree.left,array)
    if tree.right:
        preOrderTraverse(tree.right,array)
    return array

def postOrderTraverse(tree, array):
    if tree.left:
        postOrderTraverse(tree.left,array)
    if tree.right:
        postOrderTraverse(tree.right,array)
    array.append(tree.value)
    return array
#Solution Iterative-----------
def inOrderTraverse(tree, array):
    curr=tree
    stack=[]
    while curr or stack:
        while curr :
            stack.append(curr)
            curr=curr.left
        curr=stack.pop()
        array.append(curr.value)
        curr=curr.right
    return array

def preOrderTraverse(tree, array):
    stack=[tree]
    while len(stack):
        node=stack.pop()
        array.append(node.value)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return array
    
def postOrderTraverse(tree, array):
    curr=tree
    stack=[]
    while curr or len(stack):
        if curr:
            stack.append([curr,False])
            curr=curr.left
        else:
            node=stack.pop()
            if not node[1]:
                node[1]=True
                stack.append(node)
                curr=node[0].right
            else:
                array.append(node[0].value)
                curr=None
    return array
    curr=tree
    stack=[]
    while curr or len(stack):
        if curr:
            stack.append([curr,False])
            curr=curr.left
        else:
            node=stack.pop()
            if not node[1]:
                node[1]=True
                stack.append(node[0])
                curr=node[0].right
            else:
                array.append(node[0].value)
                curr=None
    return array

#Question-----------------------------------------------------minHeightBst--------------------------------------------------------------------
def minHeightBst(array): #O(n) time | O(1)
    return constructMinHeightBST(array,None,0,len(array)-1)
def constructMinHeightBST(array,startIdx,endIdx):
    if endIdx<startIdx:
        return
    midIdx=(startIdx+endIdx)//2
   
    bst=BST(array[midIdx])
    bst.left=constructMinHeightBST(array,startIdx,midIdx-1)
    bst.right=constructMinHeightBST(array,midIdx+1,endIdx)
    
    return bst

#Question-----------------------------------------------------minHeightBst--------------------------------------------------------------------
#using the reverse inorder traversal
class TreeInfo:
    def __init__(self,numberOfNodeVisited,latestVisitedNodeValue) -> None:
        self.numberOfNodeVisited=numberOfNodeVisited
        self.latestVisitedNodeValue=latestVisitedNodeValue
def findKthLargestValueInBst(tree, k): #O(h+k) time | O(h) space
    treeInfo=TreeInfo(0,-1)
    reversedInOrderTraversal(tree,k,treeInfo)
    return treeInfo.latestVisitedNodeValue
def reversedInOrderTraversal(node,k,treeInfo):
    if not node or treeInfo.numberOfNodeVisited>=k:
        return 
    
    reversedInOrderTraversal(node.right,k,treeInfo)
    if treeInfo.numberOfNodeVisited<k:
        treeInfo.numberOfNodeVisited+=1
        treeInfo.latestVisitedNodeValue=node.value
        reversedInOrderTraversal(node.left,k,treeInfo)
