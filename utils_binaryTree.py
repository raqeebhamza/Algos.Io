from logging import root


class BinaryTree:
    def __init__(self, value, left=None, right=None, parent=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent

#Question---------------------------------------------------------branchSum----------------------------------------------------------------------
def calculateSum(node,runningSum,sums): #O(n) time | O(n) space
    if not node:
        return
    newRuningSum=runningSum+node.value
    if not node.left and not node.right:
        sums.append(newRuningSum)
        return 
    calculateSum(node.left,newRuningSum,sums)
    calculateSum(node.right,newRuningSum,sums)

def branchSumss(root):
    sums=[]
    calculateSum(root,0,sums)
    return sums

#Question---------------------------------------------------------nodeDepths----------------------------------------------------------------------
#Recursion-------------
def nodeDepths(root,depth=0): #O(n) time | O(n) space
    if root is None:
        return 0
    return depth +nodeDepths(root.left,depth+1) + nodeDepths(root.right,depth+1)
#Iterative--------------
def nodeDepths(root): #O(n) time | O(n) space
    sumOfDepth=0
    stack=[{"node":root,"depth":0}]
    while len(stack)>0:
        nodeInfo=stack.pop()
        node,depth=nodeInfo["node"],nodeInfo["depth"]
        if node is None:
            continue
        sumOfDepth+=depth
        stack.append({"node":node.left,"depth":depth+1})
        stack.append({"node":node.right,"depth":depth+1})
    return sumOfDepth

#Question---------------------------------------------------------invertBinaryTree----------------------------------------------------------------------
#iterative---------------
def invertBinaryTree(tree):  #O(n) time | O(n) space
    queue=[tree]
    while len(queue):
        curr=queue.pop(0)
        if curr is None:
            continue
        swapLeftAndRight(curr)
        queue.append(curr.left)
        queue.append(curr.right)

def swapLeftAndRight(curr): 
    curr.left, curr.right=curr.right,curr.left
#Recursive---------------
def invertBinaryTree(tree): #O(n) time | O(n) space
    if tree is None:
        return
    swapLeftAndRight(tree)
    invertBinaryTree(tree.left)
    invertBinaryTree(tree.right)

#Question---------------------------------------------------------binaryTreeDiameter----------------------------------------------------------------------
class TreeInfo:
    def __init__(self,diameter,height):
        self.diameter=diameter
        self.height=height
def binaryTreeDiameter(tree): #O(n) time | O(n) space
    return getTreeInfo(tree).diameter
def getTreeInfo(tree):  
    if tree is None:
        return TreeInfo(0,0)
    leftTreeInfo=getTreeInfo(tree.left)
    rightTreeInfo=getTreeInfo(tree.right)
    
    longestPathThroghRoot=leftTreeInfo.height+rightTreeInfo.height
    maxDiameterSoFar=max(leftTreeInfo.diameter,rightTreeInfo.diameter)
    currentDiameter=max(longestPathThroghRoot,maxDiameterSoFar)
    currHeight=1+max(leftTreeInfo.height,rightTreeInfo.height)

    return TreeInfo(currentDiameter,currHeight)

#Question-----------------------------------------------------------findSuccessor--------------------------------------------------------------------

def findSuccessor(tree, node):
    if node.right:
        return getLeftMostChild(node.right)
    return getRightMostParent(node)

def getLeftMostChild(node):
    currNode=node
    while currNode.left:
        currNode=currNode.left
    return currNode
def getRightMostParent(node):
    currNode=node
    while currNode.parent and currNode.parent.right==currNode:
        currNode=currNode.parent
    return currNode.parent

#Question-----------------------------------------------------------heightBalancedBinaryTree--------------------------------------------------------------------

def heightBalancedBinaryTree(tree):
    
    def dfs(node):
        if not node:
            return (-1,True)
        
        leftOut=dfs(node.left)
        rightOut=dfs(node.right)
        
        isBalanced= leftOut[1] and rightOut[1] and abs(leftOut[0]-rightOut[0])<=1

        currHeight=max(leftOut[0],rightOut[0])+1

        return (currHeight,isBalanced)

    return dfs(tree)[1]
#Question-----------------------------------------------------------maxPathSum--------------------------------------------------------------------

def maxPathSum(tree): # O(n) time | O(logn) space 
    _,maxPath=findMaxSum(tree)
    return maxPath

def findMaxSum(tree):
    if not tree:
        return (0,0)
    
    leftSumAsBranch,leftMaxPathSum=findMaxSum(tree.left)
    rightSumAsBranch,rightMaxPathSum=findMaxSum(tree.right)
    maxChildSumAsBranch=max(leftSumAsBranch,rightSumAsBranch)
    value=tree.value
    maxSumAsBranch=max(maxChildSumAsBranch+value,value)
    maxSumAsTriangle=max(leftSumAsBranch+value+rightSumAsBranch,maxSumAsBranch)
    runningMaxPathSum=(leftMaxPathSum+value+rightMaxPathSum,maxSumAsTriangle)
    return (maxSumAsBranch,runningMaxPathSum)

#Question-----------------------------------------------------------findNodesDistanceK--------------------------------------------------------------------
#Using  BFS 
def findNodesDistanceK(tree, target, k): #O(n) time | O(n) space
    nodesToParent={}
    populateNodesToParents(tree,nodesToParent)
    targetNode=getTargetFromValue(target,tree,nodesToParent)

    return BFSForNodesDistanceK(targetNode,nodesToParent,k)

def BFSForNodesDistanceK(targetNode,nodesToParent,k):
    queue=[(targetNode,0)]
    seen={targetNode.value}
    while queue:
        currNode,distanceFromTarget=queue.pop(0)

        if distanceFromTarget==k:
            nodesDistanceK=[node.value for node,_ in queue]
            nodesDistanceK.append(currNode.value)
            return nodesDistanceK
        
        connectedNodes=[currNode.left,currNode.right,nodesToParent[currNode.value]]
        for c in connectedNodes:
            if c is None:
                continue
            if c.value in seen:
                continue
            
            seen.add(c.value)
            queue.append((c,distanceFromTarget+1))
    return []
def getTargetFromValue(target,node,nodesToParent):
    if target==node.value:
        return node
    targetParent=nodesToParent[target]
    if targetParent.left and targetParent.left.value==target:
        return targetParent.left
    return targetParent.right
def populateNodesToParents(node,nodesToParent,parent=None):
    if node:
        nodesToParent[node.value]=parent
        populateNodesToParents(node.left,nodesToParent,node)
        populateNodesToParents(node.right,nodesToParent,node)

#Question-----------------------------------------------------------iterativeInOrderTraversal--------------------------------------------------------------------

def iterativeInOrderTraversal(tree, callback): #O(n) time | O(1) space
    prevNode=None
    currNode=tree
    while currNode:
        if prevNode is None or prevNode==currNode.parent:
            if currNode.left:
                nextNode=currNode.left
            else:
                callback(currNode)
                nextNode= currNode.right if currNode.right else currNode.parent
        elif prevNode==currNode.left:
            callback(currNode)
            nextNode= currNode.right if currNode.right else currNode.parent
        else:
            nextNode=currNode.parent
        
        prevNode=currNode
        currNode=nextNode

#Question-----------------------------------------------------------flattenBinaryTree--------------------------------------------------------------------
def flattenBinaryTree(root):
    # Write your code here.
    pass