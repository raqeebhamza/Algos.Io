
#------------------------------------------------Find_missing number------------------------------------------------------
#Q.1
from heapq import merge
from msilib.schema import RemoveIniFile
from typing import List, Optional


def find_missing(arr):
    sumOfelement=sum(arr)
    n=len(arr)+1
    GuassLawSum=(n*(n+1))//2
    return GuassLawSum-sumOfelement
# print(find_missing([3,7,1,2,8,4,5]))

#------------------------------------------------Find_missing number------------------------------------------------------
#Q.2
def find_sum_of_two(arr, target):
    hm={}
    for e in arr:
       reminder=target-e
       if reminder in hm:
           return True
       hm[e]=reminder
    return False
# print(find_sum_of_two([5,7,1,2,8,4],10))

#------------------------------------------------merge two sorted list------------------------------------------------------
#Q.3

class ListNode:
    def __init__(self,val=0,next=None):
        self.val=val
        self.next=next
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:  #O(n) and O(1)   
    if list1==None:
        return list2
    if list2==None:
        return list2
    mergedHead=None
    if list1.val<list2.val:
        mergedHead=list1
        list1=list1.next
    else:
        mergedHead=list2
        list2=list2.next
    mergedTail=mergedHead
    while list1!=None and list2!=None:
        temp=None
        if list1.val<list2.val:
            temp=list1
            list1=list1.next
        else:
            temp=list2
            list2=list2.next
        mergedTail.next=temp
        mergedTail=mergedTail.next
    if list1!=None:
        mergedTail.next=list1
    elif list2!=None:
        mergedTail.next=list2
    return mergedHead

#------------------------------------------------copy linked list with random pointer------------------------------------------------------
# Q.4
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None): #O(n) and O(n)
        self.val = int(x)
        self.next = next
        self.random = random
def deep_copy_arbitrary_pointer(head):
    if head==None:
        return None
    hm=dict()
    sentinal=Node(0)
    curr,copy=head,sentinal
    while curr:
        copy.next=Node(curr.val)
        hm[curr]=copy.next
        curr,copy=curr.next,copy.next
    curr=head
    copy=sentinal.next
    while curr:
        copy.random=hm[curr.random] if curr.random else None
        curr,copy=curr.next,copy.next
    return sentinal.next

##-------------------------------------------------Binary Tree Level Order Traversal------------------------------------------------------
#Q.5
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans=[]
        level=[root]
        while root and level:
            currNodes=[]
            nextLevel=[]
            for node in level:
                currNodes.append(node.val)
                if node.left:
                    nextLevel.append(node.left)
                if node.right:
                    nextLevel.append(node.right)
            ans.append(currNodes)
            level=nextLevel
        return ans

#-------------------------------------------------Determine if a binary tree is a binary search tree------------------------------------------------------
#Q.6
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        
        def checkIsBST(node,left,right):
            if not node:
                return True
            if not left<node.val<right:
                return False
            return checkIsBST(node.left,left,node.val) and checkIsBST(node.right,node.val,right)
        
        return checkIsBST(root,float("-inf"),float("inf"))

#-------------------------------------------------String segmentation------------------------------------------------------
#Q.7
def wordBreak(self, s: str, wordDict: List[str]) -> bool: # Time O(n*m) | space O(n)
    dp=[False]*(len(s)+1)
    dp[len(s)]=True
    for i in range(len(s)-1,-1,-1):
        for w in wordDict:
            if i+len(w)<=len(s) and s[i:i+len(w)]==w:
                dp[i]=dp[i+len(w)]
            if dp[i]:
                break
    return dp[0]
#-------------------------------------------------Reverse the Words in sentence------------------------------------------------------
#Q.8
def reverseWords(self, s: str) -> str:
#     start=0
#     end=len(s)-1
#     ls=list(s)
#     ls=self.str_rev(ls,start,end)
#     ls=self.removeSides(ls)
#     ls=self.removeDuplicateSpaces(ls)
#     start=end=0
#     while end<len(ls):
#         while end<len(ls) and not ls[end].isspace():end+=1
#         ls=self.str_rev(ls,start,end-1)  
#         end+=1
#         start=end
#     return ''.join(ls)
# def str_rev(self,ls,start,end):
#     while start<end:
#         temp=ls[start]
#         ls[start]=ls[end]
#         ls[end]=temp
#         start+=1
#         end-=1
#     return ls
# def removeSides(self,ls):
#     l,r=0,len(ls)-1
#     while l<r and ls[l].isspace():l+=1
#     while l<r and ls[r].isspace():r-=1
#     return ls[l:r+1]
# def removeDuplicateSpaces(self,ls):
#     if not ls:return []
#     res=[ls[0]]
#     for i in range(1,len(ls)):
#         if res[-1].isspace() and ls[i].isspace():continue
#         res.append(ls[i])
#     return res

    split = [x for x in s.split(' ') if x != '']
    print(split)
    len_split = len(split)
    return ' '.join(split[::-1])


#-------------------------------------------------Min coins used to get the amount------------------------------------------------------
#Q.9

def coinChange(self, coins: List[int], amount: int) -> int:
    solution = [0] + [float('inf') for i in range(amount)]
    for i in range(1,amount+1): 
        for coin in coins:
            if i-coin>=0:
                solution[i]=min(solution[i],solution[i-coin]+1)
    if solution[-1]==float("inf"):
        return -1
    return solution[-1]




#-------------------------------------------------Find Kth permutation------------------------------------------------------
#Q.10

def getPermutation(self, n: int, k: int) -> str:
    ans=""
    #save factorial of number of 1-9
    fact=[1]
    f=1
    digits=[] # create digits array till n
    for i in range(1,n+1):
        f*=i
        fact.append(f)
        digits.append(i)

    def solve(n,k):
        nonlocal ans
        nonlocal fact
        nonlocal digits
        if n==1:
            ans+=str(digits[0])
            return
        idx=k//fact[n-1] # get index
        
        if k%fact[n-1]==0: # boundry check on blocks
            idx-=1
        ans+=str(digits[idx]) # add number to answer
        del digits[idx]    # remove the add number
        solve(n-1,k-fact[n-1]*idx) # update n and update k using formula
    solve(n,k)
    return ans

##-------------------------------------------------Find Subset of given Set of integer------------------------------------------------------
#Q.11