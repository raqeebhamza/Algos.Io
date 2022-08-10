#-----------------------------------------------------Sample Problem--------------------------------------------------------------------
# def processCase(caseNum):
#     (numCandyBags,numKids)=tuple(map(int,input().split()))
#     print((numCandyBags,numKids))
#     candyCount=list(map(int,input().split()))
#     totalCandy=0
#     for i in range(numCandyBags):
#         totalCandy+=candyCount[i]
#     amountRemaining=totalCandy%numKids

#     print(f"Case #{caseNum}: {amountRemaining}")
# numCases=int(input())
# for i in range(numCases):
#     processCase(i+1)

#--------------------------------------------------------Sherlock and Parentheses (5pts, 7pts)------------------------------------------


# def binomialCoefficient(n,k):
#     res=1
#     if k>n-k:
#         k=n-k
#     for i in range(k):
#         res*=(n-i)
#         res/=(i+1)
#     return res
# def catalan(n):
#     c=binomialCoefficient(2*n,n)
#     print(c)
#     return int(c/(n+1)) # 2nCn(n+1)
# def find_max_num_balanced_substrings(l, r):
#     res=0
#     minLR=min(l,r)
#     if not minLR:
#         return 0
#     for i in range(1,minLR+1):

#         res+=catalan(i)
#     return res
# def main():
#   # Get number of test cases
#   test_cases = int(input())
#   for test_case in range(1, test_cases + 1):
#     # Get total of left and right parentheses
#     l, r = map(int, input().split())
#     ans = find_max_num_balanced_substrings(l, r)
#     print(f"Case #{test_case}: {ans}")


# if __name__ == "__main__":
#   main()


#--------------------------------------------------------Sherlock and Parentheses (5pts, 7pts)------------------------------------------
#-----2nd Approach---------------

# def getAllSubStr(initStr):
#     repeat=0
#     subStrs=set()
#     for i in range(len(initStr)):
#         temp=""
#         if initStr[i]==')':
#             continue
#         for j in range(i,len(initStr)):
#             temp+=initStr[j]
#             if not len(temp)%2:
#                 # if temp not in subStrs:
#                 if temp in subStrs:
#                     repeat+=1
#                 else:
#                     if temp[len(temp)-1]!='(' and isValidParenthesesString(temp):
#                         subStrs.add(temp)
#     return (subStrs,repeat)
# def isValidParenthesesString(s):
#     stack=[]
#     hm={'(':')'}
#     last=-1
#     for e in s:
#         if e in hm:
#             stack.append(e)
#         else:
#             if not stack or hm[stack[last]]!=e: return False
#             stack.pop()
#     return not stack    
# def find_max_num_balanced_substrings(l, r):
#     minLR=min(l,r)
#     if not minLR:
#         return 0
#     initStr=[]
#     turn=0
#     while l>0 or r>0:
#         if not turn%2:
#             if l>0:
#                 initStr.append("(")
#                 l-=1
#             elif r>0:
#                 initStr.append(")")
#                 r-=1
#             turn+=1
#         else:
#             if r>0:
#                 initStr.append(")")
#                 r-=1
#             elif l>0:
#                 initStr.append("(")
#                 l-=1
#             turn+=1
#     initStr=''.join(map(str,initStr))
#     res=getAllSubStr(initStr)
#     return len(res[0])+res[1]
# def main():
#   # Get number of test cases
#   test_cases = int(input())
#   for test_case in range(1, test_cases + 1):
#     # Get total of left and right parentheses
#     l, r = map(int, input().split())
#     ans = find_max_num_balanced_substrings(l, r)
#     print(f"Case #{test_case}: {ans}")


# if __name__ == "__main__":
#   main()

#--------------------------------------------------------Sherlock and Parentheses (5pts, 7pts)------------------------------------------
#-----3nd Approach--------------- *****ACCEPTED****
# def find_max_num_balanced_substrings(l, r):
#     minLR=min(l,r)
    
#     return int((minLR*(minLR+1))/2)

# if __name__ == "__main__":
#   main()


#--------------------------------------------------------Sherlock and Parentheses (5pts, 7pts)------------------------------------------


def count_yes(N, Q, blocks, questions):
  # TODO: Complete this function and return the number of "yes" answers.
  yes_answers = 0

  return yes_answers

def main():
  test_cases = int(input())
  for test_case in range(1, test_cases + 1):
    N, Q = map(int, input().split())
    blocks = input()
    questions = []
    for i in range(Q):
      L, R = map(int, input().split())
      questions.append((L, R))

    answer = count_yes(N, Q, blocks, questions)

    print(f'Case #{test_case}: {answer}')

if __name__ == '__main__':
  main()
