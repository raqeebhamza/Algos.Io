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


# def count_yes(N, Q, blocks, questions):
#   # TODO: Complete this function and return the number of "yes" answers.
#   yes_answers = 0

#   return yes_answers

# def main():
#   test_cases = int(input())
#   for test_case in range(1, test_cases + 1):
#     N, Q = map(int, input().split())
#     blocks = input()
#     questions = []
#     for i in range(Q):
#       L, R = map(int, input().split())
#       questions.append((L, R))

#     answer = count_yes(N, Q, blocks, questions)

#     print(f'Case #{test_case}: {answer}')

# if __name__ == '__main__':
#   main()


#---------------------------------------------------------- Record breaker  (5pts, 7pts)---------------------------------------------------
# def number_of_record_breaking_days(number_of_days, visitors):
#   record_breaking_days = 0
#   currMax=-1
#   for i in range(number_of_days):
#     firstCondition=visitors[i]>currMax
#     secondCondition= visitors[i]>visitors[i+1]  if i+1<number_of_days else True
#     if firstCondition and secondCondition:
#       record_breaking_days+=1
#     if currMax<visitors[i]:
#       currMax=visitors[i]
#   return record_breaking_days

# def main():
#   test_cases = int(input())
#   for test_case in range(1, test_cases + 1, 1):
#     number_of_days = int(input())
#     vistors = list(map(int, input().split()))

#     ans = number_of_record_breaking_days(number_of_days, vistors)

#     print("Case #{}: {}".format(test_case, ans))

# if __name__ == "__main__":
#   main()

#---------------------------------------------------------- Wiggle Walk  (6pts, 8pts)---------------------------------------------------

# def main():
#   test_cases = int(input())
#   for test_case in range(1, test_cases + 1):
#     N, R, C, Sr, Sc = map(int, input().split())
#     instructions = input()

#     final_r, final_c = end_position(N, R, C, Sr, Sc, instructions)
#     print(f'Case #{test_case}: {final_r} {final_c}')
# def getNeighbours(r,c,i,neighbours):
#   if (r,c,i) in neighbours:
#     return neighbours[(r,c,i)]
#   if i=='N':
#     return (r-1,c)
#   elif i=='S':
#     return (r+1,c)
#   elif i=='E':
#     return (r,c+1)
#   else: #i=='W':
#     return (r,c-1)

# def linkNeighbours(r,c,neighbours):
#   north = getNeighbours(r, c, 'N', neighbours)
#   south = getNeighbours(r, c, 'S', neighbours)
#   east = getNeighbours(r, c, 'E', neighbours)
#   west = getNeighbours(r, c, 'W', neighbours)

#   neighbours[(*north, 'S')] = south
#   neighbours[(*south, 'N')] = north
#   neighbours[(*east, 'W')] = west
#   neighbours[(*west, 'E')] = east


# def end_position(N, R, C, Sr, Sc, instructions):
#   r,c=Sr,Sc
#   neighbours={}
#   for i in instructions:
#     linkNeighbours(r,c,neighbours)
#     r,c=getNeighbours(r,c,i,neighbours)
#   return r, c

# if __name__ == '__main__':
#   main()

#---------------------------------------------------------- GBus(5pts, 12pts)---------------------------------------------------
# create count[5002] # range of cities
# for loop over (A,B): marking boundries
#     count[ai]+=1
#     count[bi+1]-=1
# for loop over count: calculate prefix sum
#    count[i]+=count[i-1]
# for c loopover desiredCity:
#     print(count[c])
# time O(N) 


# def find_number_of_gbuses_per_city(gbus_list, cities):
#   count=[0]*5002 # counter of buses for each city
#   for a,b in gbus_list: #marking the boundries
#     count[a]+=1
#     count[b+1]-=1

#   for i in range(1,len(count)): # calculating prefix sum
#     count[i]+=count[i-1]
#   ans=[]
#   for c in cities:
#     ans.append(count[c])
#   return ans

  



# def main():
#   # Get number of test cases
#   test_cases = int(input())

#   for test_case in range(1, test_cases + 1):
#     # Get gbuses
#     num_gbuses = int(input())
#     # Put gbus cities into a list of tuples of [start, end]
#     gbus_list = input().split()
#     gbus_list = [(int(gbus_list[i]), int(gbus_list[i + 1]))
#                         for i in range(len(gbus_list))
#                         if i % 2 == 0]
#     # Get cities
#     num_cities_to_return = int(input())
#     cities_list = []
#     for i in range(num_cities_to_return):
#       cities_list.append(int(input()))

#     ans = find_number_of_gbuses_per_city(gbus_list, cities_list)
#     print("Case #%d:" % (test_case), end=" ")
#     for i in range(len(ans)):
#       print("%d" % ans[i], end=" ")
#     print("") # print new line after each case.

#     # blank line between test cases
#     if test_case != test_cases:
#       _ = input()



# if __name__ == "__main__":
#   main()


#--------------------------------------------------------Sherlock and Watson Gym Secrete-------------------------------------------------------------

# Complete the count_pairs function

# def count_pairs(a,b,n,k): # brute force
#     count = 0
#     for i in range(1, n + 1):
#         for j in range(1, n + 1):
#             if i == j:
#                 continue
#             if (fast_pow(i, a, k) + pow(j, b, k)) % k == 0:
#                 count += 1
#     return count

MOD = 1000000007
def fast_pow(x, y, n):
    # pow(x, y) % n
    ans, temp = 1, x
    while y > 0:
        if y % 2 == 1:
            ans = (ans * temp) % n
        temp = (temp * temp) % n
        y //= 2
    return ans % n
# efficient one
def count_pairs(a, b, n, k):
    aDic=[0]*k
    bDic=[0]*k
    drop=[0]*k
    res=0
    i=1
    while i<=min(n,k):
        if n%k>=i:
            count=1
        else:
            count=0
        count+=n//k
        if count==0:
            continue
        aIdx=fast_pow(i,a,k)
        bIdx=fast_pow(i,b,k)
        aDic[aIdx]+=count
        bDic[bIdx]+=count
        #i==j
        if (aIdx+bIdx)%k==0:
            drop[aIdx]+=count
        i+=1
    for i in range(k):
        other=(k-i)%k
        res+=aDic[i]*bDic[other]
        res-=drop[i]
    res%=MOD
    return res
if __name__ == '__main__':
  # Read number of test cases
  num_cases = int(input())

  for tc in range(1, num_cases + 1):
    a, b, n, k = map(int, input().split())

    print("Case #%d: %d" % (tc, count_pairs(a, b, n, k)))

