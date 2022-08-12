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

# Interview Question Calls
# print(AssasinVsGuards(['X.....>', '..v..X.', '....X..', 'A......']))               


##----------------------------------------------------------------------------------------------------------------------------------------

def tournamentWinner(competitions, results): # O(competitions) time | O(teams) space
    # Write your code here.
    teams={}
    for i in range(len(competitions)):
        for j in range(len(competitions[i])):
            if competitions[i][j] in teams:  
                if results[i] and j==0:   #[1,0]
                    teams[competitions[i][j]]+=3
                if not results[i] and j==1:
                    teams[competitions[i][j]]+=3
            else:
                teams[competitions[i][j]]=0
                if results[i] and j==0:   #[1,0]
                    teams[competitions[i][j]]+=3
                if not results[i] and j==1:
                    teams[competitions[i][j]]+=3
    return max(teams.items(),key=operator.itemgetter(1))[0]

##----------------------------------------------------------------------------------------------------------------------------------------

def nonConstructibleChange(coins): #O(nlogn) time | O(1) space
    coins.sort()
    currChangeCreated=0
    for coin in coins:
        if coin>currChangeCreated+1:
            return currChangeCreated+1
        currChangeCreated+=coin
    return currChangeCreated+1

  
        