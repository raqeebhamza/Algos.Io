# Approach - 1 (Using Two pass )
#  O(n) time | O(1) space   
# ****************************************************************************************     
def threeNumberSort(array, order):

    currEle = order[0]
    leftIdx = 0
    for i in range(len(array)):
        if array[i] == currEle:
            array[i],array[leftIdx] = array[leftIdx],array[i]
            leftIdx += 1

    currEle = order[-1]
    rightIdx = len(array)-1

    for i in range(len(array)-1,-1,-1):
        if array[i] == currEle:
            array[i],array[rightIdx] = array[rightIdx],array[i]
            rightIdx -= 1

    return array

#*****************************************************************************************
# Approach - 2 (Using One Pass)
#  O(n) time | O(1) space
def threeNumberSort(array, order):

    firstVal = order[0]
    secondVal = order[1]
    firstIdx, secondIdx, thirdIdx = 0, 0, len(array)-1
    while secondIdx <=  thirdIdx:
        val = array[secondIdx]

        if val == firstVal:
            array[secondIdx], array[firstIdx] = array[firstIdx], array[secondIdx]
            firstIdx += 1
            secondIdx += 1
        elif val == secondVal:
            secondIdx += 1
        else:
            array[secondIdx],array[thirdIdx] = array[thirdIdx], array[secondIdx]
            thirdIdx -= 1

    return array
            
        

        

    
    

        
        
