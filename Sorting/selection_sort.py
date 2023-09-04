# Best: O(n) time | O(1) space
# Average: O(n^2) time | O(1) space
# Worst: O(n^2) time | O(1) space
        
def selectionSort(array):
    for j in range(len(array)-1):
        minIdx=j
        currMin=array[j]
        for i in range(j+1,len(array)):
            if currMin>array[i]:
                currMin = array[i]
                minIdx=i
        array[j],array[minIdx] = array[minIdx],array[j]

    return array
        