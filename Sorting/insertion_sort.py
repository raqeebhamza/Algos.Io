# Best: O(n) time | O(1) space
# Average: O(n^2) time | O(1) space
# Worst: O(n^2) time | O(1) space

def insertionSort(array):
    max = float('-inf')
    for i in range(1,len(array)):
        j=i
        while j>0 and array[j]<array[j-1]:
            array[j],array[j-1] = array[j-1],array[j]
            j -= 1
    return array
        