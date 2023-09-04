
# Best: O(n) time | O(1) space
# Average: O(n^2) time | O(1) space
# Worst: O(n^2) time | O(1) space

def bubbleSort(array):
    # Write your code here.
    for i in range(len(array)):
        isSorted = True
        for j in range(len(array)-1-i):
            if array[j]>array[j+1]:
                array[j],array[j+1]=array[j+1],array[j]
                isSorted = False
        if isSorted:
            break
    return array
            
            
            
