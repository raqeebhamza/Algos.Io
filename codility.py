def Solution(a):
    targetPolution=sum(a)/2
    total=sum(a)
    print(targetPolution)
    filterPolution=0
    totalFilters=0
    a.sort(reverse=True)
    print(a)
    for i in range(len(a)):
        totalFilters+=1
        filterPolution+=a[i]/2.0
        print(filterPolution)
        if filterPolution>=targetPolution:
            break
    return totalFilters

print(Solution([5,19,8,1]))
print(Solution([10,10]))
print(Solution([3,0,5]))