# print(lambda s: ["#"*50 for _ in s])

def sort_funt(data):
    """This funt sorts a list of numbers that recived as a params"""
    n = len(data)

    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data

list_num = [4,6,7,2,9,5,3,1,8]

print("Original list: ", list_num)

sorted_nums = sort_funt(list_num)
print("Sorted list: ", sorted_nums)