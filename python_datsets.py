# Write a python program to find and print if a number given is disarium or not

num = 135
num_len = len(str(num))
n = num
sum = 0
exp = num_len
while n != 0:
    i = int(n % 10)
    n = int(n / 10)
    sum += i ** exp
    exp -= 1
if sum == num:
    print("disarium")
else:
    print("not disarium")
	
	
# Write a python program to find and print second largest number from list of numbers

num_array = [8, 6, 15, 23, 14, 28, 5, 1, 99]
largest = second_largest = num_array[0]
for i in range(1,len(num_array)):
    if num_array[i] > largest:
        second_largest = largest
        largest = num_array[i]
    elif num_array[i] > second_largest:
        second_largest = num_array[i]
print(second_largest)


# Write a python program to find and print volume of a sphere for which diameter d is given
import math

diameter = 12.
radius = diameter/2.
# Calculate volume V
V = 4./3. * math.pi * radius ** 3
print(f"Volume={V}")


# Write a python program using list comprehension to produce and print the list ['x', 'xx', 'xxx', 'xxxx', 'y', 'yy', 'yyy', 'yyyy', 'z', 'zz', 'zzz', 'zzzz']

input_string_list = ['x', 'y', 'z']
repeat_count = 4
list2 = [input_string_list[i] * (j+1)  for i in range(len(input_string_list)) for j in range(repeat_count) ]
print(list2)


# Write a python program using list comprehension to produce and print the list ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xxx', 'yyy', 'zzz', 'xxxx', 'yyyy', 'zzzz']

input_string_list = ['x', 'y', 'z']
repeat_count = 4
list3 = [input_string_list[i] * (j+1) for j in range(repeat_count)  for i in range(len(input_string_list)) ]
print(list3)


# Write a python program using list comprehension to produce and print the list  [[2],[3],[4],[3],[4],[5],[4],[5],[6]]

start_num = 2
repeat_count = 3
max_offset = 3
list4 = [[start_num + i + j ]  for j in range(max_offset) for i in range(repeat_count) ] 
print(list4)


# Write a python program using list comprehension to produce and print the list [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]

start_num = 2
repeat_count = 4
max_offset =4
list5 = [[start_num + i + j  for j in range(max_offset)]  for i in range(repeat_count) ]
print(list5)


# Write a python program using list comprehension to produce and print the list [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]

max_count = 3
list6 = [(j+1,i+1)  for i in range(max_count)  for j in range(max_count) ]
print(list6)


# Implement a python function longestWord which take input as list of words and return the longest word

import functools

def longestWord(word_list):
    if word_list is None or isinstance(word_list, list) == False or len(word_list) == 0:
        raise ValueError("Input word_list to function longestWord must be list of words of size at least 1")
    
    if len(word_list) == 1:
        return word_list[0]    
    else:
        return functools.reduce(lambda x,y: x if len(x) >= len(y) else y, word_list)
		
		
# Write a python program that maps list of words into a list of integers representing the lengths of the corresponding words

lst = ["ab","cde","erty"]
length_list = list(map((lambda element: len(element)), lst))
print(str(length_list))


# Write a python program to generate and print all sentences where subject is in["Americans", "Indians"] and verb is in ["Play", "watch"] and the object is in ["Baseball","cricket"]

subjects=["Americans","Indians"]
verbs=["play","watch"]
objects=["Baseball","Cricket"]
sentence_list = [subject + " " + verb + " " + object + "." for subject in subjects for verb in verbs for object in objects]
for sentence in sentence_list:
    print(sentence)
	
	
# Write a python program which accepts users first name and last name and print in reverse order with a space

first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")
print(last_name.strip() + " " + first_name.strip())


# Write a python function to find minimum edit distance between words given

def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)

    if m*n == 0:
        return m + n

    d = [ [0] * (n + 1) for _ in range(m+1)]
    for i in range(m+1):
        d[i][0] = i

    for j in range(n+1):
            d[0][j] = j

    for i in range(m+1):
        for j in range(n+1):
            left = d[i-1][j] + 1
            down = d[i][j-1] + 1
            left_down = d[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                left_down += 1
            d[i][j] = min(left, down, left_down)

    return d[m][n]
	
	
# Write a python function to return list of all the possible gray code for a number given

def grayCode(n):
    if n == 0:
        return [0]
        
    if n == 1:
        return [0,1]
        
    res = []
                
    start = '0'*n
    visited = set()
    stk = [start]
        
    while stk:
        node = stk.pop()
        if node not in visited:
            res.append(int(node,2))            
            visited.add(node)
        if len(visited) == 2**n:
            break
                
        for i in range(n):
            newCh = '0' if node[i] == '1' else '1' 
            newNode = node[:i] + newCh + node[i+1:]
                
            if newNode not in visited:
                    stk.append(newNode)
    return res
	

# Write a python function which takes a list of non negative numbers and target sum S, two operations (+, -) how many different ways target sum is achived re

def findTargetSumWays(nums, S):
    count = 0
    def calculate(nums, i, sum, S):
        nonlocal count
        if i == len(nums):
            if sum == S:
                count += 1
        else:
            calculate(nums, i+1, sum+ nums[i], S)
            calculate(nums, i+1, sum- nums[i], S)
            
    calculate(nums, 0, 0, S) 
    return count
	
	
	
	
# Write a python function which wil return True if list parenthesis used in a input expression is valid, False otherwise

def isValid(s):
    stack = []
    mapping = {')': '(', '}' : '{', ']':'['}
    for char in s:
        if char in mapping:                
            if not stack:
                return False
            top = stack.pop()
            if mapping[char] != top:
                return False
        else:
            stack.append(char)     

    return not stack
	
	
	
# Write a python function to solve and print Towers of Hanoi problem

def TowerOfHanoi(n , source, destination, auxiliary): 
    if n==1: 
        print("Move disk 1 from source",source,"to destination",destination) 
        return
    TowerOfHanoi(n-1, source, auxiliary, destination) 
    print("Move disk",n,"from source",source,"to destination",destination) 
    TowerOfHanoi(n-1, auxiliary, destination, source)

	
	
# Write a python function to check if a number given is a Armstrong number

def isArmstrong(x):
    n = 0
    while (x != 0): 
        n = n + 1
        x = x // 10
    temp = x 
    sum1 = 0
      
    while (temp != 0): 
        r = temp % 10
        sum1 = sum1 + r ** n
        temp = temp // 10
  
    return (sum1 == x) 
	
	
# Write a python program to find and print sum of series with cubes of first n natural numbers 

n = 10
sum = 0
for i in range(1, n+1): 
    sum += i**3

print(f"{sum}")


# Write a python  function which returns True elements in a given list is monotonically increasing or decreasing, return False otherwise 

def isMonotonic(A):  
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1))) 
			
			

# Write a python program to find and print product of two matrices

A = [[12, 7, 3], 
    [4, 5, 6], 
    [7, 8, 9]]   
  
B = [[5, 8, 1, 2], 
    [6, 7, 3, 0], 
    [4, 5, 9, 1]] 
      
result = [[0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0]] 
  

for i in range(len(A)): 
    for j in range(len(B[0])): 
        for k in range(len(B)): 
            result[i][j] += A[i][k] * B[k][j] 
  
for r in result: 
    print(r) 
	
	
# Write a python program to find and print  K th column of a matrix

test_list = [[4, 5, 6], [8, 1, 10], [7, 12, 5]] 

K = 2
res = [sub[K] for sub in test_list] 
print("The Kth column of matrix is : " + str(res)) 


# Write a python program to Convert and print Snake case to Pascal case 
test_str = 'go_east_or_west_india_is_the_best'
res = test_str.replace("_", " ").title().replace(" ", "")
print(res)


# Write a python  program to print only even length words in a sentence

def printEvenLengthWords(s):       
    s = s.split(' ')  
    for word in s:
        if len(word)%2==0: 
            print(word)  

			
# Write a python function to find uncommon words between two sentences given

def UncommonWords(A, B): 
    count = {} 
    for word in A.split(): 
        count[word] = count.get(word, 0) + 1
    for word in B.split(): 
        count[word] = count.get(word, 0) + 1
  
    return [word for word in count if count[word] == 1]
	
	
# Write a python function which determines if binary representation of a number is palindrome

def binaryPallindrome(num): 
     binary = bin(num) 
     binary = binary[2:] 
     return binary == binary[-1::-1] 
	 
	 
# Write a python program to extract and print words that starts with vowel

test_list = ["all", "love", "and", "get", "educated", "by", "gfg"] 
  
res = [] 
vow = "aeiou"
for sub in test_list: 
    flag = False
    for ele in vow: 
        if sub.startswith(ele): 
            flag = True 
            break
    if flag: 
        res.append(sub) 
print("The extracted words : " + str(res)) 


# Write a python function to extract URLs from a sentence

import re 
  
def FindUrls(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
	
	
# Write a python function to Check and print if binary representations of two numbers are anagram 

from collections import Counter 
  
def checkAnagram(num1,num2): 
    bin1 = bin(num1)[2:] 
    bin2 = bin(num2)[2:] 
  
    zeros = abs(len(bin1)-len(bin2)) 
    if (len(bin1)>len(bin2)): 
         bin2 = zeros * '0' + bin2 
    else: 
         bin1 = zeros * '0' + bin1 
  
    dict1 = Counter(bin1) 
    dict2 = Counter(bin2) 
  

    if dict1 == dict2: 
         print('Yes') 
    else: 
         print('No') 
		 
  
# Write a program to print inverted star pattern for the given number

n=11

for i in range (n, 0, -1): 
    print((n-i) * ' ' + i * '*') 
	

	
# Write a python function to find and print if IP address given is a valid IP address or not

import re
 
def Validate_IP(IP):
    regex = "(([0-9]|[1-9][0-9]|1[0-9][0-9]|"\
            "2[0-4][0-9]|25[0-5])\\.){3}"\
            "([0-9]|[1-9][0-9]|1[0-9][0-9]|"\
            "2[0-4][0-9]|25[0-5])"
     
    regex1 = "((([0-9a-fA-F]){1,4})\\:){7}"\
             "([0-9a-fA-F]){1,4}"
     
    p = re.compile(regex)
    p1 = re.compile(regex1)
 
    if (re.search(p, IP)):
        return "Valid IPv4"
 
    elif (re.search(p1, IP)):
        return "Valid IPv6"
 
    return "Invalid IP"
	

# Write a python function to find and print if a email address given is valid or not

import re 
  
regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
def check(email):   
    if(re.search(regex,email)):  
        print("Valid Email")  
          
    else:  
        print("Invalid Email")	
		
		
# Write a python program to check and print if the password is valid or not not with given rules 1. Minimum 8 characters.2. The alphabets must be between [a-z] 3.  At least one alphabet should be of Upper Case [A-Z] 4.  At least 1 number or digit between [0-9]. 5. At least 1 character from [ _ or @ or $ ].

import re 
password = "R@m@_f0rtu9e$"
flag = 0
while True:   
    if (len(password)<8): 
        flag = -1
        break
    elif not re.search("[a-z]", password): 
        flag = -1
        break
    elif not re.search("[A-Z]", password): 
        flag = -1
        break
    elif not re.search("[0-9]", password): 
        flag = -1
        break
    elif not re.search("[_@$]", password): 
        flag = -1
        break
    elif re.search("\s", password): 
        flag = -1
        break
    else: 
        flag = 0
        print("Valid Password") 
        break
  
if flag ==-1: 
    print("Not a Valid Password") 
	
	
	
# Write a python function to find and print the largest prime factor of a given number

import math 
  
def maxPrimeFactors (n): 
      
    maxPrime = -1
      
    while n % 2 == 0: 
        maxPrime = 2
        n >>= 1               

    for i in range(3, int(math.sqrt(n)) + 1, 2): 
        while n % i == 0: 
            maxPrime = i 
            n = n / i 
      
    if n > 2: 
        maxPrime = n 
      
    return int(maxPrime)
	
  
# Write a python function to determine if a year is leap year or not

def is_leap(year):
    leap = False
    
    # Write your logic here
    if year % 4 == 0:
        if year % 400 == 0:
            leap = True
        elif year % 100 == 0:
            leap = False
        else:
            leap = True
    return leap

	
# Write a python function to generate permuations of a list of given numbers

def permute(nums):
    def backtrack(first = 0):
        if first == n:        
            output.append(nums[:])
        for i in range(first, n):
            nums[first], nums[i] = nums[i], nums[first]

            backtrack(first + 1)

            nums[first], nums[i] = nums[i], nums[first]
         
    n = len(nums)
    output = []
    backtrack()
    return output

	
# Write a python function to print staircase pattern

def pattern(n): 
  
    for i in range(1,n+1):   
        # conditional operator 
        k =i + 1 if(i % 2 != 0) else i   

        for g in range(k,n): 
            if g>=k: 
                print(end="  ") 
  
        for j in range(0,k): 
            if j == k - 1: 
                print(" * ") 
            else: 
                print(" * ", end = " ") 
  
  
# Write a python function to find gcd using eucliean algorithm

def gcd(a, b):  
    if a == 0 : 
        return b  
      
    return gcd(b%a, a)
	

# Write a python function to check if number is divisible by all the digits

def allDigitsDivide( n) :
     
    temp = n
    while (temp > 0) :
        digit = temp % 10
        if not (digit != 0 and n % digit == 0) :
            return False
 
        temp = temp // 10
     
    return True
	
	
# Write a python program to flatten  a multidimensional list

my_list = [[10,20,30],[40,50,60],[70,80,90]]

flattened = [x for temp in my_list for x in temp]
print(flattened)


# Write Python Program to Print Table of a Given Number

n=int(input("Enter the number to print the tables for:"))
for i in range(1,11):
    print(n,"x",i,"=",n*i)
	
	
# Write a python program to check and print if the number is a perfect number

n = int(input("Enter any number: "))
sum1 = 0
for i in range(1, n):
    if(n % i == 0):
        sum1 = sum1 + i
if (sum1 == n):
    print("The number is a Perfect number!")
else:
    print("The number is not a Perfect number!")
	
	
# Write a python function to find and print longest continous odd sequence of a list of numbers given

def longest_continuous_odd_subsequence(array):
    final_list = []
    temp_list = []
    for i in array:
        if i%2 == 0:
            if temp_list != []:
                final_list.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)

    if temp_list != []:
        final_list.append(temp_list)

    result = max(final_list, key=len)
    print(result)
	


# Write a function to determine longest increasing subsequence of a list of numbers given

def longest_increaing_subsequence(myList):

    lis = [1] * len(myList)

    elements = [0] * len(myList)

    for i in range (1 , len(myList)):
        for j in range(0 , i):
            if myList[i] > myList[j] and lis[i]< lis[j] + 1:
                lis[i] = lis[j]+1
                elements[i] = j
    idx = 0


    maximum = max(lis)              
    idx = lis.index(maximum)


    seq = [myList[idx]]
    while idx != elements[idx]:
        idx = elements[idx]
        seq.append(myList[idx])

    return (maximum, reversed(seq))
	

# Write function for performing heapsort on a list of numbers given

def heapify(nums, heap_size, root_index):
    largest = root_index
    left_child = (2 * root_index) + 1
    right_child = (2 * root_index) + 2

    if left_child < heap_size and nums[left_child] > nums[largest]:
        largest = left_child

    if right_child < heap_size and nums[right_child] > nums[largest]:
        largest = right_child

    if largest != root_index:
        nums[root_index], nums[largest] = nums[largest], nums[root_index]
        heapify(nums, heap_size, largest)

def heap_sort(nums):
    n = len(nums)
    
    for i in range(n, -1, -1):
        heapify(nums, n, i)

    # Move the root of the max heap to the end of
    for i in range(n - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        heapify(nums, i, 0)
		
		
# Write a python function to perform quicksort sort on a list of numbers given

def partition(array, low, high):
    i = low - 1            # index of smaller element
    pivot = array[high]    # pivot 
    
    for j in range(low, high):
       
        if array[j] < pivot:
    
            i += 1
            array[i], array[j] = array[j], array[i]
            
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1

def quick_sort(array, low, high):
    if low < high:
        temp = partition(array, low, high)
        quick_sort(array, low, temp - 1)
        quick_sort(array, temp + 1, high)
		

# Given a decimal number N, write python functions check and print if a number has consecutive zeroes or not after converting the number to its K-based notation.

def hasConsecutiveZeroes(N, K): 
    z = toK(N, K) 
    if (check(z)): 
        print("Yes") 
    else: 
        print("No") 
  
def toK(N, K): 
  
    w = 1
    s = 0
    while (N != 0): 
        r = N % K 
        N = N//K 
        s = r * w + s 
        w *= 10
    return s 
  
def check(N): 

    fl = False
    while (N != 0): 
        r = N % 10
        N = N//10
  
        if (fl == True and r == 0): 
            return False
        if (r > 0): 
            fl = False
            continue
        fl = True
    return True
	
	
# Write a python class to implement circular queue with methods enqueue, dequeue

class CircularQueue(object):
    def __init__(self, limit = 10):
        self.limit = limit
        self.queue = [None for i in range(limit)]  
        self.front = self.rear = -1

    def __str__(self):
        if (self.rear >= self.front):
            return ' '.join([str(self.queue[i]) for i in range(self.front, self.rear + 1)])
  
        else: 
            q1 = ' '.join([str(self.queue[i]) for i in range(self.front, self.limit)])
            q2 = ' '.join([str(self.queue[i]) for i in range(0, self.rear + 1)])
            return q1 + ' ' + q2

    def isEmpty(self):
        return self.front == -1

    def isFull(self):
        return (self.rear + 1) % self.limit == self.front

    def enqueue(self, data):
        if self.isFull():
            print('Queue is Full!')
        elif self.isEmpty():
            self.front = 0
            self.rear = 0
            self.queue[self.rear] = data
        else:
            self.rear = (self.rear + 1) % self.limit  
            self.queue[self.rear] = data 

    def dequeue(self):
        if self.isEmpty():
            print('Queue is Empty!')
        elif (self.front == self.rear):  
            self.front = -1
            self.rear = -1
        else:
            self.front = (self.front + 1) % self.limit 
			
			
# Write a python class to implement Deque where elements can be added and deleted both ends

class Deque(object):
    def __init__(self, limit = 10):
        self.queue = []
        self.limit = limit

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def isEmpty(self):
        return len(self.queue) <= 0

    def isFull(self):
        return len(self.queue) >= self.limit

    def insertRear(self, data):
        if self.isFull():
            return
        else:
            self.queue.insert(0, data)

    def insertFront(self, data):
        if self.isFull():
            return
        else:
            self.queue.append(data)

    def deleteRear(self):
        if self.isEmpty():
            return
        else:
            return self.queue.pop(0)

    def deleteFront(self):
        if self.isFull():
            return
        else:
            return self.queue.pop()
			


# Write a python class to implement PriorityQueue

class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])


    def isEmpty(self):
        return len(self.queue) == []


    def insert(self, data):
        self.queue.append(data)


    def delete(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i] > self.queue[max]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()
			

		
			
  
# Write a python function to return minimum sum of factors of a number

def findMinSum(num): 
    sum = 0
      
    i = 2
    while(i * i <= num): 
        while(num % i == 0): 
            sum += i 
            num /= i 
        i += 1
    sum += num    

    return sum
	
	
	
# Write a function to check and print if a string starts with a substring using regex in Python

import re 
  
  
def find(string, sample) :    

  if (sample in string): 
  
      y = "^" + sample 
  
      x = re.search(y, string) 
  
      if x : 
          print("string starts with the given substring") 
  
      else : 
          print("string doesn't start with the given substring") 
  
  else : 
      print("entered string isn't a substring") 
	  
	  
# Write a python program to print square matrix in Z form

arr = [[4, 5, 6, 8],  
        [1, 2, 3, 1],  
        [7, 8, 9, 4],  
        [1, 8, 7, 5]] 
  
n = len(arr[0]) 
                   
i=0
for j in range(0, n-1): 
    print(arr[i][j], end =" ")  
          
k = 1
for i in range(0, n): 
    for j in range(n, 0, -1): 
        if(j==n-k): 
            print(arr[i][j], end = " ")  
            break;  
    k+=1
  

i=n-1;  
for j in range(0, n): 
    print(arr[i][j], end = " ") 
  
  
# Write a python function to calculate number of ways of selecting p non  consecutive stations out of n stations 
  
def stopping_station( p, n): 
    num = 1
    dem = 1
    s = p 
  
    while p != 1: 
        dem *= p 
        p-=1
      
    t = n - s + 1
    while t != (n-2 * s + 1): 
        num *= t 
        t-=1
    if (n - s + 1) >= s: 
        return int(num/dem) 
    else: 

        return -1
		
		
# Write a python program to solve and print the solution for the quadratic equation ax**2 + bx + c = 0

import cmath

a = 1
b = 5
c = 6

d = (b**2) - (4*a*c)


sol1 = (-b-cmath.sqrt(d))/(2*a)
sol2 = (-b+cmath.sqrt(d))/(2*a)

print('The solution are {0} and {1}'.format(sol1,sol2))


# Write a program to print the powers of 2 using anonymous function

terms = 10

result = list(map(lambda x: 2 ** x, range(terms)))

print("The total terms are:",terms)
for i in range(terms):
   print("2 raised to power",i,"is",result[i])
   

   
# Write a python function to find the L.C.M. of two input number

def compute_lcm(x, y):

   # choose the greater number
   if x > y:
       greater = x
   else:
       greater = y

   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1

   return lcm
   
   
# Write a Python program to shuffle and print a deck of card

import itertools, random

deck = list(itertools.product(range(1,14),['Spade','Heart','Diamond','Club']))

random.shuffle(deck)

print("You got:")
for i in range(5):
   print(deck[i][0], "of", deck[i][1])

   
   
# Write a python program to sort alphabetically the words form a string provided by the user

my_str = "Hello this Is an Example With cased letters"

words = [word.lower() for word in my_str.split()]

words.sort()


print("The sorted words are:")
for word in words:
   print(word)
   
   
# Write a python program to remove punctuations from a sentence

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

my_str = "Hello!!!, he said ---and went."

no_punct = ""
for char in my_str:
   if char not in punctuations:
       no_punct = no_punct + char

print(no_punct)


# Write a python  program to check and print Yes/No if a triangle  of positive area is possible with the given angles 
  
def isTriangleExists(a, b, c): 

    if(a != 0 and b != 0 and c != 0 and (a + b + c)== 180): 
 
        if((a + b)>= c or (b + c)>= a or (a + c)>= b): 
            return "YES"
        else: 
            return "NO"
    else: 
        return "NO"
  
  
# Write a program to rotate and print elements of a list

arr = [1, 2, 3, 4, 5];     
  
n = 3;      
  
for i in range(0, n):    
    #Stores the last element of array    
    last = arr[len(arr)-1];    
        
    for j in range(len(arr)-1, -1, -1):    
        #Shift element of array by one    
        arr[j] = arr[j-1];    
            
       
    arr[0] = last; 
    
print(arr)
   

# Write a program to find and print if a number is a Harshad number

num = 156;    
rem = sum = 0;   
   
n = num;    
while(num > 0):    
    rem = num%10;    
    sum = sum + rem;    
    num = num//10;    
     

if(n%sum == 0):    
    print(str(n) + " is a harshad number")    
else:    
    print(str(n) + " is not a harshad number")
	

# Write a program to left rotate and print a list given

arr = [1, 2, 3, 4, 5];     
  
n = 3;    
     
 
for i in range(0, n):    
    first = arr[0];    
        
    for j in range(0, len(arr)-1):    
  
        arr[j] = arr[j+1];    
            
 
    arr[len(arr)-1] = first;    
     
  
print("Array after left rotation: ");    
for i in range(0, len(arr)):    
    print(arr[i]),  


# Write a python function to implement 0/1 Knapsack problem

def knapSack(W, wt, val, n): 
  
    # Base Case 
    if n == 0 or W == 0 : 
        return 0
  
    if (wt[n-1] > W): 
        return knapSack(W, wt, val, n-1) 
  
    else: 
        return max(val[n-1] + knapSack(W-wt[n-1], wt, val, n-1), 
                   knapSack(W, wt, val, n-1))  


# Write a function to find out if permutations of a given string is a palindrome

def has_palindrome_permutation(the_string):
    unpaired_characters = set()

    for char in the_string:
        if char in unpaired_characters:
            unpaired_characters.remove(char)
        else:
            unpaired_characters.add(char)


    return len(unpaired_characters) <= 1				   
		
		
# Write a python function to determine optimal buy and sell time of stocks given stocks for yesterday

def get_max_profit(stock_prices):
    max_profit = 0

    for outer_time in range(len(stock_prices)):

        for inner_time in range(len(stock_prices)):
            earlier_time = min(outer_time, inner_time)
            later_time   = max(outer_time, inner_time)

            earlier_price = stock_prices[earlier_time]
            later_price   = stock_prices[later_time]

            potential_profit = later_price - earlier_price

            max_profit = max(max_profit, potential_profit)

    return max_profit
	

# Write a python function to check if cafe orders are served in the same order they are paid for

def is_first_come_first_served(take_out_orders, dine_in_orders, served_orders):
    # Base case
    if len(served_orders) == 0:
        return True


    if len(take_out_orders) and take_out_orders[0] == served_orders[0]:
        return is_first_come_first_served(take_out_orders[1:], dine_in_orders, served_orders[1:])


    elif len(dine_in_orders) and dine_in_orders[0] == served_orders[0]:
        return is_first_come_first_served(take_out_orders, dine_in_orders[1:], served_orders[1:])


    else:
        return False


# Write a function to merge meeting times given everyone's schedules

def merge_ranges(meetings):
    sorted_meetings = sorted(meetings)

    merged_meetings = [sorted_meetings[0]]

    for current_meeting_start, current_meeting_end in sorted_meetings[1:]:
        last_merged_meeting_start, last_merged_meeting_end = merged_meetings[-1]

        if (current_meeting_start <= last_merged_meeting_end):
            merged_meetings[-1] = (last_merged_meeting_start,
                                   max(last_merged_meeting_end,
                                       current_meeting_end))
        else:
            merged_meetings.append((current_meeting_start, current_meeting_end))

    return merged_meetings


# Write a python function which accepts or discard only string ending with alphanumeric character

import re 
  

regex = '[a-zA-z0-9]$'
      
def check(string):
    if(re.search(regex, string)):  
        print("Accept")          
    else:  
        print("Discard") 


# Write a python program to accept a number n and calculate n+nn+nn

n=int(input("Enter a number n: "))
temp=str(n)
t1=temp+temp
t2=temp+temp+temp
comp=n+int(t1)+int(t2)
print("The value is:",comp)


# Write a program to accept a number and print inverted star pattern

n=int(input("Enter number of rows: "))
for i in range (n,0,-1):
    print((n-i) * ' ' + i * '*')
	
	
# Write a program to print prime numbers in a range using Sieve of Eratosthenes.

n=int(input("Enter upper limit of range: "))
sieve=set(range(2,n+1))
while sieve:
    prime=min(sieve)
    print(prime,end="\t")
    sieve-=set(range(prime,n+1,prime))
 
print()


# Write python function to generate valid parenthesis, number of parenthesis is given as input

def generateParenthesis(n):
        
    def backtrack(S='', left=0, right=0):
        if len(S) == 2*n:
            output.append(S)
            return
        if left < n:
            backtrack(S+'(', left+1, right)
        if right < left:
            backtrack(S+')', left, right+1)
        
    output = []
    backtrack()
    return output
	
	
# Write python function which Given an list distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. 

def combinationSum(candidates, target):
    results = []
    def helper(i, path):
        if sum(path) == target:
            results.append(path[:])
            return
            
        if sum(path) > target:
            return
            
        for x in range(i, len(candidates)):
            path.append(candidates[x])
            helper(x, path)
            path.pop()
                
    helper(0, []) 
    return results
	
	
# Write a function Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead. 

def dailyTemperatures(T):
    stack = []
    res = [0 for _ in range(len(T))]
    for i, t1 in enumerate(T):
        while stack and t1 > stack[-1][1]:
            j, t2 = stack.pop()
            res[j] = i - j
        stack.append((i, t1))
    return res
	
	
# Write a function which Given an array of integers nums and a positive integer k, find whether it's possible to divide this array into sets of k consecutive numbers Return True if its possible otherwise return False

import collections
def isPossibleDivide(nums, k):
 
    d = collections.Counter(nums)
    for num in sorted(d.keys()):
        if num in d:
            for i in range(k - 1, -1, -1):
                d[num + i] -= d[num] 
                if d[num + i] == 0:
                    del d[num + i]
                if d[num + i] < 0:
                    return False

    return (True if not d else False)
	
	
# Write a function pow(x, n), which calculates x raised to the power n 

def myPow(x, n):
    def pow(y, n):
        if n == 0:
            return 1.0
        else:
            partial = pow(x, n//2)
            result = partial * partial
            if n%2 == 1:
                result *= x
            return result
           
  
    if n >= 0:
        return pow(x, n)
    else:
        return 1/ pow(x, -n)

		
# Write a python  class to implement LRU Cache

class DLinkedNode:
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None

class LRUCache(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.cache = {}
        self.size = 0
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_node(self, node):
        node.next = self.head.next
        node.prev = self.head        
        self.head.next.prev = node
        self.head.next = node

    def remove_node(self, node):
        next = node.next
        prev = node.prev
        prev.next = next
        next.prev = prev

    def move_to_head(self, node ):
        self.remove_node(node)
        self.add_node(node)

    def tail_off(self ):
        res = self.tail.prev
        self.remove_node(res)
        return res       


    def get(self, key):
        node = self.cache.get(key, None)
        if not node:
            return -1

        self.move_to_head(node )
        return node.value
        

    def put(self, key, value):
        node = self.cache.get(key, None)
        if  not node:           
            node = DLinkedNode()
            node.key = key
            node.value = value
            self.cache[key] = node
            self.add_node(node )
            self.size += 1
            if self.size > self.capacity:
                last_node = self.tail_off()
                del self.cache[last_node.key]
                self.size -= 1
        else:
            node.value = value
            self.move_to_head(node )
			

# Write functions which given Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

def cross_sum(nums, left, right, p):
    if left == right:
        return nums[left]

    left_subsum=float('-Inf')

    current_sum = 0
    for i in range(p, left-1, -1):
        current_sum += nums[i]
        left_subsum = max(left_subsum, current_sum)

        right_subsum=float('-Inf')

        current_sum = 0
        
    for i in range(p+1, right+1):
        current_sum += nums[i]
        right_subsum = max(right_subsum, current_sum)

    return left_subsum + right_subsum

def helper(nums, left, right):
    if left == right: 
        return nums[left]

    p = (left + right) // 2

    left_sum = helper(nums, left, p)
    right_sum = helper(nums, p+1, right)
    cross_sum1 =  cross_sum(nums, left, right, p)

    return max(left_sum, right_sum, cross_sum1)

def maxSubArray(nums):
    return helper(nums, 0, len(nums) -1)
	

# Write a function which Given an list of integers arr and an integer target, find two non-overlapping sub-arrays of arr each with sum equal target

from collections import defaultdict
def minSumOfLengths(arr, target):
    hashTable = defaultdict(int)
    hashTable[0] = -1
    summation = 0
    for i in range(len(arr)):
        summation = summation + arr[i]
        hashTable[summation] = i
        
    summation = 0
    minimumLeft = float('inf')
    result = float('inf')
    for i in range(len(arr)):
        summation = summation + arr[i]
        if summation - target in hashTable:
            leftLength = i-hashTable[summation-target]
            minimumLeft = min(minimumLeft,leftLength)
        if summation + target in hashTable and minimumLeft < float('inf'):
            rightLength = hashTable[summation+target]-i
            result = min(result,hashTable[summation+target]-i+minimumLeft)
        
    if result == float('inf'):
        return -1
    return result
	
	
# Write a function which Given a keyboard layout in XY plane, where each English uppercase letter is located at some coordinate, say (0,0) for A, return the minimum total distance to type such string using only two fingers. The distance distance between coordinates (x1,y1) and (x2,y2) is |x1 - x2| + |y1 - y2|. 

from functools import lru_cache

def minimumDistance(word):
    def getDist(a, b):
        if a==-1 or b==-1:
            return 0
        else:
            i = ord(a) - ord('a')
            j = ord(b) - ord('b')
            dist = abs(i//6 - j//6) + abs(i%6 - j%6)
            return dist
            
    @lru_cache(maxsize=None)
    def getMinDist(l, r, k):
        if k==len(word):
            return 0
        next = word[k].lower()
        ret = min(getMinDist(next,r,k+1)+getDist(l,next), getMinDist(l,next,k+1)+getDist(r,next))
        return ret
            
    return(getMinDist(-1,-1,0))

	
# Write a function to generate permutation of list of numbers

def permute(nums):
    def backtrack(first = 0):
        if first == n:      
            output.append(nums[:])
        for i in range(first, n):
            nums[first], nums[i] = nums[i], nums[first]            
            backtrack(first + 1)

            nums[first], nums[i] = nums[i], nums[first]
 
        
    n = len(nums)
    output = []
    backtrack()
    return output
	
        
# Write a python class to implement a Bank which which supports basic operations like depoist, withdrwa, overdrawn

class BankAccount(object):
    def __init__(self, account_no, name, initial_balance=0):
        self.account_no = account_no
        self.name = name
        self.balance = initial_balance
    def deposit(self, amount):
        self.balance += amount
    def withdraw(self, amount):
        self.balance -= amount
    def overdrawn(self):
        return self.balance < 0
		
		
# Write a function to calculate median of a list of numbers given

def median(pool):
    copy = sorted(pool)
    size = len(copy)
    if size % 2 == 1:
        return copy[int((size - 1) / 2)]
    else:
        return (copy[int(size/2 - 1)] + copy[int(size/2)]) / 2
		

# Write a program to guess a number between 1 and 20 and greet if succesfully guessed and print the results

import random

guesses_made = 0

name = input('Hello! What is your name?\n')

number = random.randint(1, 20)
print ('Well, {0}, I am thinking of a number between 1 and 20.'.format(name))

while guesses_made < 6:

    guess = int(input('Take a guess: '))

    guesses_made += 1

    if guess < number:
        print ('Your guess is too low.')

    if guess > number:
        print ('Your guess is too high.')

    if guess == number:
        break

if guess == number:
    print ('Good job, {0}! You guessed my number in {1} guesses!'.format(name, guesses_made))
else:
    print ('Nope. The number I was thinking of was {0}'.format(number))
	
	
# Write a python program to implement Rock, paper, scissor game and print the results

import random
import os
import re
os.system('cls' if os.name=='nt' else 'clear')
while (1 < 2):
    print("\n")
    print("Rock, Paper, Scissors - Shoot!")
    userChoice = input("Choose your weapon [R]ock], [P]aper, or [S]cissors: ")
    if not re.match("[SsRrPp]", userChoice):
        print("Please choose a letter:")
        print("[R]ock, [S]cissors or [P]aper.")
        continue
    print("You chose: " + userChoice)
    choices = ['R', 'P', 'S']
    opponenetChoice = random.choice(choices)
    print("I chose: " + opponenetChoice)
    if opponenetChoice == str.upper(userChoice):
        print("Tie! ")
    #if opponenetChoice == str("R") and str.upper(userChoice) == "P"
    elif opponenetChoice == 'R' and userChoice.upper() == 'S':      
        print("Scissors beats rock, I win! ")
        continue
    elif opponenetChoice == 'S' and userChoice.upper() == 'P':      
        print("Scissors beats paper! I win! ")
        continue
    elif opponenetChoice == 'P' and userChoice.upper() == 'R':      
        print("Paper beat rock, I win! ")
        continue
    else:       
        print("You win!")
		
		
# Write a python program to implement Tic Tac Toe game and print the results


import random
import sys
board=[i for i in range(0,9)]
player, computer = '',''

moves=((1,7,3,9),(5,),(2,4,6,8))

winners=((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

tab=range(1,10)
def print_board():
    x=1
    for i in board:
        end = ' | '
        if x%3 == 0:
            end = ' \n'
            if i != 1: end+='---------\n';
        char=' '
        if i in ('X','O'): char=i;
        x+=1
        print(char,end=end)
def select_char():
    chars=('X','O')
    if random.randint(0,1) == 0:
        return chars[::-1]
    return chars
def can_move(brd, player, move):
    if move in tab and brd[move-1] == move-1:
        return True
    return False
def can_win(brd, player, move):
    places=[]
    x=0
    for i in brd:
        if i == player: places.append(x);
        x+=1
    win=True
    for tup in winners:
        win=True
        for ix in tup:
            if brd[ix] != player:
                win=False
                break
        if win == True:
            break
    return win

def make_move(brd, player, move, undo=False):
    if can_move(brd, player, move):
        brd[move-1] = player
        win=can_win(brd, player, move)
        if undo:
            brd[move-1] = move-1
        return (True, win)
    return (False, False)

def computer_move():
    move=-1
    for i in range(1,10):
        if make_move(board, computer, i, True)[1]:
            move=i
            break
    if move == -1:
        for i in range(1,10):
            if make_move(board, player, i, True)[1]:
                move=i
                break
    if move == -1:
        for tup in moves:
            for mv in tup:
                if move == -1 and can_move(board, computer, mv):
                    move=mv
                    break
    return make_move(board, computer, move)
def space_exist():
    return board.count('X') + board.count('O') != 9
player, computer = select_char()
print('Player is [%s] and computer is [%s]' % (player, computer))
result='%%% Deuce ! %%%'
while space_exist():
    print_board()
    print('#Make your move ! [1-9] : ', end='')
    move = int(input())
    moved, won = make_move(board, player, move)
    if not moved:
        print(' >> Invalid number ! Try again !')
        continue
 
    if won:
        result='*** Congratulations ! You won ! ***'
        break
    elif computer_move()[1]:
        result='=== You lose ! =='
        break;
print_board()
print(result)


# Write a python function to return zodiac sign given day and month of date of birth

def zodiac_sign(day, month): 

    if month == 'december': 
        astro_sign = 'Sagittarius' if (day < 22) else 'capricorn'
          
    elif month == 'january': 
        astro_sign = 'Capricorn' if (day < 20) else 'aquarius'
          
    elif month == 'february': 
        astro_sign = 'Aquarius' if (day < 19) else 'pisces'
          
    elif month == 'march': 
        astro_sign = 'Pisces' if (day < 21) else 'aries'
          
    elif month == 'april': 
        astro_sign = 'Aries' if (day < 20) else 'taurus'
          
    elif month == 'may': 
        astro_sign = 'Taurus' if (day < 21) else 'gemini'
          
    elif month == 'june': 
        astro_sign = 'Gemini' if (day < 21) else 'cancer'
          
    elif month == 'july': 
        astro_sign = 'Cancer' if (day < 23) else 'leo'
          
    elif month == 'august': 
        astro_sign = 'Leo' if (day < 23) else 'virgo'
          
    elif month == 'september': 
        astro_sign = 'Virgo' if (day < 23) else 'libra'
          
    elif month == 'october': 
        astro_sign = 'Libra' if (day < 23) else 'scorpio'
          
    elif month == 'november': 
        astro_sign = 'scorpio' if (day < 22) else 'sagittarius'
          
    print(astro_sign) 
	
	
# Write a function to get the Cumulative sum of a list
 
def Cumulative(lists): 
    cu_list = [] 
    length = len(lists) 
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)] 
    return cu_list[1:]
	
	
# Write a python program to perform Vertical Concatenation in Matrix  
  
test_list = [["India", "good"], ["is", "for"], ["Best"]] 
print("The original list : " + str(test_list)) 
res = [] 
N = 0
while N != len(test_list): 
    temp = '' 
    for idx in test_list:       

        try: temp = temp + idx[N] 
        except IndexError: pass
    res.append(temp) 
    N = N + 1
  
res = [ele for ele in res if ele] 
  
print("List after column Concatenation : " + str(res)) 


# Write a python program code to perform Triple quote String concatenation Using splitlines() + join() + strip() 
  

test_str1 = """India 
is"""
test_str2 = """best 
for everybody 
"""
  

print("The original string 1 is : " + test_str1) 
print("The original string 2 is : " + test_str2) 
  
test_str1 = test_str1.splitlines() 
test_str2 = test_str2.splitlines() 
res = [] 
  
for i, j in zip(test_str1, test_str2): 
    res.append("   " + i.strip() + " " + j.strip()) 
res = '\n'.join(res) 
  

print("String after concatenation : " + str(res))  


# Write a program to perform Consecutive prefix overlap concatenation Using endswith() + join() + list comprehension + zip() + loop 
  
def help_fnc(i, j): 
    for ele in range(len(j), -1, -1): 
        if i.endswith(j[:ele]): 
            return j[ele:] 
  

test_list = ["India", "gone", "new", "best"] 
  
print("The original list is : " + str(test_list)) 
  
res = ''.join(help_fnc(i, j) for i, j in zip([''] + 
                           test_list, test_list)) 
  
print("The resultant joined string : " + str(res)) 


# Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules: Each row/column/subbox must contain the digits 1-9 without repetition.

def isValidSudoku(board):
    rows = [{} for i in range(9)]
    columns = [{} for i in range(9)]
    boxes = [{} for i in range(9)]
        
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != '.':
                num = int(num)
                box_index = (i//3)*3 + (j//3)
                rows[i][num] = rows[i].get(num, 0) + 1
                columns[j][num] = columns[j].get(num, 0) + 1
                boxes[box_index][num] = boxes[box_index].get(num, 0) + 1       
                if rows[i][num] > 1 or columns[j][num] > 1 or boxes[box_index][num] > 1:
                    print(" i= {0} j = {1} box_index ={2}".format(i,j,box_index))
                    print("rows[i]: ", rows[i])
                    print("columnns[j]: ", columns[j])
                    print("boxes[box_index]: ", boxes[box_index])
                    return False
                    
    return True
	
	
# Write a python function to generate unique file names in a folder for a given list of file names

from collections import Counter

def getFolderNames(names):
    seen, res = Counter(), []
    for name in names:
        if name in seen:
            while True:
                c = f'({seen[name]})'
                if name + c not in seen:
                    name += c
                    break
                else:
                    seen[name] += 1
        seen[name] += 1
        res.append(name)
    return res
	

# Write a python program to convert complex number to polar coordinates

import cmath  
      
# using cmath.polar() method  
num = cmath.polar(1)  
print(num) 


# Write a python program to print calendar of a given year

import calendar 
  
year = 2019
print(calendar.calendar(year))   



# Write a python function to perform Matrix Chain multiplication i.e. Given a sequence of matrices, find the most efficient way to multiply these matrices together

import sys 

def MatrixChainOrder(p, i, j): 
  
    if i == j: 
        return 0
  
    _min = sys.maxsize 
      
    for k in range(i, j): 
      
        count = (MatrixChainOrder(p, i, k)  
             + MatrixChainOrder(p, k + 1, j) 
                   + p[i-1] * p[k] * p[j]) 
  
        if count < _min: 
            _min = count; 
      
  
    return _min; 

	


	
	

		

		


		

  




	
			
	
	
