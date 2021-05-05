#########################################################################3

def STR(): return list(input())
def INT(): return int(input())
def MAP(): return map(int, input().split())
def MAP2():return map(float,input().split())
def LIST(): return list(map(int, input().split()))
def STRING(): return input()
import string
import sys
import datetime
from heapq import heappop , heappush
from bisect import *
from collections import deque , Counter , defaultdict
from math import *
from itertools import permutations , accumulate
dx = [-1 , 1 , 0 , 0  ]
dy = [0 , 0  , 1  , - 1]
#visited = [[False for i in range(m)] for j in range(n)]
#  primes = [2,11,101,1009,10007,100003,1000003,10000019,102345689]
#months = [31 , 28 , 31 , 30 , 31 , 30 , 31 , 31 , 30 , 31, 30 , 31]
#sys.stdin = open(r'input.txt' , 'r')
#sys.stdout = open(r'output.txt' , 'w')
#for tt in range(INT()):
#arr.sort(key=lambda x: (-d[x], x)) Sort with Freq

#code

#find number of digits
#x = 12345
#print(1 + floor(log10(x)))
#global e , vx
#import sys
#sys.stdin = open(r'input.txt' , 'r')
#sys.stdout = open(r'output.txt' , 'w')
#g = defaultdict(list) #### Vector
#vis = [[0] * m for i in range(n)]
#vis = [[float('inf') for c in range(m)] for c2 in range(n)]
#swapcase() ## swap case of letters
#For capital alphabets 65 – 90
#For small alphabets 97 – 122
#For digits 48 – 57

# Python program for slope of line
#def slope(x1, y1, x2, y2):
    #return (float)(y2-y1)/(x2-x1)

################################################################
'''
def cmod(a , b):
    return (a % b + b) % b

def add_mod(a , b , m):
    return cmod(cmod(a , m) + cmod(b , m) , m)

def sub_mod(a,b,m):
    return cmod(cmod(a , m) - cmod(b , m) , m)

def mul_mood(a , b , m):
    return cmod(cmod(a , m) * cmod(b , m) , m)
'''

##############################################################

'''
##find Divisors
from  math import sqrt
def solve(x):

     res = []
     for i in range(2 , int(sqrt(x)) + 1):
         if x % i == 0 :
             if x // i == i :
                 res.append(i)

             else:
                 res.append(i)
                 res.append(x//i)

     return res'''

#############################################################################







###############################################################################33

####convert from Dicimal To bin

'''from collections import deque
def solve(x):
    ans  =deque()
    binary_num = [0] * x
    i = 0
    while x > 0:
        binary_num[i] = x % 2
        x = int(x/ 2)
        i +=1
    for j in range(i - 1 , -1 , - 1 ):
        ans.append(binary_num[j])
    return ans

x = int(input())
y =(solve(x))

for i in y :
    print(i , end  = '')'''

##############################################################################









########################################################################33
'''
##convert from Bin To Decimal

def solve(n):

    dec = 0
    num = n

    temp = n
    base = 1

    while temp :

        last_digit = temp % 10
        temp = int(temp / 10)

        dec += last_digit * base
        base *= 2
    return dec

n = int(input())

r = solve(n)

print(r)
#####################################################################








####################################################################
'''

'''###find all Substrings
from collections import deque

def find_all_substrings(s , n ):

    d =deque()
    for i in range(n):
        temp = ''
        for j in range(i , n ):
            temp += s[j]
            d.append(temp)

    return d

s = 'koko'
n = len(s)

print(find_all_substrings(s , n))'''
#############################################################################3

'''#Search Method
import bisect

def next(x , arr):
    z = bisect.bisect_left(arr , x)
    return z

'''



#########################################################3



'''#C. Dijkstra?
from heapq import heappush , heappop
def STR(): return list(input())
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))


n , m = MAP()
g = [[] for i in range(n)]
for i in range(m):
    u , v , w = MAP()
    u -= 1
    v -= 1
    g[u].append([v , w])
    g[v].append([u , w])

cost = [2**40 +9] * n
parents =[None]*n
q = [(0 , 0)] ## Cost , Node

while q :
    s1 , s2 = heappop(q)
    for u , w in g[s2]:
        if cost[u] > s1 + w :
            parents[u] = s2
            cost[u] = s1 + w
            heappush(q , (cost[u] , u))

#print(parents)
if parents[n - 1] == None:
    print(-1)
    exit(0)

ans = [n]
x = n - 1
while x != 0 :
    x = parents[x]
    ans.append(x + 1)

print(*ans[::-1])'''

#####################################################################3333

'''#Dijkstra

from  collections import defaultdict , deque

for tt in range(int(input())):
    n , m , source , goal = map(int,input().split())
    g = defaultdict(list)

    for i in range(m):
        u , v , w  = map(int,input().split())
        g[u].append([v , w])  ## Neighbour , edgecost
        g[v].append([u , w])

    cost = [-1]*(n + 9)
    #print(cost)

    q = deque()
    q.append([0 , source]) ### totalCost , node
    q = sorted(q , key= lambda x : x[0])
    while q:

        s1 , s2 = q.pop()
        if cost[s2] != -1 : continue
        cost[s2] = s1
        for i in g[s2]:
            if cost[i[0]] == -1:
                q.append([s1 + i[1] , i[0]] )


    print('Case #'+str(tt+1) + str(':') , end = ' ')
    if cost[goal] == -1 :
        print('unreachable')

    else:
        print(cost[goal])'''

##############################################################################

'''from collections import defaultdict , deque
import queue

for tt in range(int(input())):
   n , m  = map(int,input().split())
   g = defaultdict(list)
   for i in range(m):
       u , v , w  = map(int,input().split())
       g[u].append([v , w])  ## Neighbour , edgecost
       g[v].append([u , w])
   cost = [10**10]*(n + 9)
   source = int(input())
   cost[source] = 0
   q = queue.PriorityQueue()
   q.put((0 , source))  ### totalCost , node
   while not q.empty() :
       s1 , s2 = q.get() # cost , node
       for i in g[s2]:
           if cost[i[0]] > s1 + i[1]:
               cost[i[0]] = s1 + i[1]
               q.put((s1 + i[1] , i[0]))

   #print(cost)
   a = []
   for i in range(1 , n + 1 ):
       if i == source : continue
       else:a.append(cost[i])

   print(*a)'''





#################################################################################


'''
#Check_prime
def solve(n):

    if n <= 1  : return False
    if n == 2 or n == 3  : return True

    if n % 2 == 0 or n % 3 == 0 :
        return False

    i = 5
    while i * i <= n :
        if (n % i == 0 or n%(i+2) == 0):
            return False
        i += 6

    return True
'''


###############################################

'''def solve(n):
    d = []
    cnt1 = 0
    while n % 2 == 0 :
        n//=2
        cnt1 +=1

    d.append([2 , cnt1])
    i = 3
    while i * i <= n :
        cnt2 = 0
        while n % i == 0 :
            cnt2 +=1
            n//= i
        d.append([i , cnt2])
        i +=1

    if n > 2 :
        d.append([n , 1])
    return d


n = INT()
print(solve(n))

'''
#####################################################################
#power
'''#olog2(y)
def power(x , y):
    if y == 0 :
        return 1

    k = power(x , y // 2)
    k = k * k
    print(k)
    if (y % 2 != 0 ):
        k = k * x

    return k

print(power(2 , 5))

'''

#######################################################################
'''#rn a^1+a^1+a^2+.....a^k	in O(k).
def sumpow(n , k ):
    if k == 0 :
        return 0

    if (k % 2  != 0 ):
        return n * (1 + sumpow(n , k - 1))

    half = sumpow(n , k // 2)
    print(half)
    return half * (1 + half - sumpow(n , k // 2 - 1))


print(sumpow(2 , 6))

'''
############################################################333

'''
def fibo(n):
    if n == 0 :
        return 0
    elif n == 1 :
        return 1
    else:
        a , b = 0 , 1
        for i in range(2 , n + 1):
            c = a + b
            a = b
            b = c
        return b


def fibo2(n):
    arr = [0] * (n + 1)
    arr[0] = 0
    arr[1] = 1
    for i in range(2 , n + 1):
        arr[i] = arr[i - 1] + arr[i - 2]

    return arr

def fibo3(n):
    if n == 0 :
        return 0
    elif n == 1 :
        return 1

    else:
        return fibo3(n - 1) + fibo3(n - 2)


print(fibo3(5))


'''

#################################################################
'''
#rn a^1+a^1+a^2+.....a^k
def power(x  ,  y):
    if y == 0 :
        return 1

    k = power(x , y // 2)
    k = k * k
    if y % 2 != 0 :
        k = k * x

    return k

n , k = MAP()
sm = 0
for i in range(1 , k + 1):
    sm += power(n , i)

print(sm)'''



####################################################################
'''#o(NloglogN)
def solve(n):
    i = 2
    while i * i <= n :
        if min_prime[i] == 0 :
            for j in range(i * i , n + 1 , i ):
                if min_prime[j] == 0 :
                    min_prime[j] = i
        i += 1

    for i in range(2 , n+1):
        if min_prime[i] == 0 :
            min_prime[i] = i
    #return min_prime


def factorize(n):
    v = []
    while n != 1 :
        v.append(min_prime[n])
        k = min_prime[n]
        n//=k

    return v



n = INT()
min_prime = [0] * (n + 1)
solve(n)
#print(min_prime)
print(factorize(n))
'''
#########################################################

'''#O(sqrt(R))
# find all the primes that are in the range 
def solve(l ,r ):
    i = 2
    while i * i <= r :
        for j in range(max(i * i  , (l + (i - 1) // i * i )) , r + 1 , i ):
            print(i , j,(l + (i - 1) // i * i ))
            prime[j - l] = False
        i += 1
    return prime

l ,r = MAP()
prime = [True] * (r - l + 1)
print(solve( l , r))

'''



################################################################

'''
# Python program to print prime factors

import math
from collections import deque

def solve(n):
    d = deque()
    i = 0
    while n % 2 == 0 :
        n //=2
        i += 1

    if i > 0 :
        d.append((2 , i))

    for j in range(3 , int(math.sqrt(n)) + 1):

        k = 0
        while n % j == 0 :
            k +=1
            n //= j

        if k > 0 :
            d.append((j , k))

    if n > 2 :
        d.append((n , 1))

    return d

n = int(input())
s = solve(n)
print(s)

for i in s :
    print(i)
'''


########################################################################33






#########################################################################



'''
#SieveOfEratosthenes
def solve(n):
    prime = [True for i in range(n + 1)]
    prime[0] = prime[1] = False

    p = 2
    while p * p <= n:
        if prime[p]:
            i = p * p
            while i <= n:
                prime[i] = False
                i += p

        p += 1

    for i in range(2, n + 1):
        if prime[i]:
            print(i)


n = 10
solve(n)


'''


################################################################################









###########################################################################33
'''
##GCD
def gcd(a , b):
    if b == 0  :
        return a
    else:
        return gcd(b , a%b)

'''
######################################################
'''#fibo
def solve(n):
    a , b = 0, 1
    if n == 0 :
        return a
    if n ==1 :
        return b
    else:
        for i in range(2 , n + 1):
            c = a + b
            a = b
            b = c

        return b'''

##########################################################




'''from math import sqrt
from collections import deque
def divisors(n):
    res =deque()
    i = 1
    while i <= int(sqrt(n)):
        if n % i == 0 :
            if n // i == i :
                res.append(i)
            else:
                res.append(i)
                res.append(n // i)

        i +=1
    return  res

'''

###################################################################3


'''
#O(Log y)
def fast_power(x , y):

    res =  1
    while y > 0 :
        if y % 2 != 0 :
            print(x)
            res = res * x
            print(res)
            print()

        y = y // 2
        print(y)
        x = x * x
        print(x)
    return res

print(fast_power(2,5))
'''

###########################################
'''def fast_power2(x , y):

    if y == 0 :
        return 1

    temp = fast_power2(x , int(y / 2))
    if y % 2 == 0 :
        return temp * temp
    else:
        return x * temp * temp

print(fast_power2(2 , 5))'''







#######################################333

'''import heapq

l1 = [1 , 2 ,3 , 4, 5]
heapq.heapify(l1)
print(list(l1))

heapq.heappush(l1 , 10)
print(list(l1))

k = heapq.heappop(l1)
print(k)
'''
'''
#MST

# Class to represent a graph
class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])


    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)


        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot


        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def KruskalMST(self):

        result = []  # This will store the resultant MST

        i = 0
        e = 0

        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        MC = 0
        for u,v,w in result:
            MC += w
            print(u , v , w)
        print(MC)


g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)
g.KruskalMST()
'''

#########################################################################3

'''#check whether given graph is bipartite or not

from collections import defaultdict

res = [True]
def dfs(node):
    global res
    for child in g[node]:
        if colored[child] != -1 :
            if colored[child] == colored[node]:
                res[0] = False
                return
            continue
        colored[child] = 1 - colored[node]
        dfs(child)

g = defaultdict(list)
for i in range(10):
    u , v = map(int,input().split())
    g[u].append(v)

#print(g)
colored = [-1] * (5)

for i in range(5):
    if colored[i] == -1 :
        dfs(i)

print(res)'''

#########################################################################3

'''
#isCyclic
from collections import defaultdict
# This class represents a undirected graph using adjacency list representation
class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = defaultdict(list)  # default dictionary to store graph

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A utility function to find the subset of an element i
    def find_parent(self, parent, i):
        if parent[i] == -1:
            return i
        if parent[i] != -1:
            return self.find_parent(parent, parent[i])

    # A utility function to do union of two subsets
    def union(self, parent, x, y):
        x_set = self.find_parent(parent, x)
        y_set = self.find_parent(parent, y)
        parent[x_set] = y_set

    # The main function to check whether a given graph
    # contains cycle or not
    def isCyclic(self):

        # Allocate memory for creating V subsets and
        # Initialize all subsets as single element sets
        parent = [-1] * (self.V)

        # Iterate through all edges of graph, find subset of both
        # vertices of every edge, if both subsets are same, then
        # there is cycle in graph.
        for i in self.graph:
            for j in self.graph[i]:
                #print(i , j)
                x = self.find_parent(parent, i)
                y = self.find_parent(parent, j)
                #print(x , y)
                if x == y :
                    return True
                self.union(parent,x,y)

g = Graph(3)
g.addEdge(0 ,1)
g.addEdge(1 ,2)
g.addEdge(2 ,0)

r = g.isCyclic()
print(r)
'''

#########################################################################3



'''
#Merge_Sort
def mergeSort(arr):
    if len(arr) > 1:

        # Finding the mid of the array
        mid = len(arr) // 2

        # Dividing the array elements
        L = arr[:mid]

        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        mergeSort(L)

        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
        print(arr)
    return arr'''


#################################################333
'''
# merge_Sort and number of swaps to sort array with swap adjasnt numbers
def solve(arr, temp, left, mid, right):
    cnt = 0
    i = left
    j = mid
    k = left
    while (i <= mid - 1) and (j <= right):
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
            k += 1
        else:
            temp[k] = arr[j]
            k += 1
            j += 1
            cnt += (mid - i)

    while (i <= mid - 1):
        temp[k] = arr[i]
        k += 1
        i += 1

    while (j <= right):
        temp[k] = arr[j]
        k += 1
        j += 1
    for i in range(left, right + 1, 1):
        arr[i] = temp[i]
    return cnt

def solve2(arr, temp, left, right):
    cnt = 0
    if right > left:
        mid = (right + left) // 2
        cnt = solve2(arr, temp, left, mid)
        cnt += solve2(arr, temp, mid + 1, right)
        cnt += solve(arr, temp, left, mid + 1, right)
    return cnt
def solve3(arr, n):
    temp = [0] * n
    return solve2(arr, temp, 0, n - 1)

for tt in range(INT()):
    n = INT()
    arr = LIST()
    ans = solve3(arr, n)
    print(ans)

'''
#######################################


'''

#From Decimal To bin
def f(n):
    d = ''
    while n >0 :
        x = n % 2
        d+=str(x)
        n //= 2
    return d[::-1]'''

#############################################################

'''#Bubble_Sort
#O(n)
arr = [1 , 3 , 8  , 4 ,5 ,2 , 9 , 6 , 7 ,10]
n = len(arr)
isswaped = True
i = 0
while isswaped :
    isswaped = False
    for j in range(0 , n - i - 1):
        if arr[j] > arr[j + 1]:
            arr[j] , arr[j + 1] = arr[j + 1] , arr[j]
            isswaped = True
    i+=1

print(arr)'''

#####################

'''
#Bubble_Sort
#O(n^2)
arr = [1 , 3 , 4 ,5 ,2 , 9 , 6 , 7 ,10]
n = len(arr)

for i in range(n):
    for j in range(0 , n - i - 1):
        if arr[j] > arr[j + 1] :
            arr[j] , arr[j + 1]= arr[j + 1] , arr[j]

print(arr)

'''
#######################################################

'''#selection_sort
#O(n^2)
arr = [10 , 6 , 9 , 1 , 5  , 100 , 7 , 4 , 2 , 8 , 3]
n = len(arr)

for i in range(n):
    mn = 100000000000000
    index = -1
    for j in range(i , n):
        if arr[j] < mn :
            mn = arr[j]
            index = j

    arr[i] , arr[index] = arr[index] , arr[i]

print(arr)
'''

#####################################################

'''#insertion_Sort
#O(n^2)
arr = [10 , 6 , 2 , 7 , 9 , 3 , 8 , 1 , 4 , 5]
n = len(arr)

for i in range(n):
    k = arr[i]
    j = i - 1
    while j >= 0 and arr[j] > k :
        arr[j + 1] = arr[j]
        j-=1

    arr[j + 1] = k

print(arr)
'''

##############################################


'''#minimum Number of coins
#O(1)
x = INT()
cnt = 0
cnt += x // 100
x = x % 100
cnt += x // 50
x = x % 50
cnt += x // 10
x = x % 10
cnt += x // 1
x = x % 1

print(cnt)
'''

#############################################################

'''
def power(x , n):
    if n == 1 :
        return x

    res = power(x , n//2)
    if n % 2 == 0 :
        return res * res
    else:
        return res * res * x

print(power(2 , 4))
print(power(2 , 5))
'''

###################################################

'''#Time Complexity of above solution is O(Log y).
#Modular Exponentiation
def moduler_power(x , y , p ):
    res = 1
    if x == 0 :
        return 0

    x %= y
    while y > 0 :
        if y % 2 != 0 :
            res = (res * x) % p

        y = y // 2
        x = (x * x) % p

    return res

print(moduler_power(2,5,13))

'''
#############################################################
'''#Extended Euclidean Algorithm
def gcdExtended(a, b):

    if b == 0:
        print(True)
        return a, 1 , 0

    else:
        g , x1, y1 = gcdExtended(b, a % b)
        print(g,x1,y1)
        x = y1
        y = x1 - (a // b ) * y1 
    return g, x, y

print(gcdExtended(16 , 10))
'''


####################################################################


'''#Fractional_Napsack_Problem_With_Greedy
capacity = 50
weights = [10 , 20 , 30]
values = [60 , 100 , 120]

x = []
for i in range(len(weights)):
    x.append([values[i] // weights[i] , values[i] , weights[i]])

x.sort(reverse= True , key= lambda  x : x[0])
print(x)
mx_Sum = 0
for i in range(len(x)):
    if capacity >= x[i][2]:
        mx_Sum += x[i][1]
        capacity -= x[i][2]

    else:
        mx_Sum += capacity * x[i][0]
        capacity = 0

    if capacity == 0 :
        break

print(mx_Sum)
'''

################################################################

'''#Activity_Selection_With_Greedy
start = [12 , 10 , 20]
finish = [25 , 20 , 30]

x = []
for i in range(len(start)):
    x.append([start[i] , finish[i]])

x.sort(key= lambda x : x[0])
print(x)
cnt = 0
last = -1

for i in range(len(x)):
    if last == -1 :
        cnt += 1
        last = x[i][1]

    else:
        if x[i][0] >= last:
            cnt+=1
            last = x[i][1]

print(cnt)
'''
#############################################################

'''#DP
def fibo(n):

    arr[1] = 1
    arr[2] = 1

    for i in range(3 , n + 1):
        arr[i] = arr[i - 1] + arr[i - 2]

    return arr[n]

arr = [0] * (101)
print(fibo(7))
'''

##################################################################


'''#Napsack_Problem_DP
def solve(i , c , w , v):
    if i == len(w) :
        return 0

    if (c < w[i]) :
        dp[i][c] = solve (i + 1 , c , w , v )
        return dp[i][c]

    else:
        dp[i][c] =  max(solve(i + 1 , c , w , v) , (v[i] + solve(i + 1 , c - w[i] , w, v)))
        return dp[i][c]

capacity = 50
weights = [10 , 20 , 30]
values = [60 , 100 , 120]

dp =[[-1]*100]*100

print(solve(0 , capacity , weights , values))
'''

###################################################################################3
'''
#MAx_SUM_IN_MAtrix
n = int(input())
mem = [[-1]*(n+1)]*(n + 1)
grid = []
for i in range(n):
    grid.append(list(map(int,input().split())))
#print(mem)

def max_sum(r , c):

    #check_valid
    if (r < 0 or c < 0 or r >= n or c >= n):
        return 0

    #base_case
    if (r == n - 1 and c == n - 1 ):
        return grid[r][c]

    if (mem[r][c] != -1):
        return mem[r][c]

    else:

        path1 = max_sum(r + 1 , c)
        path2 = max_sum(r , c + 1)

        mem[r][c] =  grid[r][c] + max(path1 , path2)
        return mem[r][c]


print(max_sum(0 , 0 ))
'''

########################################################

'''#O(n)
def ways_to_climb(n):

    if n <= 1 : return 1
    if n == 2 : return 2

    res = [0] * (n+ 1)
    res[0] = 1
    res[1] = 1
    res[2] = 2

    for i in range(3 , n + 1):
        res[i] = res[i - 1] + res[i-2] + res[i - 3]

    return res[n]

n= int(input())

print(ways_to_climb(n))'''

#############################################################
'''
#fast_power
def solve(x , y):

    res = 1
    while y > 0 :
        if y % 2 != 0 :
            res *= x

        x *= x
        y//=2
    return res


print(solve(2 , 4))

'''

##############################################################

'''
#Shortest Base From source with BFS in undirected_graph and construct Path
def BFS(src ):
    l = {srce : src}
    d = deque([srce])
    while d :
        s = d.popleft()
        c = cost[s]
        for i in graph[s]:
            if cost[i]==-1:
                l[i] = s
                cost[i] = c + 1
                d.append(i)
    return l

n, m = MAP()
graph = defaultdict(list)
cost = [-1] * (n + 1)
srce , goal  = MAP()
cost[srce] = 0
for i in range(m):
    u,v = MAP()
    graph[u].append(v)
    graph[v].append(u)

r = BFS(1)
#print(r)
#print(cost)
path  = [goal]
for i in range(goal , 0,-1):
    x = r[i]
    path.append(x)
    if x == 1 :
        break
print(cost[goal])
print(path[::-1])

'''

'''#Number OF Components with queue
def solve(node):
    s = deque()
    s.append(node)
    while s :
        x = s.popleft()
        visited[x] = True
        for i in graph[x]:
            if not visited[i]:
                s.append(i)

n , m = MAP()
graph = defaultdict(list)
ans = 0
for i in range(m):
    u,v = MAP()
    graph[u].append(v)
    graph[v].append(u)

visited = [False] * (n + 1)
for i in range(1 , n + 1):
    if not visited[i]:
        ans+=1
        solve(i)

print(ans)
'''


'''#Number OF Components with DFS
def DFS(node):
    visited[node] = True
    #print(node, end=' ')
    for i in graph[node]:
        if not visited[i]:
            DFS(i)

n , m = MAP()
graph = defaultdict(list)
ans = 0
for i in range(m):
    u,v = MAP()
    graph[u].append(v)
    graph[v].append(u)

visited = [False] * (n + 1)

for i in range(1 , n + 1):
    if not visited[i]:
        ans+=1
        DFS(i)
print()
print(ans)
'''
###########################################################





















