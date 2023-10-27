#Luke Stodgel
#DS7335
#Jan 24, 2023
import matplotlib.pyplot as plt
import numpy as np

#list functions
print("\nLists\n")
names = ['Tom', 'John']
print("starting list, names: ", names, "\n")

#append()
names.append('Luke')
print("append() adds an item to the end of a list. \nex."
      "\nnames.append('Luke') \nprint(names) \nOutput:", names, "\n")

#extend()
numbers = [1, 2, 3]
names.extend(numbers)
print("new list: numbers = ", numbers)
print("extend() adds all elements of an iterable to the end of a list."
      "\nex. \nprint(names.extend(numbers)) \nOutput:", names, "\n")

#index()
print("Current list:", names)
print("index() returns the index of an element in a list. \nex."
      "\nprint(names.index(\"Tom\")) \nOutput:", names.index("Tom"), "\n")

#index(value, integer)
print("index(value, integer) returns the first occurence of an item after "
      "a given index. \nex. \nprint(names.index(\"Luke\", 0)) \nOutput:", 
      names.index("Luke", 0), "\n")

#insert(position)
names.insert(0, "hi")
print("insert(position) inserts an item at a given index. \nex. "
      "\nnames.insert(0, \"hi\") \nprint(names) \nOutput:", names, "\n")

#remove()
names.remove("hi")
print("remove() removes an item from a list. \nex. \nnames.remove(\"hi\")"
      " \nprint(names) \nOutput:", names, "\n")

#pop()
names.pop()
print("pop() removes one item at the end of a list. \nex. \nnames.pop() "
      "\nprint(names) \nOutput:", names, "\n")

#count()
print("count() provides the number of occurences of an item in a list. "
      "\nex. \nprint(names.count(\"Tom\")) \nOutput:", names.count("Tom"),
      "\n")

#reverse()
names.reverse()
print("reverse() reverses the order of a list. \nex. \nnames.reverse() "
      "\nprint(names) \nOutput:", names, "\n")

#sort()
nums = [3, 2, 1]
print("new list, nums:", nums)

nums.sort()
print("sort() sorts the items in a list in ascending or descending order. "
      "\nex. \nnums.sort() \nprint(nums) \nOutput:", nums, "\n")

#[1]+[1]
print("[1]+[1] is an example of list concatenation. [1]+[1] will result "
      "with the list [1, 1]. \nex. \nprint([1]+[1]) \nOutput:", [1]+[1], "\n")

#[2]*2
print("[2]*2 will result in a list with the number of occurrences of the "
      "original list multiplied by whatever factor you want. Here we used a "
      "factor of 2. \nex. \nprint([2]*2) \nOutput:", [2]*2)
print("[2, 2]*2 for example will result in a list containing four twos in it."
      " \nex. \nprint([2, 2]*2) \nOutput:", [2, 2]*2, "\n")

#[1,2][1:]
print("[1,2][1:] is an example of list splicing. In our example, it reads, "
      "\"Given list [1,2], start at index 1, go until the end of the list and"
      " output that list.\" The result of [1,2][1:] will be [2]. \nex. "
      "\nprint([1,2][1:]) \nOutput:", [1,2][1:], "\n")

#[x for x in [2,3]]
print("[x for x in [2,3]] is an example of list comprehension. Our example "
      "will create a new list containing each x in [2,3], resulting in [2,3]."
      " \nex. \nprint([x for x in [2,3]]) \nOutput:", [x for x in [2,3]], "\n")

#[x for x in [1,2] if x ==1]
print("[x for x in [1,2] if x ==1] is also a list comprehension. It reads,"
      "\"create a list with each x in [1,2] if x = 1 is true.\". The result "
      "will be = [1]. \nex. \nprint([x for x in [1,2] if x ==1]) \nOutput:", 
      [x for x in [1,2] if x ==1], "\n")

#[y*2 for x in [[1,2],[3,4]] for y in x]
print("[y*2 for x in [[1,2],[3,4]] for y in x] is a nested list comprehension"
      " that will output a list. It will multiply each item (1, 2, 3, 4) "
      "within each item [1,2], [3,4] by 2. The output will be [2, 4, 6, 8]. "
      "\nex. \nprint([y*2 for x in [[1,2],[3,4]] for y in x]) \nOutput:", 
      [y*2 for x in [[1,2],[3,4]] for y in x], "\n")

#A = [1]
A = [1]
print("A = [1] is an example of assigning a variable, in our case, \"A\", "
      "to a list with a 1 in it. \nex. \nA = [1] \nprint(A) \nOutput:", A, 
      "\n")


#Tuple:
print("\nTuples\n")
x = (1, 1, 1)
print("New tuple x:", x, "\n")

#count()
print("count() will print out the number of occurences of a specific "
      "item in a tuple. \nex. \nprint(x.count(1)) \nOutput:", x.count(1), 
      "\n")

#index()
print("index() will output the first occurence of a value passed into the "
      "function. In our case, index(1) will return 0. \nex. "
      "\nprint(x.index(1)) \nOutput:", x.index(1), "\n")

#build a dictionary from tuples
new_tuple = (('a', 1), ('b', 2))
print("new_tuple:", new_tuple, "\n")
new_tuple_dict = dict(new_tuple)
print("To create a dictionary using a tuple, we use the dict() function. "
      "\nnew_tuple_dict = dict(new_tuple) \nprint(new_tuple_dict) \nOutput:"
      , new_tuple_dict, "\n")

#unpack tuples
tuple1 = (1, 2, 3)
x, y, z = tuple1
print("Tuple unpacking is when you assign items in a tuple to multiple "
      "variables. You can also unpack variables into a list or a set."
      "\ntuple1 = (1, 2, 3) \nx, y, z = tuple1 \nprint(x):", x, "\nprint(y):"
      , y,"\nprint(z):", z, "\n")


#Dicts:
print("\nDictionaries", "\n")
#a_dict = {'I hate':'you', 'You should':’leave’}
a_dict = {'I hate':'you', 'You should':'leave'}
print("a_dict = {'I hate':'you', 'You should':'leave'} will create a "
      "dictionary with two key:value pairs. \nex. \na_dict = {'I hate':'you', "
      "'You should':'leave'}\nprint(a_dict) \nOutput: ", a_dict, "\n")

#keys()
print("keys() will print out the keys in a dictionary. \nex. "
      "\nprint(a_dict.keys()) \nOutput:", a_dict.keys(), "\n")

#items()
print("items() will output a view object containing a list of a dictionary's"
      " key:value pairs. \nex. \nprint(a_dict.items()) \nOutput:",
      a_dict.items(), "\n")

#hasvalues()
print("hasvalues() does not exist but values() does. values() returns a "
      "view object containing all of the values of all the key:value pairs "
      "in a dictionary. \nex. print(a_dict.values()) \nOutput:",
      a_dict.values(), "\n")

#_key()
print("There does not appear to be a _key() built in function in python.",
      "\n")

#'never' in a_dict
print("'never' in a_dict - we can check if a key or value is in a_dict by "
      "using the 'in' keyword. \nex.\nif 'never' in a_dict.keys() \n  "
      "print(\"\'never\' is a key in a_dict\")\nelse:\n  print(\"\'never\' "
      "is not a key in a_dict\")")
print("Output: ")
if 'never' in a_dict.keys():
    print("\'never\' is a key in a_dict")
else:
    print("\'never\' is not a key in a_dict", "\n")

#del a_dict['me']
print("del a_dict['me'] will attempt to delete the key:value pair in a_dict"
      " with the key 'me'. If 'me' doesn't exist, it will raise a KeyValue "
      "error. \nex. \nif 'me' in a_dict: \n  del a_dict['me']",  "\n")
if 'me' in a_dict: 
    del a_dict['me']
        
#a_dict.clear()
print("a_dict.clear() will remove all of the key:value pairs in a "
      "dictionary. The method takes no arguments and returns None. \nex."
      " print(a_dict.clear()) \nOutput:", a_dict.clear(), "\n")


#Sets:
new_set = {1, 2, 3}
print("\nSets\n\nnew_set:", new_set, "\n")

#add()
new_set.add(4)
print("add() will add an element to a set. You can only have unique "
      "elements in sets. \nex. \nnew_set.add(4) \nprint(new_set) \nOutput:",
      new_set, "\n")

#clear()
new_set.clear()
print("clear() will remove all items from a set. \nex. \nnew_set.clear()"
      " \nprint(new_set) \nOutput:", new_set, "\n")
new_set = {1, 2, 3}

#copy()
copy_new_set = new_set.copy()
print("copy() is used to create a shallow copy of a set. \nex. "
      "\ncopy_new_set = new_set.copy() \nprint(copy_new_set) \nOutput:",
      copy_new_set, "\n")

#difference()
new_set = {1, 2, 3} 
new_set2 = {2, 3, 4}
new_set3 = new_set.difference(new_set2)
print("difference() returns a new set containing the elements that are "
      "in the first set but not in the second set. \nex. \nnew_set = {1, 2, 3}"
      " \nnew_set2 = {2, 3, 4} \nnew_set3 = new_set.difference(new_set2) "
      "\nprint(new_set3) \nOutput:", new_set3, "\n")

#discard()
new_set.discard(1)
print("discard() will remove an element from a set. If the element does "
      "not exist it will throw and error. \nex. \nset1 = {1, 2, 3} "
      " \nnew_set.discard(1) \nprint(new_set) \nOutput:", new_set, "\n")

#intersection()
set1 = {1, 2, 3} 
set2 = {2, 3, 4} 
set3 = set1.intersection(set2)
print("intersection() returns a new set containing the elements that "
      "are common to all the sets. \nex. \nset1 = {1, 2, 3} \nset2 = "
      "{2, 3, 4} \nset3 = set1.intersection(set2) \nprint(set3) \nOutput:",
      set3, "\n")

#issubset()
set1 = {1, 2} 
set2 = {1, 2, 3}
print("issubset() returns a Boolean indicating whether all elements of "
      "the set are present in another set. \nex. \nset1 = {1, 2} \nset2 = "
      "{1, 2, 3} \nprint(set1.issubset(set2)) \nOutput:", 
      set1.issubset(set2), "\n")

#pop()
set1 = {1, 2, 3}
set1.pop()
print("pop() removes and returns an arbitrary element from a set. If the "
      "set is empty, the method will raise a KeyError. \nex. \nset1 = "
      "{1, 2, 3} \nset1.pop() \nprint(set1) \nOutput:", set1, "\n")

#remove()
set1 = {1, 2, 3} 
set1.remove(2)
print("remove() removes an element from a set. If the set is empty, the "
      "method will raise a KeyError. \nex. \nset1 = {1, 2, 3} "
      "\nset1.remove(2) \nprint(set1) \nOutput:", set1, "\n")

#union()
set1 = {1, 2, 3}
set2 = {2, 3, 4}
set3 = set1.union(set2)
print("union() returns a new set containing all the elements from the "
      "original set and all the elements from one or more other sets "
      "passed as arguments. Sets contain only unique values so duplicates "
      "will not be in the resulting set. \nex. \nset1 = {1, 2, 3} \nset2 = "
      "{2, 3, 4} \nset3 = set1.union(set2) \nprint(set3) \nOutput:", 
      set3, "\n")

#update()
set1 = {1, 2, 3}
set2 = {2, 3, 4}
set1.update(set2)
print("update() is used to add elements from one or more other sets "
      "(or any iterable) to an existing set. \nex. \nset1 = {1, 2, 3} "
      "\nset2 = {2, 3, 4} \nset1.update(set2) \nprint(set1) \nOutput:", 
      set1, "\n")


#Strings:

#capitalize()
s = 'hello world'
s = s.capitalize()
print("capitalize() will convert the first character of a string to "
      "uppercase and all other characters to lowercase. \nex. \ns = "
      "'hello world' \ns = s.capitalize() \nprint(s) \nOutput:", s, "\n")

#casefold()
s = 'Hello World'
s = s.casefold()
print("casefold() will convert all the characters in a string to their "
      "lowercase form. \nex. \ns = 'Hello World' \ns = s.casefold() "
      "\nprint(s) \nOutput:", s, "\n")

#center()
s = 'hi'
s = s.center(10)
print("center() is used to center a string within a specified width by "
      "adding padding characters to the left and right of the string. "
      "\nex. \ns = 'hi'\ns = s.center(10) \nprint(s) \nOutput:", s, "\n")

#count()
s = 'hello, world'
print("count() returns the number of occurrences of a specified substring "
      "in the string. \nex. \ns = 'hello, world' \nprint(s.count('l')) "
      "\nOutput:", s.count("l"), "\n")

#encode()
s = 'hello, world'
b = s.encode()
print("encode() encodes a string into bytes using a specified encoding. "
      "\nex. \ns = 'hello, world' \nb = s.encode() \nprint(b) \nOutput:",
      b, "\n")

#find()
s = 'hello, world'
print("find() returns the index of the first occurrence of a substring "
      "in a string. \nex. \ns = 'hello, world' \nprint(s.find(\"l\")) "
      "\nOutput:", s.find("l"), "\n")

#partition()
s = 'hello, world'
print("partition() is used to search for a specified separator in a string."
      " It returns a tuple containing the part of the string before the "
      "separator, the separator itself, and the part of the string after "
      "the separator. \nex. \ns = 'hello, world' \nprint(s.partition(\",\"))"
      " \nOutput:", s.partition(","), "\n")

#replace()
s = 'hello, world'
s = s.replace("l", "x")
print("replace() replaces all occurrences of a specified substring in a "
      "string with another substring. \nex. \ns = 'hello, world' \ns = "
      "s.replace(\"l\", \"x\") \nprint(s) \nOutput:", s, "\n")

#split()
s = 'hello, world'
l = s.split(",")
print("split() splits a string into a list of substrings using a specified "
      "separator. \nex. \ns = 'hello, world' \nl = s.split(\",\") \nprint(l)"
      " \nOutput:", l, "\n")

#title()
s = 'hello, world'
s = s.title()
print("title() converts the first character of each word in a string to "
      "uppercase and all other characters to lowercase. \nex. \ns = "
      "'hello, world' \ns = s.title() \nprint(s) \nOutput:", s, "\n")

#zfill()
s = 'hi'
s = s.zfill(5)
print("zfill() will pad a string with a specified character (0 by default) "
      "on the left, until the specified width is reached. \nex. \ns = 'hi' "
      "\ns = s.zfill(5) \nprint(s) \nOutput:", s, "\n")


from collections import Counter
#print(dir(Counter))
print("\ncollections.Counter class - 10 most used functions.\n")

#elements()
c = Counter(a=2, b=2)
print("elements() returns an iterator over elements repeating each as many "
      "times as its count. \nex. \nc = Counter(a=2, b=2) "
      "\nprint(list(c.elements())) \nOutput:", list(c.elements()), "\n")

#most_common()
c = Counter(a=1, b=2, c=3, d=4, e=5)
print("most_common() returns a list of the n most common elements and "
      "their counts from most common to least common. \nex. \nc = "
      "Counter(a=1, b=2, c=3, d=4, e=5) \nprint(c.most_common()) "
      "\nOutput:", c.most_common(), "\n")

#subtract()
c1 = Counter(a=3, b=1, c=2)
c2 = Counter(a=1, b=2, d=3)
c1.subtract(c2)
print("subtract() takes an iterable (e.g. list, tuple, dict) or another "
      "counter and subtracts the elements count from the current counter. "
      "\ex. \nc1 = Counter(a=3, b=1, c=2) \nc2 = Counter(a=1, b=2, d=3) "
      "\nc1.subtract(c2) \nprint(c1) \nOutput:", c1, "\n")

#update()
c1 = Counter(a=2, b=1, c=0)
c2 = Counter(a=1, b=2, d=3)
c1.update(c2)
print("update() takes an iterable (e.g. list, tuple, dict) or another "
      "counter and adds the elements count to the current counter. \nex. "
      "\nc1 = Counter(a=2, b=1, c=0) \nc2 = Counter(a=1, b=2, d=3) "
      "\nc1.update(c2) \nprint(c1) \nOutput:", c1, "\n")

#clear()
c1 = Counter(a=1, b=2, c=3)
c1.clear()
print("clear() removes all elements from the counter. \nex. \nc1 = "
      "Counter(a=1, b=2, c=3) \nc1.clear() \nprint(c1) \nOutput:", c1, 
      "\n")

#items()
c = Counter(a=1, b=2, c=3)
print("items() returns a view of the counter's items in the form of a "
      "list of (element, count) pairs. \nex. \nc = Counter(a=1, b=2, c=3) "
      "\nprint(c.items()) \nOutput:", c.items(), "\n")

#keys()
c = Counter(a=1, b=2, c=3)
print("keys() returns a view of the counter's keys in the form of a list "
      "of elements. \nex. \nc = Counter(a=1, b=2, c=3) \nprint(c.keys())"
      " \nOutput:", c.keys(), "\n")

#values()
c = Counter(a=1, b=2, c=3)
print("values() returns a view of the counter's values in the form of a "
      "list of counts. \nex. \nc = Counter(a=1, b=2, c=3) "
      "\nprint(c.values()) \nOutput:", c.values(), "\n")

#setdefault()
c = Counter(a=1, b=2, c=3)
print("setdefault() returns the count of an element in the counter. "
      "If that element does not exist, it is added to the counter with "
      "a default count of 0 or the count passed as the second argument. "
      "\nex. \nc = Counter(a=1, b=2, c=3) \nprint(c.setdefault('a', 5))"
      " \nOutput:", c.setdefault('a', 5), "\n")

#get()
c = Counter(a=1, b=2, c=3)
print("get() returns the count of an element in the counter. If no element "
      "is present then it will return \"None\". \nex. \nc = Counter(a=1, b=2,"
      " c=3) \nprint(c.get(\"a\")) \nOutput:", c.get('a'), "\n")

#from itertools import * (Bonus: this one is optional, but recommended)

#Q2.

print("\nQ2\n")
flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
              'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
              'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
              'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R',
              'W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R',
              'W/R','W/R','W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y',
              'R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V',
              'W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V',
              'W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V',
              'W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y',
              'W/R/B/Y','W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y',
              'R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y',
              'W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/G','W/G','W/G',
              'W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y',
              'N/R/V/Y','W/R/B/V','W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y',
              'W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y',
              'W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y',
              'R/B/V/Y','R/B/V/Y','W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y',
              'W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y',
              'R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y',
              'W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y',
              'W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y',
              'V','N/R/V','N/V/Y','R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O',
              'W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O',
              'W/B/V/M','W/B/V/O','N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G',
              'N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M',
              'W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']

#1. Build your own counter object, then use the built-in Counter() and 
#   confirm they have the same values.
print("#1 Build your own counter object, then use the built-in Counter() and"
      " confirm they have the same values.")

counter = {}
for item in flower_orders:
    if item not in counter:
        counter[item] = 0
    counter[item] += 1

print(counter,"\n")

color_count = Counter(flower_orders)
print(color_count, "\n")

#2. Count how many objects have color W in them.
print("#2 Count how many objects have color W in them.")

count = 0
for item in flower_orders:
    if "W" in item:
        count += 1
print(count, "\n")

#3. Make histogram of colors
print("#3 Make histogram of colors")

counter = {}
for string in flower_orders:
    for letter in string:
        if letter in counter:
            counter[letter] += 1
        else:
            counter[letter] = 1

print(counter)

#create the histogram using the counter dictionary
plt.bar(counter.keys(), counter.values())
#add labels and title
plt.xlabel('Letters')
plt.ylabel('Counts')
plt.title('Letter Counts Histogram')
#display the histogram
plt.show()

## Hint from JohnP - Itertools has a permutation function that might 
#help with these next two.
#4. Rank the pairs of colors in each order regardless 
#   of how many colors are in an order.
print("\n#4- - Top 10 ranked pair permutations from colors list")
import itertools
from collections import Counter

pair_count = Counter()
for order in flower_orders:
    color_list = order.split("/")
    pairs = itertools.permutations(color_list, 2)
    for pair in pairs:
        pair_count[pair] += 1

top_10_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:10]
print(top_10_pairs)

#5. Rank the triplets of colors in each order regardless of how many 
#   colors are in an order.
print("\n#5 - Top 10 ranked triplet permutations from colors list")
import itertools
from collections import Counter

triplet_counts = Counter()
for order in flower_orders:
    color_triplets = order.split('/')
    triplets = itertools.permutations(color_triplets, 3)
    for triplet in triplets:
        triplet_counts[triplet] += 1

top_10_triplets = sorted(triplet_counts.items(), key=lambda x: x[1], 
                         reverse=True)[:10]

print(top_10_triplets)

#6. Make a dictionary with key=”color” and values = “what other colors it 
#   is ordered with”.
print("\n#6 Make a dictionary with key=”color” and values ="
      "“what other colors it is ordered with”.")
from collections import defaultdict

color_dict = defaultdict(set)
for order in flower_orders:
    for color in order.split('/'):
        for c in order.split('/'):
            if c != color:
                color_dict[color].add(c)

print(color_dict)

#7. Make a graph showing the probability of having an edge between two colors
# based on how often they co-occur. (a numpy square matrix)
print("\n#7 Make a graph showing the probability of having an "
      "edge between two colors based on how often they co-occur."
      " (a numpy square matrix)")

colors = set()
for order in flower_orders:
    for color in order.split("/"):
        colors.add(color)
colors = list(colors)

matrix = np.zeros((len(colors), len(colors)))

for order in flower_orders:
    color_list = order.split("/")
    for i in range(len(color_list)):
        for j in range(i+1, len(color_list)):
            matrix[colors.index(color_list[i]), 
                   colors.index(color_list[j])] += 1
            matrix[colors.index(color_list[j]), 
                   colors.index(color_list[i])] += 1

for i in range(len(colors)):
    matrix[i] /= matrix[i].sum()
print(matrix)

#8. Make 10 business questions related to the questions we asked above.
print("\n#8 Make 10 business questions related to the questions we asked"
      "above.\na.) What is the most common color among the flower orders?"
      " \nb.) Are there any color combinations that appear more "
      "frequently than others? \nc.) How does the frequency of flower orders"
      " change by color? \nd.) Is there a relationship between the number of"
      " colors in a flower order and its frequency? \ne.) How does the"
      " distribution of white flowers among the different color"
      " combinations vary? \nf.) What is the proportion of flower orders"
      " with red and white colors? \ng.) Are there any color combinations"
      " that are unique to the flower orders? \nh.) How has the frequency"
      " of flower orders with red, white and blue colors changed over time? "
      "\ni.) How does the frequency of flower orders change by season?"
      " \nj.) What is the impact of promotions on flower orders with "
      "certain color combinations?")


#Q3.

print("\nQ3\n")

dead_men_tell_tales = [
    'Four score and seven years ago our fathers brought forth on this',
'continent a new nation, conceived in liberty and dedicated to the',
'proposition that all men are created equal. Now we are engaged in',
'a great civil war, testing whether that nation or any nation so',
'conceived and so dedicated can long endure. We are met on a great',
'battlefield of that war. We have come to dedicate a portion of',
'that field as a final resting-place for those who here gave their',
'lives that that nation might live. It is altogether fitting and',
'proper that we should do this. But in a larger sense, we cannot',
'dedicate, we cannot consecrate, we cannot hallow this ground.',
'The brave men, living and dead who struggled here have consecrated',
'it far above our poor power to add or detract. The world will',
'little note nor long remember what we say here, but it can never',
'forget what they did here. It is for us the living rather to be',
'dedicated here to the unfinished work which they who fought here',
'have thus far so nobly advanced. It is rather for us to be here',
'dedicated to the great task remaining before us--that from these',
'honored dead we take increased devotion to that cause for which',
'they gave the last full measure of devotion--that we here highly',
'resolve that these dead shall not have died in vain, that this',
'nation under God shall have a new birth of freedom, and that',
'government of the people, by the people, for the people shall',
'not perish from the earth.']

#1. Join everything
print("#1 Join everything")

new_string = ""
full_sentence = new_string.join(dead_men_tell_tales)
print(full_sentence)

#2. Remove spaces
print("\n#2 Remove spaces")
full_sentence_no_spaces = full_sentence.replace(" ","")
print(full_sentence_no_spaces)

#3. Occurrence probabilities for letters
print("\n#3 Occurrence probabilities for letters")
import collections

new_string = ""
full_sentence = new_string.join(dead_men_tell_tales)
letter_counts = collections.Counter(full_sentence)
probabilities = {}

for letter, count in letter_counts.items():
    probabilities[letter] = count / len(full_sentence)

print(probabilities)

#4. Tell me transition probabilities for every pair of letters
print("\n#4 Transition probabilities for every pair of letters")

#join the list of sentences into one string
sentence = ' '.join(dead_men_tell_tales)

#create the transition counts dictionary
transition_counts = {}
for i in range(len(sentence) - 1):
    current_letter = sentence[i]
    next_letter = sentence[i + 1]
    if current_letter in transition_counts:
        if next_letter in transition_counts[current_letter]:
            transition_counts[current_letter][next_letter] += 1
        else:
            transition_counts[current_letter][next_letter] = 1
    else:
        transition_counts[current_letter] = {next_letter: 1}

#calculate the transition probabilities
transition_probabilities = {}
for current_letter, next_letter_counts in transition_counts.items():
    transition_probabilities[current_letter] = {}
    total_count = sum(next_letter_counts.values())
    for next_letter, count in next_letter_counts.items():
        transition_probabilities[current_letter][next_letter] = count / total_count

print(transition_probabilities)

#5. Make a 26x26 graph of 4. in numpy
print("\n#5 Make a 26x26 graph of 4. in numpy")
import numpy as np

# create a 26x26 array filled with 0s
graph = np.zeros((26,26))

#map the transition probabilities to the graph
for current_letter, next_letter_probs in transition_probabilities.items():
    if current_letter.isalpha() and current_letter.islower():
        for next_letter, prob in next_letter_probs.items():
            if next_letter.isalpha() and next_letter.islower():
                graph[ord(current_letter) - ord('a'), ord(next_letter) - ord('a')] = prob
        
print(graph)

#6. plot graph of transition probabilities from letter to letter
print("\n#6 - See heatmap of transition probabilities. Also, this plot assumes"
      " that all letters are lowercase.")

#create a heatmap of the graph
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(graph, cmap='hot')

# set the ticks and labels
ax.set_xticks(np.arange(26))
ax.set_yticks(np.arange(26))
ax.set_xticklabels([chr(i) for i in range(ord('a'), ord('z')+1)])
ax.set_yticklabels([chr(i) for i in range(ord('a'), ord('z')+1)])

# set the axis labels
ax.set_xlabel('Next letter')
ax.set_ylabel('Current letter')

# create the colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")

plt.show()

#Unrelated:
#7. Flatten a nested list
print("\n#7 Flatten a nested list")

nested_list = [[1, 2, [3, 4]], [5, 6], [7, 8, 9]]
flattened_list = []

# Create a queue to hold the nested lists
queue = nested_list.copy()

# Iterate while the queue is not empty
while queue:
    current_element = queue.pop(0)
    if isinstance(current_element, list):
        # If the current element is a list, add its elements to the queue
        queue.extend(current_element)
    else:
        # If the current element is not a list, append it to the flattened list
        flattened_list.append(current_element)

print(flattened_list)

