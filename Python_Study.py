import matplotlib as mpl
import matplotlib.pyplot as plt
plt.xkcd()
ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640, 84666,
84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708, 108423, 101407, 112542, 122870, 120000]
plt.plot(ages_x, py_dev_y, marker='D',label='Python')
js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674, 68745, 68746, 74583, 79000,
78508, 79996, 80403, 83820, 88833, 91660, 87892, 96243, 90000, 99313, 91660, 102264, 100000, 100000, 91660, 99240, 108000, 105000, 104000]
# plt.plot(ages_x, js_dev_y, label='Java')
plt.bar(ages_x, js_dev_y, label='Java')
plt.legend()
plt.xlabel('age')
plt.ylabel('salary')
plt.title('salary by age')
plt.show()

import csv
with open('test.csv') as csv_file:
    csv_reader=csv.DictReader(csv_file)
    row=next(csv_reader)
    print(row['language'].split('，'))

from collections import Counter
c=Counter()
for row in csv_reader:
    c.update(row['language'].split('，'))
languages=[]
popularity=[]
for item in c.most_common(15)):
    language.append(item[0])
    popularity.append(item[1])
language.reverse()
popularity.reverse()
plt.bar(language,popularity)
plt.barh(language,popularity)
#you just need to switch axis label for vertical bar chart

import pandas as pd
df=pd.read_csv('test.csv')
ids=df['responderID']
responses=df['languages']

#pie chart
slices=[40,60]
labels=['forty','sixty']
explode=[0,0,0,0.1,0] #to offset the slice
plt.pie(slices,labels=labels, explode=explode,autopct='%1.1f%%')

import matplotlib.pyplot as plt
ages=[1,2,3,3,5,3,4,6,8,2,3,5,7]
plt.hist(ages,bins=5,edgecolor='black')
#you clarify you want 5 bins
plt.show()

def multi_sum(*args):
    s=0
    for num in args:
        s+=num
    return s
print(multi_sum(3,4,5,7,8,9))

def print_values(**kwargs):
    for key, value in kwargs.items():
        print("The value of {} is {}".format(key, value))
print_values(my_name="Sammy", your_name="Casey")

def do_something(name, age, gender='M', *args, **kwds):
    print('Name:%s Age:%d,Gender:%s'%(name, age, gender))
    print(args)
    print(kwds)
do_something('John', 50, 'M', 175, 75, math=99, english=90)

y=5
print('y is negative' if y<0 else'y is not negative')
x=5*y if y<0 else 10*y
print(x)

with open(r"D:\CSDN\Column\temp\mpmap.py", 'r') as fp:
    contents = fp.readlines()
# no need to close after you open it

a = [1, 2, 3, 4, 5]
result = [i*i for i in a]
print(result)

a = [{'name':'B', 'age':50}, {'name':'A', 'age':30}, {'name':'C', 'age':40}]
print(sorted(a, key=lambda x:x['name']))

a = [1,2,3]
for item in map(lambda x:x**3, a):
    print(item, end=',')

a = [1,2,3]
a_iter = iter(a)
for i in a_iter:
    print(i**4, end=', ')
#iterator is only for one-time use to save memory

def get_square(n):
    for i in range(n):
        yield(pow(i,2))
a = get_square(5)
print(list(a))
for i in a:
    print(i, end=', ')
# yield is also only for one-time use to save memory

def decorator_func(original_func):
    def wrapper_func(*args,**kwds):
        print('Genius')
        return original_func()
    return wrapper_func

@decorator_func # equals to do_something=decorator_func(do_something)
def do_something():
    print('original_func')

do_something()
# wrapper decorator used to put on func into another func

import csv
with open('test.csv','r') as myFile:
    lines=csv.reader(myFile)
    for line in lines:
        print (line)

import pandas as pd
pd.read_csv('test.csv')
pd.read_excel('test.xlsx')
