x=(1,2,3,6) #tuple
y=[4,52,34,4] #list
y.append(3.3) # y is a list, when using method, you dont have to reassign value to y

i=0
while i!=len(y):
    print(y[i])
    i+=1

print(y*3)
print(4 in y)
print(y[0:2])
print(y[-1])
print(y[1:])
Quotes= "industrious and intelligent"
m = Quotes.split(" ")[0]

"hello"+str(2)

#dictionaries
z={"John":12,"Hui":234}
print(z["John"])
z["Shan"]=222
print(z)
for name in z:
    print(z[name])
for values in z.values():
    print(values) #remember to put square brackets
for items in z.items():
    print(items) #returns sets of tuples
for name, values in z.items():
    print(name, values)

x=(1,3,5)
a,b,c=x
print(a)

temp_dic={"fig1":33,"fig2":11}
temp_word= "{}/{}={}"
print(temp_word.format(temp_dic["fig1"],temp_dic["fig2"],
temp_dic["fig1"]/temp_dic["fig2"]))

import csv
%precision 2 # two digits after decimal
with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))
len(mpg)
mpg[0].keys()
sum(float(d['cty']) for d in mpg) / len(mpg)
cylinders = set(d['cyl'] for d in mpg)
# find unique values
CtyMpgByCyl = []

for c in cylinders: # iterate over all the cylinder levels
    summpg = 0
    cyltypecount = 0
    for d in mpg: # iterate over all dictionaries
        if d['cyl'] == c: # if the cylinder level type matches,
            summpg += float(d['cty']) # add the cty mpg
            cyltypecount += 1 # increment the count
    CtyMpgByCyl.append((c, summpg / cyltypecount)) # append the tuple ('cylinder', 'avg mpg')

CtyMpgByCyl.sort(key=lambda x: x[0]) # sort by keys
CtyMpgByCyl

a=[(1,9),(2,8),(3,7)]
a.sort(reverse=True, key=lambda x: x[1])
print(a)

# Objects and maps
class Person:
    department = 'School of Information' #a class variable
    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location
person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.
format(person.name, person.location, person.department))

class Rectangle:
   def __init__(self, length, breadth, unit_cost=0):
       self.length = length
       self.breadth = breadth
       self.unit_cost = unit_cost
   def get_area(self):
       return self.length * self.breadth
   def calculate_cost(self):
       area = self.get_area()
       return area * self.unit_cost
# breadth = 120 units, length = 160 units, 1 sq unit cost = Rs 2000
r = Rectangle(160, 120, 2000)
print("Area of Rectangle: %s sq units" % (r.get_area()))

store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
print(cheapest) # will only print map new_location
for i in cheapest:
    print(i)

my_function = lambda a, b, c : a + b
my_list = [number for number in range(0,1000) if number % 2 == 0]
my_list = []
for number in range(0, 1000):
    if number % 2 == 0:
        my_list.append(number)
my_list

import numpy as np
x = np.array([[1,2,3],[4,5,6]])
# 2*3 array remember double square brackets
print(x.shape)
x=x.reshape(3,2) # need to reassign
x.resize(2,3) # does not need to reassign
print(x)
np.array([1, 2, 3] * 3) #1,2,3,1,2,3,1,2,3
np.repeat([1, 2, 3], 3) #1,1,1,2,2,2,3,3,3
np.vstack([p, 2*p])
x.dot(y) # dot product  1*4 + 2*5 + 3*6

condition = True
if condition:
    x=1
else:
    x=0

x = 1 if condition else 0 #the same effect

# you can add underscore to the number to add readability
num1=10_000_000_000
num2=100_000_000
num3=num1+num2
print(f'{num3:,}')
# add more readability

f=open('test.txt','r')
file_content=f.read()
f.close()
# it is very easy to forget to close the 'f'
# you can use context manager like this

with open('test.txt','r') as f:
    file_content= f.read()

names=[1,2,3,4]
for key, value in enumerate(names,start=1):
    print(key, value)

fnames=['zexiang','hui']
lnames=['huang','shan']
for key,value in enumerate(fnames):
    lname=lnames[key]
    print(lname,value)
    print(f'{lname} is {value}')
#f'' contains expressions in between
# another way to do this is unpacking
for fname, lname in zip(fnames,lnames):
    print(fname,lname)
# remember zip will stop at the shortest list is exhausted
for value in zip(fnames,lnames):
    print(value) # this will just print tuples of the values
# Ctrl + / can comment and uncomment a bunch of codes mutiple cylinders

a,b=(1,2)
print(a,b) # you can just name the second variable as underscore to ignore it
# in the case you dont really need it.
a,b,*c=(1,2,3,4,5,6)
print(a,b,c) # * means assign the rest of the values to c as a my_list
# class can allow you to dynamically add attributes and values to it.

z.T # transpose
z.sum().max().mean().min().std()
z.argmax()/z.argmin() #returns the max min value position of the array
r[r > 30]

r_copy = r.copy()
# you have to copy the slice of the array before you want to make modifications
# cause it will just cause modifications on both sides
import pandas as pd
s=pd.Series(['Tiger', 'Bear', 'Moose'])
print(s[0])
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
print(s.index)
iloc/loc
total=series.sum()
for label, value in s.iteritems():
    s.set_value(label, value+2)
# series can be appended
s1=s2.append(s3)
df = pd.DataFrame([s1, s2, s3], index=['Store 1', 'Store 1', 'Store 2'])
df.T # dataframe can be transposed
df.loc['Store 1', 'Cost'] # two dimentional selection
if you do not quote iloc or loc, it will index the column the label
df.T.loc['Cost']=df['Cost']
df.loc[:,['Name', 'Cost']]
remember do not chain square brackets
df.loc['Store 1']['Cost'] not like this

copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df
# also remember to copy before drop a row
del copy_df['Name']
copy_df
df['Location'] = None
df
df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.columns
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
df['Gold'] > 0
df=df[df['Gold']>0]
df['Gold'].count()
only_gold = only_gold.dropna()
len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)]) # or
df[(df['Gold.1'] > 0) & (df['Gold'] == 0)] # and
df['country'] = df.index # set column name of the original index,
df = df.set_index('Gold') # set new index
df.head()
df = df.reset_index() # reset index as 0,1,2,3,
df['SUMLEV'].unique() find the unique value of a column
columns_to_keep = ['STNAME',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
df = df.set_index(['STNAME', 'CTYNAME']) # double index
df.loc['Michigan', 'Washtenaw County'] # after you need to quote two elements in index_col
df = df.set_index('time')
df = df.sort_index()
df = df.reset_index()
df = df.set_index(['time', 'user'])
df = df.fillna(method='ffill') # fill down the NAN

df['Gold'].argmax()
(df['Gold']-df['Gold.1']).abs().argmax()
elegible = df[(df['Gold']>=1) & (df['Gold.1']>=1)]
    ratios = (elegible['Gold'] - elegible['Gold.1']).abs()/elegible['Gold.2']
    return ratios.argmax()
points = np.zeros(len(df))
points += df['Gold.2'] * 3
points += df['Silver.2'] * 2
points += df['Bronze.2']
return pd.Series(points, index=df.index)
return counties.groupby("STNAME").count().COUNTY.argmax()
top_threes = counties.groupby("STNAME")["CENSUS2010POP"].nlargest(3)
states = top_threes.groupby(level=0).sum()
return list(states.nlargest(3).index)

Week 3, join
pd.merge(staff_df, student_df,
how='outer', left_index=True, right_index=True)
# how=inner/left/right
pd.merge(staff_df, student_df,
how='left', left_on='Name', right_on='Name')
# when index is not name use left_on,right_on
# Idiomatic Pandas: Making Code Pandorable
(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME','CTYNAME'])
    .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))

def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})
df.apply(lambda x: np.max(x[rows]), axis=1)

for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])
    print('Counties in state ' + state + ' have an average population of ' + str(avg))
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg))
df.groupby('STNAME').agg({'CENSUS2010POP': np.average})
# agg means aggregrate
groupby(level=0) # first index

df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)

ts3.index = pd.to_datetime(ts3.index)
pd.to_datetime('4.7.12', dayfirst=True)
pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')
pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')

tab indent, shift+tab dis indent

import numpy as np
x = np.random.binomial(20, .5, 10000)
print((x>=15).mean())

a=[1,2,3,4,5,6,7]
b=list(number for number in a if number>2)
print(b)

names=[1,2,3,4]
for key, value in enumerate(names,start=1):
    print(key, value)

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
