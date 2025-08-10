---
title: "Python基础知识"
date: "2024-01-20"
tags: ["python", "programming", "backend", "basics"]
category: "notes"
description: "Python编程语言的基础概念和语法"
draft: false
---

# Python基础知识

## 变量和数据类型

Python是一种动态类型语言，具有简洁的语法和强大的功能。

### 基本数据类型

```python
# 数字类型
age = 25
price = 99.99
complex_num = 3 + 4j

# 字符串类型
name = "张三"
message = f"你好，{name}！"

# 布尔类型
is_active = True
is_completed = False

# None类型
empty_var = None
```

### 集合数据类型

```python
# 列表（可变）
fruits = ["苹果", "香蕉", "橙子"]
numbers = [1, 2, 3, 4, 5]

# 元组（不可变）
coordinates = (10, 20)
rgb = (255, 128, 0)

# 字典
person = {
    "name": "李四",
    "age": 30,
    "city": "北京"
}

# 集合
unique_numbers = {1, 2, 3, 4, 5}
```

## 函数

Python中的函数定义简洁明了。

### 基本函数

```python
def greet(name):
    return f"你好，{name}！"

def add(a, b=0):  # 默认参数
    return a + b

print(greet("王五"))  # 输出：你好，王五！
print(add(3, 4))     # 输出：7
print(add(5))        # 输出：5
```

### Lambda函数

```python
# Lambda表达式
square = lambda x: x ** 2
multiply = lambda x, y: x * y

print(square(5))      # 输出：25
print(multiply(3, 4)) # 输出：12

# 与内置函数结合使用
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # 输出：[1, 4, 9, 16, 25]
```

## 控制流

### 条件语句

```python
score = 85

if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")
```

### 循环语句

```python
# for循环
for i in range(5):
    print(f"第{i + 1}次循环")

# 遍历列表
colors = ["红色", "绿色", "蓝色"]
for index, color in enumerate(colors):
    print(f"{index}: {color}")

# while循环
count = 0
while count < 3:
    print(f"计数：{count}")
    count += 1
```

## 列表推导式

Python的列表推导式是一个强大的特性：

```python
# 基本列表推导式
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件的列表推导式
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# 字典推导式
word_lengths = {word: len(word) for word in ["python", "java", "javascript"]}
print(word_lengths)  # {'python': 6, 'java': 4, 'javascript': 10}
```

## 类和对象

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"我是{self.name}，今年{self.age}岁"
    
    def have_birthday(self):
        self.age += 1
        return f"{self.name}现在{self.age}岁了"

# 创建对象
person = Person("小明", 25)
print(person.introduce())     # 我是小明，今年25岁
print(person.have_birthday()) # 小明现在26岁了
```

## 相关笔记

- [[javascript-basics]] - JavaScript基础知识对比
- [[python-advanced]] - Python高级特性

## 参考资源

- [Python官方文档](https://docs.python.org/zh-cn/3/)
- [Python教程](https://www.runoob.com/python3/python3-tutorial.html)