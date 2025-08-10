---
title: "JavaScript基础知识"
date: "2024-01-15"
tags: ["javascript", "programming", "frontend", "basics"]
category: "notes"
description: "JavaScript编程语言的基础概念和语法"
draft: false
---

# JavaScript基础知识

## 变量和数据类型

JavaScript是一种动态类型语言，变量可以存储不同类型的数据。

### 基本数据类型

```javascript
// 数字类型
let age = 25;
let price = 99.99;

// 字符串类型
let name = "张三";
let message = `你好，${name}！`;

// 布尔类型
let isActive = true;
let isCompleted = false;

// undefined和null
let undefinedVar;
let nullVar = null;
```

### 复合数据类型

```javascript
// 数组
let fruits = ["苹果", "香蕉", "橙子"];
let numbers = [1, 2, 3, 4, 5];

// 对象
let person = {
    name: "李四",
    age: 30,
    city: "北京"
};
```

## 函数

函数是JavaScript中的一等公民，可以用多种方式定义。

### 函数声明

```javascript
function greet(name) {
    return `你好，${name}！`;
}

console.log(greet("王五")); // 输出：你好，王五！
```

### 箭头函数

```javascript
const add = (a, b) => a + b;
const multiply = (x, y) => {
    return x * y;
};

console.log(add(3, 4)); // 输出：7
console.log(multiply(5, 6)); // 输出：30
```

## 控制流

### 条件语句

```javascript
let score = 85;

if (score >= 90) {
    console.log("优秀");
} else if (score >= 80) {
    console.log("良好");
} else if (score >= 60) {
    console.log("及格");
} else {
    console.log("不及格");
}
```

### 循环语句

```javascript
// for循环
for (let i = 0; i < 5; i++) {
    console.log(`第${i + 1}次循环`);
}

// forEach循环
let colors = ["红色", "绿色", "蓝色"];
colors.forEach((color, index) => {
    console.log(`${index}: ${color}`);
});
```

## 相关笔记

- [[python-basics]] - Python基础知识对比
- [[javascript-advanced]] - JavaScript高级特性

## 参考资源

- [MDN JavaScript文档](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript)
- [JavaScript.info](https://zh.javascript.info/)