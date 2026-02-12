---
title: "数据结构实现系列：线性表、链表与栈的完整实现"
date: 2014-08-19T12:23:17+08:00
draft: false
tags:
  - 数据结构
  - C语言
  - 算法
  - 综述
description: "详细介绍线性表、链表、栈等基础数据结构的C语言实现，包括顺序存储和链式存储两种方式，以及指针实现和引用实现的对比。"
cover:
  image: "images/covers/1509228627129-669005e74585.jpg"
  alt: "数据结构与算法"
  caption: "数据结构：程序设计的基石"
---

数据结构是计算机科学的基础，掌握各种数据结构的实现原理对于编写高效程序至关重要。本文将详细介绍线性表、链表、栈等基础数据结构的实现方法。

## 一、线性表的顺序存储实现

线性表是最基本的数据结构之一，其顺序存储方式使用连续的内存空间来存储数据元素。

### 1.1 使用指针实现

```c
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 20
typedef int Status;
typedef int ElemType;

/* 定义顺序表结构 */
typedef struct {
    ElemType data[MAXSIZE];
    int length;
} SqList;

/* 初始化顺序表 */
Status InitList(SqList *L) {
    L->length = 0;
    return OK;
}

/* 判断顺序表是否为空 */
Status ListEmpty(SqList L) {
    if(L.length == 0)
        return TRUE;
    else
        return FALSE;
}

/* 清空顺序表 */
Status ClearList(SqList *L) {
    L->length = 0;
    return OK;
}

/* 获取顺序表长度 */
int ListLength(SqList L) {
    return L.length;
}

/* 获取第i个元素 */
Status GetElem(SqList L, int i, ElemType *e) {
    if(L.length == 0 || i < 1 || i > L.length)
        return ERROR;
    *e = L.data[i-1];
    return OK;
}

/* 查找元素e的位置 */
int LocateElem(SqList L, ElemType e) {
    int i;
    if (L.length == 0)
        return 0;
    for(i = 0; i < L.length; i++) {
        if (L.data[i] == e)
            break;
    }
    if(i >= L.length)
        return 0;
    return i + 1;
}

/* 在第i个位置插入元素e */
Status ListInsert(SqList *L, int i, ElemType e) {
    int k;
    if (L->length == MAXSIZE)
        return ERROR;
    if (i < 1 || i > L->length + 1)
        return ERROR;

    if (i <= L->length) {
        for(k = L->length - 1; k >= i - 1; k--)
            L->data[k+1] = L->data[k];
    }

    L->data[i-1] = e;
    L->length++;
    return OK;
}

/* 删除第i个元素 */
Status ListDelete(SqList *L, int i, ElemType *e) {
    int k;
    if (L->length == 0)
        return ERROR;
    if (i < 1 || i > L->length)
        return ERROR;

    *e = L->data[i-1];

    if (i < L->length) {
        for(k = i; k < L->length; k++)
            L->data[k-1] = L->data[k];
    }

    L->length--;
    return OK;
}

/* 遍历顺序表 */
Status ListTraverse(SqList L) {
    int i;
    for(i = 0; i < L.length; i++)
        printf("%d ", L.data[i]);
    printf("\n");
    return OK;
}
```

### 1.2 使用C++引用实现

```cpp
#include <stdio.h>
#include <stdlib.h>
#define MaxSize 50

typedef char ElemType;
typedef struct {
    ElemType data[MaxSize];
    int length;
} SqList;

/* 创建顺序表 */
void CreateList(SqList *&L, ElemType a[], int n) {
    int i;
    L = (SqList *)malloc(sizeof(SqList));
    for (i = 0; i < n; i++)
        L->data[i] = a[i];
    L->length = n;
}

/* 初始化顺序表 */
void InitList(SqList *&L) {
    L = (SqList *)malloc(sizeof(SqList));
    L->length = 0;
}

/* 销毁顺序表 */
void DestroyList(SqList *&L) {
    free(L);
}

/* 判断是否为空 */
int ListEmpty(SqList *L) {
    return(L->length == 0);
}

/* 获取长度 */
int ListLength(SqList *L) {
    return(L->length);
}

/* 显示顺序表 */
void DispList(SqList *L) {
    int i;
    if (ListEmpty(L)) return;
    for (i = 0; i < L->length; i++)
        printf("%c ", L->data[i]);
    printf("\n");
}

/* 获取第i个元素 */
int GetElem(SqList *L, int i, ElemType &e) {
    if (i < 1 || i > L->length)
        return 0;
    e = L->data[i-1];
    return 1;
}

/* 查找元素 */
int LocateElem(SqList *L, ElemType e) {
    int i = 0;
    while (i < L->length && L->data[i] != e) i++;
    if (i >= L->length)
        return 0;
    else
        return i + 1;
}

/* 插入元素 */
int ListInsert(SqList *&L, int i, ElemType e) {
    int j;
    if (i < 1 || i > L->length + 1)
        return 0;
    i--;
    for (j = L->length; j > i; j--)
        L->data[j] = L->data[j-1];
    L->data[i] = e;
    L->length++;
    return 1;
}

/* 删除元素 */
int ListDelete(SqList *&L, int i, ElemType &e) {
    int j;
    if (i < 1 || i > L->length)
        return 0;
    i--;
    e = L->data[i];
    for (j = i; j < L->length - 1; j++)
        L->data[j] = L->data[j+1];
    L->length--;
    return 1;
}
```

**关键区别**：
- 指针版本使用 `*L` 访问结构体
- 引用版本使用 `&L` 声明参数，可以直接使用 `L->` 访问，代码更简洁

## 二、链表的实现

链表采用链式存储结构，通过指针将各个节点连接起来。

### 2.1 使用指针实现单向链表

```c
#include "stdio.h"
#include "stdlib.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

typedef int Status;
typedef int ElemType;

/* 定义节点结构 */
typedef struct Node {
    ElemType data;
    struct Node *next;
} Node;
typedef struct Node *LinkList;

/* 初始化链表 */
Status InitList(LinkList *L) {
    *L = (LinkList)malloc(sizeof(Node));
    if(!(*L))
        return ERROR;
    (*L)->next = NULL;
    return OK;
}

/* 判断链表是否为空 */
Status ListEmpty(LinkList L) {
    if(L->next)
        return FALSE;
    else
        return TRUE;
}

/* 清空链表 */
Status ClearList(LinkList *L) {
    LinkList p, q;
    p = (*L)->next;
    while(p) {
        q = p->next;
        free(p);
        p = q;
    }
    (*L)->next = NULL;
    return OK;
}

/* 获取链表长度 */
int ListLength(LinkList L) {
    int i = 0;
    LinkList p = L->next;
    while(p) {
        i++;
        p = p->next;
    }
    return i;
}

/* 获取第i个元素 */
Status GetElem(LinkList L, int i, ElemType *e) {
    int j;
    LinkList p = L->next;
    j = 1;
    while (p && j < i) {
        p = p->next;
        ++j;
    }
    if (!p || j > i)
        return ERROR;
    *e = p->data;
    return OK;
}

/* 查找元素位置 */
int LocateElem(LinkList L, ElemType e) {
    int i = 0;
    LinkList p = L->next;
    while(p) {
        i++;
        if(p->data == e)
            return i;
        p = p->next;
    }
    return 0;
}

/* 插入元素 */
Status ListInsert(LinkList *L, int i, ElemType e) {
    int j;
    LinkList p, s;
    p = *L;
    j = 1;
    while (p && j < i) {
        p = p->next;
        ++j;
    }
    if (!p || j > i)
        return ERROR;
    s = (LinkList)malloc(sizeof(Node));
    s->data = e;
    s->next = p->next;
    p->next = s;
    return OK;
}

/* 删除元素 */
Status ListDelete(LinkList *L, int i, ElemType *e) {
    int j;
    LinkList p, q;
    p = *L;
    j = 1;
    while (p->next && j < i) {
        p = p->next;
        ++j;
    }
    if (!(p->next) || j > i)
        return ERROR;
    q = p->next;
    p->next = q->next;
    *e = q->data;
    free(q);
    return OK;
}

/* 头插法创建链表 */
void CreateListHead(LinkList *L, int n) {
    LinkList p;
    int i;
    srand(time(0));
    *L = (LinkList)malloc(sizeof(Node));
    (*L)->next = NULL;
    for (i = 0; i < n; i++) {
        p = (LinkList)malloc(sizeof(Node));
        p->data = rand() % 100 + 1;
        p->next = (*L)->next;
        (*L)->next = p;
    }
}

/* 尾插法创建链表 */
void CreateListTail(LinkList *L, int n) {
    LinkList p, r;
    int i;
    srand(time(0));
    *L = (LinkList)malloc(sizeof(Node));
    r = *L;
    for (i = 0; i < n; i++) {
        p = (Node *)malloc(sizeof(Node));
        p->data = rand() % 100 + 1;
        r->next = p;
        r = p;
    }
    r->next = NULL;
}
```

### 2.2 使用C++引用实现链表

```cpp
#include <stdio.h>
#include <stdlib.h>

typedef char ElemType;
typedef struct LNode {
    ElemType data;
    struct LNode *next;
} LinkList;

/* 头插法创建链表 */
void CreateListF(LinkList *&L, ElemType a[], int n) {
    LinkList *s;
    int i;
    L = (LinkList *)malloc(sizeof(LinkList));
    L->next = NULL;
    for (i = 0; i < n; i++) {
        s = (LinkList *)malloc(sizeof(LinkList));
        s->data = a[i];
        s->next = L->next;
        L->next = s;
    }
}

/* 尾插法创建链表 */
void CreateListR(LinkList *&L, ElemType a[], int n) {
    LinkList *s, *r;
    int i;
    L = (LinkList *)malloc(sizeof(LinkList));
    L->next = NULL;
    r = L;
    for (i = 0; i < n; i++) {
        s = (LinkList *)malloc(sizeof(LinkList));
        s->data = a[i];
        r->next = s;
        r = s;
    }
    r->next = NULL;
}
```

## 三、栈的实现

栈是一种后进先出(LIFO)的数据结构，包括顺序栈和链栈两种实现方式。

### 3.1 顺序栈实现

```c
#include "stdio.h"
#include "stdlib.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0
#define MAXSIZE 1000

typedef int Status;
typedef int SElemType;

/* 顺序栈结构 */
typedef struct {
    SElemType data[MAXSIZE];
    int top;  /* 栈顶指针 */
} SqStack;

/* 构造空栈 */
Status InitStack(SqStack *S) {
    S->top = -1;
    return OK;
}

/* 清空栈 */
Status ClearStack(SqStack *S) {
    S->top = -1;
    return OK;
}

/* 判断栈是否为空 */
Status StackEmpty(SqStack S) {
    if (S.top == -1)
        return TRUE;
    else
        return FALSE;
}

/* 返回栈长度 */
int StackLength(SqStack S) {
    return S.top + 1;
}

/* 获取栈顶元素 */
Status GetTop(SqStack S, SElemType *e) {
    if (S.top == -1)
        return ERROR;
    else
        *e = S.data[S.top];
    return OK;
}

/* 入栈 */
Status Push(SqStack *S, SElemType e) {
    if(S->top == MAXSIZE - 1)  /* 栈满 */
        return ERROR;
    S->top++;
    S->data[S->top] = e;
    return OK;
}

/* 出栈 */
Status Pop(SqStack *S, SElemType *e) {
    if(S->top == -1)
        return ERROR;
    *e = S->data[S->top];
    S->top--;
    return OK;
}

/* 遍历栈 */
Status StackTraverse(SqStack S) {
    int i = 0;
    while(i <= S.top) {
        printf("%d ", S.data[i++]);
    }
    printf("\n");
    return OK;
}
```

### 3.2 链栈实现

```c
#include "stdio.h"
#include "stdlib.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

typedef int Status;
typedef int SElemType;

/* 链栈节点 */
typedef struct StackNode {
    SElemType data;
    struct StackNode *next;
} StackNode, *LinkStackPtr;

/* 链栈结构 */
typedef struct {
    LinkStackPtr top;
    int count;
} LinkStack;

/* 初始化链栈 */
Status InitStack(LinkStack *S) {
    S->top = (LinkStackPtr)malloc(sizeof(StackNode));
    if(!S->top)
        return ERROR;
    S->top = NULL;
    S->count = 0;
    return OK;
}

/* 清空链栈 */
Status ClearStack(LinkStack *S) {
    LinkStackPtr p, q;
    p = S->top;
    while(p) {
        q = p;
        p = p->next;
        free(q);
    }
    S->count = 0;
    return OK;
}

/* 判断是否为空 */
Status StackEmpty(LinkStack S) {
    if (S.count == 0)
        return TRUE;
    else
        return FALSE;
}

/* 获取长度 */
int StackLength(LinkStack S) {
    return S.count;
}

/* 获取栈顶元素 */
Status GetTop(LinkStack S, SElemType *e) {
    if (S.top == NULL)
        return ERROR;
    else
        *e = S.top->data;
    return OK;
}

/* 入栈 */
Status Push(LinkStack *S, SElemType e) {
    LinkStackPtr s = (LinkStackPtr)malloc(sizeof(StackNode));
    s->data = e;
    s->next = S->top;
    S->top = s;
    S->count++;
    return OK;
}

/* 出栈 */
Status Pop(LinkStack *S, SElemType *e) {
    LinkStackPtr p;
    if(StackEmpty(*S))
        return ERROR;
    *e = S->top->data;
    p = S->top;
    S->top = S->top->next;
    free(p);
    S->count--;
    return OK;
}
```

## 四、数据结构选择建议

### 4.1 顺序存储 vs 链式存储

**顺序存储的优点**：
- 存储密度大，空间利用率高
- 随机访问能力强，时间复杂度O(1)
- 不需要额外的指针空间

**顺序存储的缺点**：
- 插入和删除需要移动大量元素
- 需要预先分配连续内存空间
- 容易造成内存浪费

**链式存储的优点**：
- 插入和删除操作方便，不需要移动元素
- 内存空间动态分配，利用率高
- 不需要预先知道数据规模

**链式存储的缺点**：
- 需要额外的指针空间
- 只能顺序访问，不能随机访问
- 访问特定位置元素需要遍历

### 4.2 使用场景选择

1. **线性表**：适合元素数量相对稳定、频繁进行查找操作的场景
2. **链表**：适合频繁进行插入删除操作、数据规模不固定的场景
3. **栈**：适合需要后进先出特性的场景，如函数调用、表达式求值等

## 五、实际应用示例

### 5.1 从无序链表中删除重复元素

```c
#include <iostream>
#include <cstring>
using namespace std;

typedef struct node {
    int data;
    node *next;
} node;

bool myhash[100];

/* 初始化链表 */
node* init(int a[], int n) {
    node *head, *p;
    for(int i = 0; i < n; ++i) {
        node *nd = new node();
        nd->data = a[i];
        if(i == 0) {
            head = p = nd;
            continue;
        }
        p->next = nd;
        p = nd;
    }
    return head;
}

/* 使用哈希表删除重复元素 */
void removedulicate(node *head) {
    if(head == NULL) return;
    node *p = head, *q = head->next;
    myhash[head->data] = true;
    while(q) {
        if(myhash[q->data]) {
            node *t = q;
            p->next = q->next;
            q = p->next;
            delete t;
        }
        else {
            myhash[q->data] = true;
            p = q;
            q = q->next;
        }
    }
}

/* 不使用缓冲区删除重复元素 */
void removedulicate1(node *head) {
    if(head == NULL) return;
    node *p, *q, *c = head;
    while(c) {
        p = c;
        q = c->next;
        int d = c->data;
        while(q) {
            if(q->data == d) {
                node *t = q;
                p->next = q->next;
                q = p->next;
                delete t;
            }
            else {
                p = q;
                q = q->next;
            }
        }
        c = c->next;
    }
}

/* 打印链表 */
void print(node *head) {
    while(head) {
        cout << head->data << " ";
        head = head->next;
    }
    cout << endl;
}

int main() {
    int n = 10;
    int a[] = {3, 2, 1, 3, 5, 6, 2, 6, 3, 1};
    memset(myhash, false, sizeof(myhash));
    node *head = init(a, n);
    removedulicate1(head);
    print(head);
    return 0;
}
```

## 小结

本文详细介绍了线性表、链表和栈三种基础数据结构的实现方法，包括：

1. **顺序存储和链式存储**两种实现方式
2. **C语言指针**和**C++引用**两种编程风格
3. 完整的**增删改查**操作实现
4. 实际应用场景的**代码示例**

掌握这些基础数据结构的实现原理，是学习高级算法和设计复杂系统的基础。在实际开发中，应根据具体需求选择合适的数据结构，以获得最优的性能。

下一篇文章将介绍树形结构的实现，包括二叉树的遍历算法和递归与非递归的实现对比。
