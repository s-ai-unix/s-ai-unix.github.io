---
title: "Python数据分析完整指南：NumPy、Pandas与可视化实战"
date: 2019-07-04T11:43:34+08:00
draft: false
tags:
  - Python
  - 数据分析
description: "全面掌握Python数据分析核心工具，从NumPy数组操作到Pandas数据处理，再到Matplotlib和Seaborn可视化，构建完整的数据分析技能体系。"
cover:
  image: "images/covers/1551288049-bebda4e38f71.jpg"
  alt: "Python数据分析"
  caption: "Python数据分析：NumPy、Pandas与可视化"
---

Python数据分析生态系统包含了多个强大的库，它们各自承担不同的职责。本文将整合NumPy、Pandas、Matplotlib和Seaborn的核心功能，提供一份完整的数据分析实战指南。

## 环境配置与最佳实践

### Jupyter Notebook优化设置

在Mac上使用Jupyter时，通过以下配置可以显著提升绘图质量。在`~/.ipython/profile_default/ipython_kernel_config.py`中添加：

```python
c.IPKernelApp.matplotlib = 'inline'
c.InlineBackend.figure_format = 'retina'
```

### 绘图样式设置

使用更美观的绘图样式：

```python
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# 查看所有可用样式
print(plt.style.available)
```

推荐使用的样式包括：`'seaborn'`、`'ggplot'`、`'bmh'`等。

### 核心库导入

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
```

## NumPy：高效数值计算基础

NumPy是Python数据分析的基石，提供了高性能的多维数组和数值计算功能。

### 数组创建与基本操作

```python
# 加载示例数据
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# 选择特定列
data = np.array(df.iloc[:100, [0, 1, -1]])
print(f"DataFrame shape: {df.shape}")  # (150, 5)
print(f"Data shape: {data.shape}")     # (100, 3)
```

### 数据提取技巧

**从ndarray提取数据和标签：**

```python
# 假设最后一列是标签，前面是特征
X = dataset[:, 0:-1]  # 或 dataset[:, 0:8]
y = dataset[:, 8]
```

**从DataFrame提取数据和标签：**

```python
X, y = data.iloc[:, :-1], data.iloc[:, -1]
```

**分离特征和标签：**

```python
# 获取前两列特征和最后一列标签
X, y = data[:, :-1], data[:, -1]
print(f"X shape: {X.shape}")  # (100, 2)
print(f"y shape: {y.shape}")  # (100,)
```

### 数组切片的重要区别

理解切片的维度差异至关重要：

```python
an_array = np.array([[11, 12, 13, 14],
                     [21, 22, 23, 24],
                     [31, 32, 33, 34]])

row_rank1 = an_array[1, :]    # shape: (4,) - 一维数组
row_rank2 = an_array[1:2, :]  # shape: (1, 4) - 二维数组
```

### 布尔索引与条件过滤

```python
an_ndarray = np.array([[11, 12], [21, 22], [31, 31]])

# 方法1：显式创建布尔数组
bigger_than_fifteen = (an_ndarray > 15)
filtered = an_ndarray[bigger_than_fifteen]

# 方法2：直接使用条件表达式
filtered = an_ndarray[(an_ndarray > 15)]
```

### 标签转换

使用列表推导式进行标签转换：

```python
# 二分类转换
y = np.array([1 if i == 1 else -1 for i in y])
```

### 统计运算

```python
# 基本统计
an_ndarray.mean()              # 所有元素均值
an_ndarray.mean(axis=1)        # 每行均值
an_ndarray.mean(axis=0)        # 每列均值
an_ndarray.sum()               # 所有元素和

# 中位数计算
np.median(an_ndarray, axis=1)  # 每行中位数
np.median(an_ndarray, axis=0)  # 每列中位数

# 唯一值
np.unique(an_ndarray)
```

### 集合操作

```python
s1 = np.array(['desk', 'chair', 'bulb'])
s2 = np.array(['lamp', 'bulb', 'chair'])

# 交集
np.intersect1d(s1, s2)

# 去重并集
np.union1d(s1, s2)

# 差集（在s1但不在s2）
np.setdiff1d(s1, s2)

# 判断元素是否在s2中
np.in1d(s1, s2)
```

### 文件存取

```python
# 二进制格式（推荐）
x = np.array([23.23, 24.24])
np.save('an_array', x)
loaded = np.load('an_array.npy')

# 文本格式
np.savetxt('array.txt', X=x, delimiter=',')
loaded = np.loadtxt('array.txt', delimiter=',')
```

### 数组拼接

```python
K = np.random.randint(low=2, high=50, size=(2, 2))
M = np.random.randint(low=2, high=50, size=(2, 2))

# 垂直拼接（行）
np.vstack((K, M))

# 水平拼接（列）
np.hstack((K, M))

# 使用concatenate（更灵活）
np.concatenate([K, M], axis=0)      # 等价于vstack
np.concatenate([K, M.T], axis=1)    # 水平拼接转置后的M
```

## Pandas：数据处理的瑞士军刀

Pandas建立在NumPy之上，提供了更高级的数据结构和数据分析工具。

### 数据加载与预处理

**读取数据时处理空值：**

```python
# 读取时指定空值表示
auto = pd.read_csv('Data/Auto.csv', na_values='?').dropna()

# 选择特定列
credit = pd.read_csv('Data/Credit.csv', usecols=list(range(1, 12)))
advertising = pd.read_csv('Data/Advertising.csv', usecols=[1, 2, 3, 4])

# 设置索引列
data = pd.read_csv('../data/Smarket.csv', index_col=0)
```

**使用sklearn数据集：**

```python
from sklearn.datasets import load_boston
boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
```

### 数据清洗

**替换异常值：**

```python
auto['horsepower'] = auto['horsepower'].replace('?', np.nan)
```

**删除空值：**

```python
auto = auto.dropna()
```

**类型转换：**

```python
auto['horsepower'] = auto['horsepower'].astype('int')
```

**原地替换：**

```python
df_x.replace(to_replace={0: 'No', 1: 'Yes', 'True': 'Yes', 'False': 'No'},
            inplace=True)
```

### 数据选择与过滤

**使用iloc选择：**

```python
# 选择所有行，从第二列开始的所有列
data = data.iloc[:, 1:]

# 选择特定行列
corr_matrix = boston.corr()
corr_matrix.iloc[1:, 0].sort_values()
```

**条件筛选：**

```python
# 筛选特定条件的行
elite_colleges = college[college['Elite'] == 'Yes']

# 根据索引删除行
info = auto.drop(auto.index[10:85]).describe().T
```

### 数据排序

```python
# 单列排序
auto = auto.sort_values(by=['horsepower'], ascending=True, axis=0)

# 多列排序
boston.sort_values(by=['CRIM', 'TAX', 'PTRATIO'], ascending=False).head().index
```

### 数据转换

**条件赋值：**

```python
# 使用np.where
college['Elite'] = np.where(college['Top10perc'] > 50, 'Yes', 'No')

# 使用map
credit['Student2'] = credit.Student.map({'No': 0, 'Yes': 1})
```

**分桶操作：**

```python
college['Enroll'] = pd.cut(college['Enroll'], bins=3,
                           labels=['Low', 'Medium', 'High'])
```

**创建新列：**

```python
info['range'] = info['max'] - info['min']
info = info[['mean', 'range', 'std']]
```

### 数据框操作

**设置索引：**

```python
college = college.set_index(['Unnamed: 0'], append=True, verify_integrity=True)
college.rename_axis([None, 'Name'], inplace=True)
```

**拼接数据框：**

```python
# 按列拼接
features = pd.concat([constant, features], axis=1)
```

**构造数据框：**

```python
X = np.random.normal(size=100)
y = np.random.permutation(X)
data = pd.DataFrame({'X': X, 'y': y})
```

### 数据探索

```python
# 查看唯一值数量
auto.nunique()

# 查看数据类型和内存使用
auto.info()

# 查看某列的唯一值
auto['horsepower'].unique()

# 值计数
college['Elite'].value_counts()
```

## Matplotlib：基础可视化

Matplotlib是Python最基础的可视化库，提供了完整的绘图API。

### 基础图形

**散点图与曲线图：**

```python
plt.figure(figsize=(14, 8))
plt.scatter(auto['horsepower'], auto['mpg'])
plt.plot(auto['horsepower'], pred_1, color='orange', label='Degree 1')
plt.plot(auto['horsepower'], pred_2, color='green', label='Degree 2')
plt.plot(auto['horsepower'], pred_5, color='black', label='Degree 5')
plt.show()
```

**直方图：**

```python
fig = plt.figure()

plt.subplot(2, 2, 1)
college['Enroll'].value_counts().plot.bar(title='Enroll')

plt.subplot(2, 2, 2)
college['PhD'].value_counts().plot.bar(title='PhD')

plt.subplot(2, 2, 3)
college['Terminal'].value_counts().plot.bar(title='Terminal')

# 调整子图间距
fig.subplots_adjust(hspace=1)
```

**添加参考线：**

```python
error_1 = auto['mpg'] - pred_1
plt.figure(figsize=(12, 6))
sns.scatterplot(x=auto['mpg'], y=error_1)
plt.axhline(y=0, linestyle='dashed', color='black', linewidth=0.5)
```

### 图表装饰

**设置标签和标题：**

```python
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
```

**添加图例：**

```python
plt.figure(figsize=(14, 6))
sns.scatterplot(X, Y)
plt.xlabel('X')
plt.ylabel('Y')

plt.plot(data['X'], lin_model.predict(data['X'].to_frame()),
         color='orange', label='Predicted Line')
plt.plot(tmp_x, tmp_y, color='green', label='True Line')
plt.legend()
plt.show()
```

### 坐标轴控制

**设置坐标范围：**

```python
plt.xlim(-10, 310)
plt.ylim(ymin=0)
```

**等比例坐标轴：**

```python
fig, ax = plt.subplots()
sns.scatterplot(simple_coeff, multi_coeff)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()])
]

ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, color='orange')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
```

### 高级绘图

**等高线图和3D图：**

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 6))
fig.suptitle('RSS - Regression coefficients', fontsize=20)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# 等高线图
CS = ax1.contour(xx, yy, Z, cmap=plt.cm.Set1,
                 levels=[2.15, 2.2, 2.3, 2.5, 3])
ax1.scatter(regr.intercept_, regr.coef_[0], c='r', label=min_RSS)
ax1.clabel(CS, inline=True, fontsize=10, fmt='%1.1f')

# 3D曲面图
ax2.plot_surface(xx, yy, Z, rstride=3, cstride=3, alpha=0.3)
ax2.contour(xx, yy, Z, zdir='z', offset=Z.min(), cmap=plt.cm.Set1,
            alpha=0.4, levels=[2.15, 2.2, 2.3, 2.5, 3])
ax2.scatter3D(regr.intercept_, regr.coef_[0], min_rss, c='r', label=min_RSS)
ax2.set_zlabel('RSS')

# 通用设置
for ax in fig.axes:
    ax.set_xlabel(r'$\beta_0$', fontsize=17)
    ax.set_ylabel(r'$\beta_1$', fontsize=17)
    ax.legend()
```

**3D散点图：**

```python
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(advertising.Radio, advertising.TV, advertising.Sales, c='r')
ax.set_xlabel('Radio')
ax.set_ylabel('TV')
ax.set_zlabel('Sales')
```

**使用注解：**

```python
df.plot('Years', 'Hits', kind='scatter', color='orange', figsize=(7, 6))
plt.xlim(0, 25)
plt.ylim(ymin=-5)
plt.xticks([1, 4.5, 24])
plt.yticks([1, 117.5, 238])
plt.vlines(4.5, ymin=-5, ymax=250)
plt.hlines(117.5, xmin=4.5, xmax=25)
plt.annotate('R1', xy=(2, 117.5), fontsize='xx-large')
plt.annotate('R2', xy=(11, 60), fontsize='xx-large')
plt.annotate('R3', xy=(11, 170), fontsize='xx-large')
```

## Seaborn：高级统计可视化

Seaborn建立在Matplotlib之上，提供了更美观的默认样式和高级统计图表。

### 常用图表类型

**成对关系图：**

```python
# 选择特定列
sns.pairplot(college.iloc[:, 1:11])

# 使用所有列
sns.pairplot(auto)
```

**箱线图：**

```python
sns.boxplot(x=college['Private'], y=college['Outstate'])
```

**回归图：**

```python
# 简单线性回归
sns.regplot(data['LSTAT'], data['MEDV'])

# 自定义参数
sns.regplot(advertising.TV, advertising.Sales,
            order=1, ci=None,
            scatter_kws={'color': 'r', 's': 9})
plt.xlim(-10, 310)
plt.ylim(ymin=0)
```

**热力图（相关系数矩阵）：**

```python
sns.heatmap(auto.iloc[:, :-1].corr())
```

### 按类别分组可视化

**使用hue参数：**

```python
plt.figure(figsize=(12, 8))

tmp = pd.DataFrame({
    'Lag1': X_train['Lag1'],
    'Lag2': X_train['Lag2'],
    'Direction': y_train
})

sns.scatterplot(y='Lag2', x='Lag1', hue='Direction', data=tmp)
```

**复杂多图布局：**

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 4)
ax1 = plt.subplot(gs[0, :-2])
ax2 = plt.subplot(gs[0, -2])
ax3 = plt.subplot(gs[0, -1])

# 采样策略
df_no = df[df.default2 == 0].sample(frac=0.15)
df_yes = df[df.default2 == 1]
df_ = df_no.append(df_yes)

# 散点图
ax1.scatter(df_[df_.default == 'Yes'].balance,
            df_[df_.default == 'Yes'].income,
            s=40, c='orange', marker='+', linewidths=1)
ax1.scatter(df_[df_.default == 'No'].balance,
            df_[df_.default == 'No'].income,
            s=40, marker='o', linewidths=1,
            edgecolors='lightblue', facecolors='white', alpha=.6)

# 箱线图
c_palette = {'No': 'lightblue', 'Yes': 'orange'}
sns.boxplot('default', 'balance', data=df, orient='v',
            ax=ax2, palette=c_palette)
sns.boxplot('default', 'income', data=df, orient='v',
            ax=ax3, palette=c_palette)

gs.tight_layout(plt.gcf())
```

## 实战案例：完整数据分析流程

让我们通过一个完整的案例整合所学知识：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. 数据加载
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 2. 数据探索
print(f"数据形状: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n基本统计:\n{df.describe()}")

# 3. 数据可视化
plt.style.use('seaborn')

# 成对关系
sns.pairplot(df, hue='species', height=2.5)
plt.suptitle('Iris数据集特征关系', y=1.02)

# 相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap='coolwarm')
plt.title('特征相关性矩阵')

# 4. 特征工程
df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']

# 5. 数据转换
species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['species_code'] = df['species'].map(species_map)

# 6. 高级分析
# 按物种分组统计
group_stats = df.groupby('species').agg({
    'sepal length (cm)': ['mean', 'std'],
    'petal length (cm)': ['mean', 'std']
})

print("\n按物种分组的统计信息:")
print(group_stats)

# 7. 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

df.boxplot(column='sepal length (cm)', by='species', ax=axes[0, 0])
df.boxplot(column='petal length (cm)', by='species', ax=axes[0, 1])

for species, color in zip(['setosa', 'versicolor', 'virginica'], ['red', 'green', 'blue']):
    subset = df[df['species'] == species]
    axes[1, 0].scatter(subset['sepal length (cm)'],
                       subset['petal length (cm)'],
                       label=species, alpha=0.6)

axes[1, 0].set_xlabel('Sepal Length')
axes[1, 0].set_ylabel('Petal Length')
axes[1, 0].legend()
axes[1, 0].set_title('萼片vs花瓣长度')

plt.tight_layout()
plt.show()
```

## 最佳实践与技巧

### 性能优化

1. **使用向量化操作**：避免在DataFrame上使用循环
2. **合理使用数据类型**：`category`类型用于低基数分类变量
3. **批量操作**：使用`apply()`而不是逐行迭代

### 代码可读性

1. **方法链**：`df.dropna().sort_values().groupby()`
2. **有意义的变量名**：`df_filtered`而不是`df2`
3. **注释和文档**：解释复杂的数据转换逻辑

### 调试技巧

```python
# 检查缺失值
df.isnull().sum()

# 检查数据类型
df.dtypes

# 查看内存使用
df.memory_usage()

# 采样查看大数据集
df.sample(100)
```

## 常见问题与解决方案

### axis参数的理解

- `axis=0`：沿列向下操作（行操作）
- `axis=1`：沿行向右操作（列操作）

```python
df.mean(axis=0)  # 每列的均值
df.mean(axis=1)  # 每行的均值
```

### copy与view的区别

```python
# 创建副本（推荐）
df_copy = df.copy()

# 可能创建视图
df_view = df.iloc[:, :2]

# 修改视图可能影响原数据
df_view.iloc[0, 0] = 999  # 可能影响df
```

### 链式赋值警告

```python
# 避免
df[df['A'] > 0]['B'] = 1  # 警告

# 推荐
df.loc[df['A'] > 0, 'B'] = 1
```

## 总结

Python数据分析工具链的掌握需要理解各库的职责分工：

- **NumPy**：底层数值计算，高性能数组操作
- **Pandas**：数据清洗、转换、分析的核心工具
- **Matplotlib**：灵活的基础绘图库
- **Seaborn**：美观的统计可视化

通过本文的整合学习，你应该能够：
1. 使用NumPy进行高效的数值计算
2. 使用Pandas进行数据清洗和转换
3. 使用Matplotlib和Seaborn创建各种可视化图表
4. 理解完整的数据分析工作流程

持续练习和实际项目应用是掌握这些工具的关键。建议使用真实的 Kaggle 数据集或工作中的实际问题来巩固所学知识。

## 参考资源

- [NumPy官方文档](https://numpy.org/doc/)
- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [Matplotlib官方文档](https://matplotlib.org/stable/contents.html)
- [Seaborn官方文档](https://seaborn.pydata.org/tutorial.html)
- [ISLR Python](https://github.com/JWarmenhoven/ISLR-python)
- [An Introduction to Statistical Learning with Python](https://github.com/hardikkamboj/An-Introduction-to-Statistical-Learning)
