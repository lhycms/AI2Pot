import numpy as np

# 创建一个示例的布尔型索引数组，维度为(10,)
bool_index = np.array([True, False, True, True, False, True, True, False, True, True])

# 创建一个示例的二维数组，维度为(10, 20)
arr_2d = np.random.randint(0, 100, size=(10, 20))

# 使用布尔型索引进行筛选
result = arr_2d[bool_index]

print(result.shape)
