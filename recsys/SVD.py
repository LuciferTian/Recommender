# -*- coding:utf-8 -*-
# 读入数据
import numpy as np
import random
file_name1 = 'user_item_rating.txt'
rating_matrix = np.loadtxt(file_name1,dtype=bytes).astype(float)
# 用户数
user_num = rating_matrix.shape[0]
# 项目数
item_num = rating_matrix.shape[1]
# 初始化用户隐向量矩阵和项目隐向量矩阵，隐特征数为2
feature_num = 2
random.seed(1)
user_matrix = np.random.random_sample((user_num,feature_num))
item_matrix = np.random.random_sample((item_num,feature_num))

# 计算已有评分项的均值
def get_miu(data_matrix):
    non_zero_num = 0
    non_zero_sum = 0
    # 对用户
    for i in range(data_matrix.shape[0]):
        # 对物品
        for j in range(data_matrix.shape[1]):
            if data_matrix[i][j] != 0:
                non_zero_num += 1
                non_zero_sum += data_matrix[i][j]
    # 返回已有评分项个数及全局均值
    return non_zero_sum/non_zero_num,non_zero_num
miu,non_zero_num = get_miu(rating_matrix)

# 采用随机梯度下降训练两个隐向量矩阵
# 返回用户隐向量，项目隐向量，偏置向量
def SGD_bias(data_matrix,user,item,alpha,lam,iter_num,miu):
    # 偏置
    # 初始化用户偏置
    b_u = [1] * rating_matrix.shape[0]
    # 初始化项目偏置
    b_i = [1] * rating_matrix.shape[1]
    for j in range(iter_num):
        for u in range(data_matrix.shape[0]):
            for i in range(data_matrix.shape[1]):
                if data_matrix[u][i] != 0:
                    b_ui = b_u[u] + b_i[i] + miu
                    e_ui = data_matrix[u][i] - b_ui - sum(user[u,:] * item[i,:])
                    user[u,:] += alpha * (e_ui * item[i,:] - lam * user[u,:])
                    item[i,:] += alpha * (e_ui * user[u,:] - lam * item[i,:])
                    b_u[u] += alpha * (e_ui - lam * b_u[u])
                    b_i[i] += alpha * (e_ui - lam * b_i[i])
    return user,item,b_u,b_i
print('全局均值为%f,已有评分数为%d' % (miu,non_zero_num))
print('矩阵的稀疏率为：%.3f' % (non_zero_num/2500))
user_bias,item_bias,b_u,b_i = SGD_bias(rating_matrix,user_matrix,item_matrix,0.001,0.1,1000,miu)

# 预测
matrix_predict_bias = np.dot(user_bias,item_bias.transpose())  # user_bias和item_bias的转置相乘，结果是一个矩阵
for u in range(matrix_predict_bias.shape[0]):
    for i in range(matrix_predict_bias.shape[1]):
        matrix_predict_bias[u][i] += (miu + b_u[u] + b_i[i])
np.savetxt('matrix_predict.txt', matrix_predict_bias, fmt='%.2f')

# 评估：只对训练数据评估
def MSE(data_matrix,predict_matrix_bias,non_zero_num):
    filter_matrix_entry = data_matrix > 0
    # 对应位置相乘
    predict_matrix_filtered = np.multiply(predict_matrix_bias, filter_matrix_entry)
    err = (predict_matrix_filtered - data_matrix) * (predict_matrix_filtered - data_matrix)
    mse = err.sum()/non_zero_num
    return mse
mse = MSE(rating_matrix,matrix_predict_bias,non_zero_num)
print('the mse is:%.4f' % mse)