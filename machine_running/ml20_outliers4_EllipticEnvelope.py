# import numpy as np

# aaa = np.array([[1,2,-1000,4,5,6,7,8,90,100,500,12,13],[100,200,3,400,500,600,7,800,900,190,1001,1002,99]])
# #(2, 13) -> (13, 2)

# aaa = np.transpose(aaa)


# from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.1) # 10%구간 오염도로 잡는다.

# outliers.fit(aaa)
# resulits = outliers.predict(aaa)
# print(resulits)


import  numpy as np
aaa = np.array([[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]])
# (2, 13) -> (13, 2)
aaa = np.transpose(aaa)   # (13, 2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.2)    # contamination을 조정해서 아웃라이어의 범위를 지정해줄수 있음

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [ 1  1  1  1  1  1  1  1  1 -1 -1  1  1]  -> 아웃라이어의 위치를 -1로 표기해줌
b = list(results)
print(b.count(-1))
index_for_outlier = np.where(results == -1)
print('outier indexex are', index_for_outlier)
outlier_value = aaa[index_for_outlier]
print('outlier_value :', outlier_value)
'''
[ 1  1 -1  1  1  1  1  1  1 -1 -1  1  1]
3
outier indexex are (array([ 2,  9, 10], dtype=int64),)
outlier_value : [[ -20    3]
 [ 100  190]
 [ 500 1001]]
'''