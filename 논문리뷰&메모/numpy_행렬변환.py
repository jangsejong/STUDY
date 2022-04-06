import numpy as np

A = np.array([[1, 2, 3],[4, 5, 6]])

#3 X 2 행렬 B 만들기

B = A.reshape((3, 2))

#소괄호가 2개 있음에 주의해야 하며 성분의 배치 순서는 행렬 A의 첫 행의 처음부터 시작하여 오른쪽으로 가면서 차례로 행렬 B의 성분을 채운다.

# 행렬 A 자체를 3 X 2 행렬로 바꿀때

A.resize((3, 2))
print("A=", A)

# 전치(transpode)는 ndarray.T 를 사용.

C = ndarray.T

C = B.T
print("B=", B, "\nC=", C)

# 넘파이에서 어레이의 인덱스는 0에서 시작한다.

# 콜론(:)은 전체를 의미한다.

# 행렬 A에 C를 추가하고 싶으면 np.append 함수를 사용한다. axis=0 이면 행으로 추가,
# axis= 1이면 열로 추가한다.

A = np.array([[1,2,3], [4,5,6]])
C = np.array([[1,3,5], [2,4,6]])

D = np.append(A, C, axis=0)
print("D=", D)

E = np.append(A, C, axis=1)
print("E=", E)

# np.concatenate 도 같은 기능을 한다. axis=0 이면 행으로 추가, axis=1 이면 열로 추가한다. 행렬 A 와 C를 소괄호로 묶는데 주의해야 한다.

F1 = np.concatenate((A,C),axis=0)
print("F1=", F1)

F2 = np.concatenate((A,C),axis=1)
print("F2=", F2)

# np.vstack 은 두 행렬을 세로로 쌓고, np.hstack 은 가로로 쌓는다.

G = np.hstack((A,C))
print("G=", G)

# 행렬을 성분으로 하는 큰 행렬을 만드려면 np.tile 을 사용한다. 가령 행렬 A를 성분으로 하는 2행3열의 큰 행렬은 다음과 같이 만든다.

H = np.tile(A, [2,3])
print(H.shape)

# 차원 또는 축(axis)를 추가시키려면 np.expand_dim 함수를 사용한다. axis=0 이면 행축 추가, axis=1 이면 열축 추가, axis=-1 이면 마지막 축 추가를 의미한다.
# A의 모양 (2,3) 에서 행축 (axis=0)이 추가됐으므로 I의 모양은 (1,2,3) 이 된다.

A = np.array([[1,2,3], [4,5,6]])
I = np.expand_dims(A, axis=0)
print(A.shape, I.shape)
print("I=", I)

# 열축 (axis=1)이 추가되면 J의 모양은 (2,1,3) 이 된다.
J = np.expand_dims(A, axis=1)
print(A.shape, J.shape)
print("J=", J)

# 깊이축 또는 마지막축 (axis=2 또는 axis= -1)이 추가되면 K의 모양은 (2,3,1) 이 된다.
K = np.expand_dims(A, axis=-1)
print(A.shape, K.shape)
print("K=", K)