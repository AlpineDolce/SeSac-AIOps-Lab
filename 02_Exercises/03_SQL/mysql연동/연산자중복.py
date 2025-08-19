class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)

        # data = [[1, 2, 3], [4, 5, 6]]
        # cols = len(data[0])  # 3

        # data = []
        # cols = 0
        # print(cols)

        self.cols = len(data[0]) if data else 0  #data가 존재하면 data길이를  존재안하면 0

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.data])

    def __repr__(self):
        return f"Matrix({self.data})"

    def __getitem__(self, idx):
        return self.data[idx]

    def __add__(self, other):
        if self.size() != other.size():
            raise ValueError("행렬 크기가 다릅니다.")
        result = [
            [self.data[i][j] + other[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def __sub__(self, other):
        if self.size() != other.size():
            raise ValueError("행렬 크기가 다릅니다.")
        result = [
            [self.data[i][j] - other[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):  # 스칼라 곱
            result = [[elem * other for elem in row] for row in self.data]
            return Matrix(result)
        elif isinstance(other, Matrix):  # 행렬 곱
            if self.cols != other.rows:
                raise ValueError("행렬 곱이 불가능합니다.")
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    val = sum(self.data[i][k] * other[k][j] for k in range(self.cols))
                    row.append(val)
                result.append(row)
            return Matrix(result)
        else:
            raise TypeError("지원하지 않는 곱셈 타입입니다.")

    def size(self):
        return (self.rows, self.cols)

    def __len__(self):
        return self.rows



A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B:")
print(A + B)

print("\nA - B:")
print(A - B)

print("\nA * 3 (스칼라 곱):")
print(A * 3)

print("\nA * B (행렬 곱):")
print(A * B)

print("\nA[1][0] =>", A[1][0])  # 인덱싱

print("\nlen(A) =>", len(A))    # 행 수


#3항연산자
a = 10
b = 20

min_value = a if a < b else b
print(min_value)  # 출력: 10

x = -5
result = "양수" if x > 0 else "음수 또는 0"
print(result)  # 출력: 음수 또는 0