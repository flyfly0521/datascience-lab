# 编译器
CXX = g++

# 编译选项
CXXFLAGS = -O3 -mavx2 -mfma -std=c++17 -Wall -Wextra

# 源文件
SVM_SRC = svm.cpp
KFOLD_SRC = k-fold.cpp

# 可执行文件
SVM_TARGET = svm
KFOLD_TARGET = k-fold

# 默认编译所有
all: $(SVM_TARGET) $(KFOLD_TARGET)

# ===== svm =====
$(SVM_TARGET): $(SVM_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# ===== k-fold =====
$(KFOLD_TARGET): $(KFOLD_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# 清理
clean:
	rm -f $(SVM_TARGET) $(KFOLD_TARGET) *.o

.PHONY: all clean