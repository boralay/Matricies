# CXX=clang++
CXX=g++ 
NVCC=nvcc
INCLUDES=
# CXXFLAGS= -Wall -std=c++20 $(INCLUDES)
CXXFLAGS= -std=c++20 $(INCLUDES) -g

# hel: main.cc hello.cc
#	$(CXX) $(CXXFLAGS) $^

build: clean vector

# test: main.o
#	$(CXX) main.o -o test

# main.o: main.cc
#    $(CXX) $(CXXFLAGS) main.cc -c

vector: vector.o kernels.o
	$(NVCC) kernels.o vector.o -o vector

vector.o: vector.cc
	$(CXX) $(CXXFLAGS) vector.cc -c


kernels.o: kernels.cu
	$(NVCC) $(CXXFLAGS) kernels.cu -c

# hello.o: hello.cc
#	$(CXX) $(CXXFLAGS) hello.cc -c


.PHONY: clean build tests

clean:
	rm -fr vector vector.o test main.o kernels.o
