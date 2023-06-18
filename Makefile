CXX=clang++
INCLUDES=
CXXFLAGS= -Wall -std=c++20 $(INCLUDES)

# hel: main.cc hello.cc
#	$(CXX) $(CXXFLAGS) $^

build: clean vector

# test: main.o
#	$(CXX) main.o -o test

# main.o: main.cc
#    $(CXX) $(CXXFLAGS) main.cc -c

vector: vector.o
	$(CXX) vector.o -o vector

vector.o: vector.cc
	$(CXX) $(CXXFLAGS) vector.cc -c


# hello.o: hello.cc
#	$(CXX) $(CXXFLAGS) hello.cc -c


.PHONY: clean build tests

clean:
	rm -fr vector vector.o test main.o