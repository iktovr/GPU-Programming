CXX = mpic++

CFLAGS = --std=c++11 -Wall -Wextra -pedantic

all: lab7 lab7_bench

lab7: lab7.cpp
	$(CXX) $(CFLAGS) lab7.cpp -o lab7

lab7_bench: lab7.cpp
	$(CXX) $(CFLAGS) -DTIME lab7.cpp -o lab7_bench