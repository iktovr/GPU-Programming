CXX = g++

CFLAGS = --std=c++11 -Wall -Wextra -pedantic
LDFLAGS = -lopengl32 -lglu32 -lfreeglut -lglew32

.PHONY: clean

all: lab6_cpu lab6_cpu_bench

lab6_cpu: lab6_cpu.cpp
	$(CXX) $(CFLAGS) -o lab6_cpu lab6_cpu.cpp $(LDFLAGS)

lab6_cpu_bench: lab6_cpu.cpp
	$(CXX) $(CFLAGS) -o lab6_cpu_bench -DTIME lab6_cpu.cpp $(LDFLAGS)

clean:
	rm -f lab6_cpu_bench lab6_cpu