CXX = g++

CFLAGS = --std=c++11 -Wall -Wextra -pedantic
LDFLAGS = -lopengl32 -lglu32 -lfreeglut -lglew32

.PHONY: all main bench clean

all: main bench

main: lab6_cpu.cpp
	$(CXX) $(CFLAGS) -o lab6_cpu lab6_cpu.cpp $(LDFLAGS)

bench: lab6_cpu.cpp
	$(CXX) $(CFLAGS) -o bench/lab6_cpu -DTIME lab6_cpu.cpp $(LDFLAGS)

clean:
	rm -f bench/lab6_cpu lab6_cpu