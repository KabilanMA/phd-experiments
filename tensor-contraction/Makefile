# Makefile for tensor-contraction

CXX = g++
CXXFLAGS = -std=c++11 -I include
LDFLAGS =
LIBS =
SRC_DIRS = ./src   # Add other directories with cpp files
SRCS = $(wildcard $(addsuffix /*.cpp,$(SRC_DIRS)))
OBJS = $(SRCS:.cpp=.o)

TACO_INC = -I ../tools/taco/include
TACO_LIB = -L ../tools/taco/build/lib -ltaco -Wl,-rpath,../tools/taco/build/lib

# Default target: build all objects
all: $(OBJS)

# Build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS) ./src/*.out

# Taco build: build specific files with TACO library linked
# Example usage: make taco FILE=experiments/exp1.cpp
taco:
	$(CXX) $(SRCS) $(CXXFLAGS) $(TACO_INC) $(TACO_LIB) -o $(FILE:.cpp=.out)
