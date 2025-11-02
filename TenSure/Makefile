CXX = g++
CXXFLAGS = -std=c++17 -I include
LDFLAGS =
LIBS =
SRC_DIRS = ./src   # Add other directories with cpp files
SRCS = $(wildcard $(addsuffix /*.cpp,$(SRC_DIRS)))
OBJS = $(SRCS:.cpp=.o)

TACO_INC = -I ../tools/taco/include
TACO_LIB = -L ../tools/taco/build/lib -ltaco -Wl,-rpath,../tools/taco/build/lib

all:
	$(CXX) $(SRCS) $(CXXFLAGS) $(TACO_INC) $(TACO_LIB) -o $(FILE:.cpp=.out)