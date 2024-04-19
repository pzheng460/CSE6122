# Makefile for MPI program

# Compiler
CXX = mpicxx

# Compiler flags
CXXFLAGS = -Wall -std=c++17 -O3

# Target executable names
TARGET = spmat
BONUS_TARGET = spmatBonus

# Source files
SOURCES = main.cpp
BONUS_SOURCES = bonus.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)
BONUS_OBJECTS = $(BONUS_SOURCES:.cpp=.o)

# Default target
all: $(TARGET) $(BONUS_TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

$(BONUS_TARGET): $(BONUS_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(BONUS_TARGET) $(BONUS_OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# Clean target
clean:
	rm -f $(OBJECTS) $(TARGET) $(BONUS_OBJECTS) $(BONUS_TARGET)

# Phony targets
.PHONY: all clean