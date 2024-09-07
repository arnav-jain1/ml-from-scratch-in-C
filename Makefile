# Compiler
CC = clang

# Compiler flags
CFLAGS = -O3 -Wall -g

# Libraries
LIBS = -lm

# Source files
SRCS = main.c

# Output executable
TARGET = main

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# Clean up
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
