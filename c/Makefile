CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
LDFLAGS = -lm

SRCS = mst.c union_find.c
OBJS = $(SRCS:.c=.o)
TARGET = mst

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mst.o: mst.c union_find.h
	$(CC) $(CFLAGS) -c $<

union_find.o: union_find.c union_find.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)

# Debug build with sanitizers
debug: CFLAGS += -g -fsanitize=address -fsanitize=undefined
debug: clean all
