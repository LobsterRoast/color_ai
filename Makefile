SRC := main.c ai.c ai_utils.c
C := gcc
CFLAGS :=
OBJS := $(SRC:.c=.o)
TARGET := color_ai

%.o: %.c %.h
	$(C) $(CFLAGS) -c $< -o $@
$(TARGET): $(OBJS)
	$(C) $(OBJS) -o $(TARGET)
