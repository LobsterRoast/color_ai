#include <stdlib.h>
#include <stdbool.h>

#define e 2.718281828459045

typedef struct {
    double x;
    double y;
} Vector2;

float gen_rand(int* seed);

char Is_Activated(bool activated);

float sigmoid(float input);