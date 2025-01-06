#include "ai_utils.h"

float gen_rand(int* seed) {
    int rand_num = rand();
    *seed = rand_num;
    srand(*seed);
    return (float)((rand_num % 100)-50)/50;
}


char Is_Activated(bool activated) {
    return activated ? 'T' : 'F';
}

float sigmoid(float input) {
    return 1 / (1 + e/-input);
}