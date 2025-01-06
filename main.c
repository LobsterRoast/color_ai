#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "ai.h"

int main() {
    float inputs[] = {1.0, 0.0, 0.0};
    AI_Client client;
    Create_AI_Client(&client, 3, 1, 10, 5);
    Forward_Propagate(&client, inputs, 3);
    Clear_Nodes(&client);
    Free_AI_Client(&client);
}