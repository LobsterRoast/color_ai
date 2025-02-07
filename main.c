#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "ai.h"

#ifdef _WIN32

#include <windows.h>
#define sleep() Sleep()

#endif

#ifdef __unix__

#include <unistd.h>

#endif

int main() {
    float inputs[] = {1.0};
    float outputs[] = {1.0};
    AI_Client* client;
    Create_AI_Client(client, 1, 1, 10, 5, 1);
    while(true) {
        Forward_Propagate(client, inputs, 3);
        printf("Output: %f\n", client->output_layer->nodes[0].output);
        Back_Propagate(client, outputs, 1);
    }
    Clear_Nodes(client);
    Free_AI_Client(client);
}