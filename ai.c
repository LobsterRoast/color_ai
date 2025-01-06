#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "ai.h"
#include "ai_utils.h"

void Create_Input_Layer(AI_Client* client, uint16_t input_layer_depth) {
    client->input_layer = (Layer*)malloc(sizeof(Layer));
    client->input_layer->nodes = (Node*)malloc(sizeof(Node)*input_layer_depth);
    client->input_layer->layer_type = LAYER_TYPE_INPUT;
    client->input_layer->depth = input_layer_depth;
    client->input_layer->layer_id = -1;
    // Ensure input nodes are activated by default and that their layer field is set
    for (int i = 0; i < input_layer_depth; i++) {
        client->input_layer->nodes[i].activated = true;
        client->input_layer->nodes[i].layer = client->input_layer;
    }
}
void Create_Remaining_Layers(AI_Client* client, uint16_t output_layer_depth, uint16_t hidden_layer_depth, uint16_t hidden_layer_count) {
    Layer* current_layer = client->input_layer;
    // Recursively create the hidden layers and output layer
    for (int i = 0; i < hidden_layer_count+1; i++) {
        current_layer->next = (Layer*)malloc(sizeof(Layer));
        current_layer->next->last = current_layer;
        if (i < hidden_layer_count) {
            current_layer->next->layer_id = i;
            current_layer->next->depth = hidden_layer_depth;
            current_layer->next->nodes = (Node*)malloc(sizeof(Node)*hidden_layer_depth);
            current_layer->next->layer_type = LAYER_TYPE_HIDDEN;
            for (int j = 0; j < hidden_layer_depth; j++) {
                current_layer->next->nodes[j].layer = current_layer;
            }
        }
        else {
            current_layer->next->layer_id = -2;
            current_layer->next->depth = output_layer_depth;
            current_layer->next->nodes = (Node*)malloc(sizeof(Node)*output_layer_depth);
            current_layer->next->layer_type = LAYER_TYPE_OUTPUT;
            for (int j = 0; j < output_layer_depth; j++) {
                current_layer->next->nodes[j].layer = current_layer;
            }
            client->output_layer = current_layer->next;
        }
        current_layer = current_layer->next;
    }
}

void Create_Input_Connections(AI_Client* client) {
    for (int i = 0; i < client->input_layer->depth; i++) {
        client->input_layer->nodes[i].outgoing_connection_count = client->input_layer->next->depth;
        client->input_layer->nodes[i].outgoing_connections = (Connection**)malloc(sizeof(Connection*) * client->input_layer->nodes[i].outgoing_connection_count);
        for (int j = 0; j < client->input_layer->next->depth; j++) {
            client->input_layer->nodes[i].outgoing_connections[j] = (Connection*)malloc(sizeof(Connection));
            client->input_layer->nodes[i].outgoing_connections[j]->sending_node = &client->input_layer->next->nodes[j];
            client->input_layer->nodes[i].outgoing_connections[j]->receiving_node = &client->input_layer->nodes[i];
        }
    }
}

void Create_Hidden_Connections(AI_Client* client) {
    // Start at the first hidden layer and iterate through the hidden layers
    Layer* current_layer = client->input_layer->next;
    while (current_layer->layer_type != LAYER_TYPE_OUTPUT) {
        // Iterate through nodes in the current layer
        for (int i = 0; i < current_layer->depth; i++) {
            current_layer->nodes[i].incoming_connection_count = current_layer->last->depth;
            current_layer->nodes[i].outgoing_connection_count = current_layer->next->depth;
            // Ensure that each node in the current layer has as many incoming connections as there are nodes in the previous layer
            current_layer->nodes[i].incoming_connections = (Connection**)malloc(sizeof(Connection*) * current_layer->nodes[i].incoming_connection_count);
            // Ensure that each node in the current layer has as many outgoing connections as there are nodes in the next layer
            current_layer->nodes[i].outgoing_connections = (Connection**)malloc(sizeof(Connection*) * current_layer->nodes[i].outgoing_connection_count);
            // Iterate through nodes of the previous layer
            for (int j = 0; j < current_layer->last->depth; j++) {
                current_layer->nodes[i].incoming_connections[j] = current_layer->last->nodes[j].outgoing_connections[i];
            }
            // Iterate through nodes of the next layer
            for (int j = 0; j < current_layer->next->depth; j++) {
                current_layer->nodes[i].outgoing_connections[j] = (Connection*)malloc(sizeof(Connection));
                current_layer->nodes[i].outgoing_connections[j]->sending_node = &current_layer->nodes[i];
                current_layer->nodes[i].outgoing_connections[j]->receiving_node = &current_layer->next->nodes[j];
            }
        }
        current_layer = current_layer->next;
    }
}

void Create_Output_Connections(AI_Client* client) {
    for (int i = 0; i < client->output_layer->depth; i++) {
        client->output_layer->nodes[i].incoming_connection_count = client->output_layer->last->depth;
        client->output_layer->nodes[i].incoming_connections = (Connection**)malloc(sizeof(Connection*) * client->output_layer->nodes[i].incoming_connection_count);
        for (int j = 0; j < client->output_layer->last->depth; j++) {
            client->output_layer->nodes[i].incoming_connections[j] = client->output_layer->last->nodes[j].outgoing_connections[i];
        }
    }
}


void Create_Connections(AI_Client* client) {
    // THIS NEEDS TO BE OVERHAULED
    // Currently, the outgoing connections of one node and the incoming connections of the next node are completely independent
    // There needs to be checks in place to ensure that no duplicate connections are made
    Create_Input_Connections(client);
    Create_Hidden_Connections(client);
    Create_Output_Connections(client);
}
void Deparallelize(AI_Client* client) {
    // Initialize weights and biases to random values to break symmetry
    // Starting at input_layer->next because input_layer does not have weights
    Layer* current_layer = client->input_layer->next;
    int seed = time(NULL);
    srand(time(NULL));
    while (current_layer != NULL) {
        for (int i = 0; i < current_layer->depth; i++) {
            // This is an SRNG chain where the result of each call of gen_rand is used as the seed of the next one
            // This is not cryptographically secure, but it doesn't need to be. As long as it breaks symmetry, it has
            // done what it needs to do.
            // The weights cannot be parallel (i.e. all initialized to 0), or else the neural network will not be able to learn properly.
            //current_layer->nodes[i].weight = gen_rand(&seed);
        }
        current_layer = current_layer->next;
    }
}
int Create_AI_Client(
    AI_Client* client, 
    uint16_t input_layer_depth, 
    uint16_t output_layer_depth, 
    uint16_t hidden_layer_depth, 
    uint16_t hidden_layer_count
    ) {
    Create_Input_Layer(client, input_layer_depth);
    Create_Remaining_Layers(client, output_layer_depth, hidden_layer_depth, hidden_layer_count);
    Create_Connections(client);
    Deparallelize(client);
    return 0;
}

int Free_AI_Client(AI_Client* client) {
    Layer* current_layer = client->input_layer;
    Layer* next_layer;
    while(current_layer != NULL) {
        for (int i = 0; i < current_layer->depth; i++) {
            for (int j = 0; j < current_layer->nodes[i].outgoing_connection_count; j++) {
                free(current_layer->nodes[i].outgoing_connections[j]);
            }
            free(current_layer->nodes[i].incoming_connections);
            free(current_layer->nodes[i].outgoing_connections);
        }
        next_layer = current_layer->next;
        free(current_layer->nodes);
        free(current_layer);
        current_layer = next_layer;
    }
    return 0;
}

void Forward_Pass_On_Layer(Layer* current_layer) {
    Layer* last_layer = current_layer->last;
    // i represents an iterator through the nodes of the layer we're CURRENTLY OPERATING ON.
    for (int i = 0; i < current_layer->depth; i++) {
        // j represents an iterator through the nodes of the layer we're RECEIVING INPUTS FROM.
        for (int j = 0; j < last_layer->depth; j++) {
            current_layer->nodes[i].input += last_layer->nodes[j].output;
        }
        //current_layer->nodes[i].input = Activation_Function()
    }
}

int Forward_Propagate(AI_Client* client, float* input_vector, uint16_t input_vector_depth) {
    if (input_vector_depth < client->input_layer->depth) {
        printf("Input vector is too small! Expects %i elements.", client->input_layer->depth);
        return 1;
    }
    for (int i = 0; i < client->input_layer->depth; i++) {
        client->input_layer->nodes[i].input = input_vector[i];
    }
    Layer* current_layer = client->input_layer->next;
    while (current_layer != NULL) {
        Forward_Pass_On_Layer(current_layer);
    }
    return 0;
}

int Back_Propagate(AI_Client* client, float* output_vector, uint16_t output_vector_depth) {
    if (output_vector_depth < client->output_layer->depth) {
        printf("Output vector is too small! Expects %i elements.", client->output_layer->depth);
        return 1;
    }
}
int Clear_Nodes(AI_Client* client) {
    Layer* current_layer = client->input_layer;
    while(current_layer->layer_type != LAYER_TYPE_OUTPUT) {
        for (int i = 0; i < current_layer->depth; i++) {
            current_layer->nodes[i].input = 0;
            current_layer->nodes[i].activated = false;
        }
        current_layer = current_layer->next;
    }
    for (int i = 0; i < current_layer->depth; i++) {
        current_layer->nodes[i].input = 0;
        current_layer->nodes[i].activated = false;
    }
}

float Activation_Function(Node* node) {

}