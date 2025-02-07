#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "ai.h"
#include "ai_utils.h"

int seed = 0;

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
    for (int i = 0; i < hidden_layer_count + 1; i++) {
        current_layer->next = (Layer*)malloc(sizeof(Layer));
        current_layer->next->last = current_layer;
        if (i < hidden_layer_count) {
            // Working on a hidden layer
            current_layer->next->layer_id = i;
            current_layer->next->depth = hidden_layer_depth;
            current_layer->next->nodes = (Node*)malloc(sizeof(Node)*hidden_layer_depth);
            current_layer->next->layer_type = LAYER_TYPE_HIDDEN;
            for (int j = 0; j < hidden_layer_depth; j++) {
                current_layer->next->nodes[j].layer = current_layer->next;
            }
        }
        else {
            // Working on the output layer
            current_layer->next->layer_id = -2;
            current_layer->next->depth = output_layer_depth;
            current_layer->next->nodes = (Node*)malloc(sizeof(Node)*output_layer_depth);
            current_layer->next->layer_type = LAYER_TYPE_OUTPUT;
            for (int j = 0; j < output_layer_depth; j++) {
                current_layer->next->nodes[j].layer = current_layer->next;
            }
            client->output_layer = current_layer->next;
        }
        current_layer = current_layer->next;
    }
}

void Create_Input_Connections(AI_Client* client) {
    // Iterate through nodes in the current layer
    for (int i = 0; i < client->input_layer->depth; i++) {
        client->input_layer->nodes[i].outgoing_connection_count = client->input_layer->next->depth;
        client->input_layer->nodes[i].outgoing_connections = (Connection**)malloc(sizeof(Connection*) * client->input_layer->nodes[i].outgoing_connection_count);
        // Iterate through nodes of the next layer
        for (int j = 0; j < client->input_layer->next->depth; j++) {
            client->input_layer->nodes[i].outgoing_connections[j] = (Connection*)malloc(sizeof(Connection));
            client->input_layer->nodes[i].outgoing_connections[j]->sending_node = &client->input_layer->nodes[i];
            client->input_layer->nodes[i].outgoing_connections[j]->receiving_node = &client->input_layer->next->nodes[j];
            // Deparallelize the weights so the AI can learn properly
            client->input_layer->nodes[i].outgoing_connections[j]->weight = gen_rand(&seed);
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
                // Deparallelize the weights so the AI can learn properly
                current_layer->nodes[i].outgoing_connections[j]->weight = gen_rand(&seed);
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
    seed = time(NULL);
    srand(seed);
    Create_Input_Connections(client);
    Create_Hidden_Connections(client);
    Create_Output_Connections(client);
}

int Create_AI_Client(
    AI_Client* client, 
    uint16_t input_layer_depth, 
    uint16_t output_layer_depth, 
    uint16_t hidden_layer_depth, 
    uint16_t hidden_layer_count,
    uint16_t learning_rate
    ) {
    Create_Input_Layer(client, input_layer_depth);
    Create_Remaining_Layers(client, output_layer_depth, hidden_layer_depth, hidden_layer_count);
    Create_Connections(client);
    client->learning_rate = learning_rate;
    return 0;
}

int Free_AI_Client(AI_Client* client) {
    Layer* current_layer = client->input_layer;
    Layer* next_layer;
    // Recurse through each layer
    while(current_layer != NULL) {
        // Recurse through nodes in each layer freeing the connection pointers
        for (int i = 0; i < current_layer->depth; i++) {
            for (int j = 0; j < current_layer->nodes[i].outgoing_connection_count; j++) {
                // outgoing_connections is a Connection**. This frees the inner pointer
                // incoming_connections don't need to be freed because each incoming connection shares memory
                // with a corresponding outgoing connection
                free(current_layer->nodes[i].outgoing_connections[j]);
            }
            // This frees the outer encompassing pointers
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
    // i represents an iterator through the nodes of the layer we're CURRENTLY OPERATING ON.
    for (int i = 0; i < current_layer->depth; i++) {
        float sum = 0;
        // j represents an iterator through the nodes of the layer we're SENDING INPUTS TO.
        for (int j = 0; j < current_layer->nodes[i].outgoing_connection_count; j++) {
            Connection* connection = current_layer->nodes[i].outgoing_connections[j];
            Node* sending_node = connection->sending_node;
            Node* receiving_node = connection->receiving_node;
            sending_node->output = Activation_Function(sending_node);
            receiving_node->input += sending_node->output * connection->weight + receiving_node->bias;
        }
    }
}
void Zero_Out(AI_Client* client) {
    Layer* current_layer = client->input_layer;
    while (current_layer != NULL) {
        for (int i = 0; i < current_layer->depth; i++)
            current_layer->nodes[i].input = 0;
        current_layer = current_layer->next;
    }
}
int Forward_Propagate(AI_Client* client, float* input_vector, uint16_t input_vector_depth) {
    if (input_vector_depth < client->input_layer->depth) {
        printf("Input vector is too small! Expects %i elements.", client->input_layer->depth);
        return 1;
    }
    Zero_Out(client);
    for (int i = 0; i < client->input_layer->depth; i++) {
        client->input_layer->nodes[i].input = input_vector[i];
    }
    Layer* current_layer = client->input_layer;
    while (current_layer->layer_type != LAYER_TYPE_OUTPUT) {
        Forward_Pass_On_Layer(current_layer);
        current_layer = current_layer->next;
    }
    for (int i = 0; i < client->output_layer->depth; i++) {
        client->output_layer->nodes[i].output = Activation_Function(&client->output_layer->nodes[i]) + client->output_layer->nodes[i].bias;
    }
    return 0;
}
void Update_Parameters(AI_Client* client) {
    Layer* current_layer = client->output_layer;
    while (current_layer->layer_type != LAYER_TYPE_INPUT) {
        for (int i = 0; i < current_layer->depth; i++) {
            Node* node = &current_layer->nodes[i];
            node->bias -= node->error_signal * client->learning_rate;
            for (int j = 0; j < node->incoming_connection_count; j++) {
                node->incoming_connections[j]->weight -= node->incoming_connections[j]->weight_gradient * client->learning_rate;
            }
        }
        current_layer = current_layer->last;
    }
}
int Back_Propagate(AI_Client* client, float* true_output, uint16_t output_vector_depth) {
    if (output_vector_depth < client->output_layer->depth) {
        printf("Output vector is too small! Expects %i elements.", client->output_layer->depth);
        return 1;
    }
    client->output_layer->loss = BCE_Loss(client, true_output);
    Backwards_Pass_Output(client, true_output);
    Backwards_Pass_Hidden(client);
    Update_Parameters(client);
    return 0;
}
int Clear_Nodes(AI_Client* client) {
    Layer* current_layer = client->input_layer;
    while (current_layer->layer_type != LAYER_TYPE_OUTPUT) {
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
    switch(node->layer->layer_type) {
        case LAYER_TYPE_INPUT:
            //printf("Input node output: %f\n", node->input);
            return node->input;
            break;
        case LAYER_TYPE_HIDDEN:
            //printf("Hidden node input: %f\n", node->input);
            //printf("Hidden node output: %f\n", fmax(0, node->input));
            return fmax(0, node->input); // ReLU activation function
            break;
        case LAYER_TYPE_OUTPUT:
            //printf("Output node input: %f\n", node->input);
            //printf("Output node output: %f\n", sigmoid(node->input));
            return sigmoid(node->input); // Sigmoid activation function (Between 0 and 1)
            break;
    }
}
float BCE_Loss(AI_Client* client, float* true_output) {
    float loss = 0;
    for (int i = 0; i < client->output_layer->depth; i++) {
        float predicted_output = client->output_layer->nodes[i].input;
        loss += (true_output[i] * log(predicted_output) + (1 - true_output[i]) * log(1 - predicted_output));
    }
    loss *= (-1 / client->output_layer->depth);
    return loss;
}
void Update_Weight_Gradient(Connection* connection) {
    connection->weight_gradient = connection->sending_node->output * connection->receiving_node->error_signal;
}
void Backwards_Pass_Output(AI_Client* client, float* true_output) {
    Layer* output_layer = client->output_layer;
    Node* nodes = output_layer->nodes;
    for (int i = 0; i < output_layer->depth; i++) {
        nodes[i].error_signal = nodes[i].output - true_output[i];
        for (int j = 0; j < nodes[i].incoming_connection_count; j++) {
            Connection* connection = nodes[i].incoming_connections[j];
            Update_Weight_Gradient(connection);
        }
    }
}
void Backwards_Pass_Hidden(AI_Client* client) {
    Layer* current_layer = client->output_layer->last;
    while (current_layer->layer_type != LAYER_TYPE_INPUT) {
        Node* nodes = current_layer->nodes;
        for (int i = 0; i < current_layer->depth; i++) {
            nodes[i].error_signal = 0;
            if (nodes[i].input <= 0)
                continue;
            for (int j = 0; j < nodes[i].outgoing_connection_count; j++) {
                Connection* connection = nodes[i].outgoing_connections[j];
                nodes[i].error_signal += connection->weight * connection->receiving_node->error_signal;
            }
            for (int  j = 0; j < nodes[i].incoming_connection_count; j++) {
                Connection* connection = nodes[i].incoming_connections[j];
                Update_Weight_Gradient(connection);
            }
        }
        current_layer = current_layer->last;
    }
}