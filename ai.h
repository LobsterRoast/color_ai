#ifndef AI_HEADER
#define AI_HEADER

typedef struct Node Node;
typedef struct Layer Layer;
typedef struct Connection Connection;

typedef enum {
    LAYER_TYPE_INPUT  = 1 << 0, // 0b00000001
    LAYER_TYPE_HIDDEN = 1 << 1, // 0b00000010
    LAYER_TYPE_OUTPUT = 1 << 2  // 0b00000100
} Layer_Type;

struct Connection {
    float weight;
    Node* sending_node;
    Node* receiving_node;
};

struct Node {
    float bias;
    float input;
    float output;
    uint16_t incoming_connection_count;
    uint16_t outgoing_connection_count;
    bool activated;
    Layer* layer;
    Connection** incoming_connections;
    Connection** outgoing_connections;
};

struct Layer {
    Node* nodes;
    Layer* last;
    Layer* next;
    uint16_t depth;
    int layer_id;
    Layer_Type layer_type;
};

typedef struct {
    Layer* input_layer;
    Layer* output_layer;
    int layerId;
} AI_Client;


int Create_AI_Client(
    AI_Client* client, 
    uint16_t input_layer_depth, 
    uint16_t output_layer_depth, 
    uint16_t hidden_layer_depth, 
    uint16_t hidden_layer_count
    );
int Free_AI_Client(AI_Client* client);
int Forward_Propagate(AI_Client* client, float* input_vector, uint16_t input_vector_depth);
int Clear_Nodes(AI_Client* client);
float Activation_Function(Node* node);

#endif