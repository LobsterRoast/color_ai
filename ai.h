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

typedef struct {
    Layer* input_layer;
    Layer* output_layer;
    float learning_rate;
    int layer_id;
} AI_Client;

struct Connection {
    float weight;
    float weight_gradient;
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
    float error_signal;
};

struct Layer {
    Node* nodes;
    Layer* last;
    Layer* next;
    uint16_t depth;
    int layer_id;
    Layer_Type layer_type;
    float loss;
};

int Create_AI_Client(
    AI_Client* client, 
    uint16_t input_layer_depth, 
    uint16_t output_layer_depth, 
    uint16_t hidden_layer_depth, 
    uint16_t hidden_layer_count,
    uint16_t learning_rate
    );
int Free_AI_Client(AI_Client* client);
int Forward_Propagate(AI_Client* client, float* input_vector, uint16_t input_vector_depth);
int Back_Propagate(AI_Client* client, float* true_output, uint16_t output_vector_depth);
int Clear_Nodes(AI_Client* client);
float Activation_Function(Node* node);
// Mean Square Error Loss Function (For Regression)
float MSE_Loss(AI_Client* client, float* true_output);
// Cross Entropy Loss Function (For Multi-Class Classification)
float Cross_Entropy_Loss(AI_Client* client, float* true_output);
// Binary Cross-Entropy Loss (For Multi-Label Classification)
float BCE_Loss(AI_Client* client, float* true_output);

void Backwards_Pass_Output(AI_Client* client, float* true_output);
void Backwards_Pass_Hidden(AI_Client* client);

#endif