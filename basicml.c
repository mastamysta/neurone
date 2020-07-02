#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

//HYPERPARAMETERS
const int INPUTWIDTH = 3;
const int HIDDENDEPTH = 3;
const int HIDDENWIDTH = 6;
const int OUTPUTWIDTH = 3;

//pow function because pow isnt included in gcc for linux?????

float power(float base, int index){
  float value = 1;
  for(int i = 0; i <= index - 1; i ++){
    value = value * base;
  }
  return value;
}

//structures ------------------------------------------------

//data storage structure
struct column{
  int length;
  float values[];
};
typedef struct column column;

struct table{
  int width;
  int depth;
  column *columns[];
};
typedef struct table table;

//low level structures
struct edge{
  float weight;
};
typedef struct edge edge;


//node structures
struct hiddenNode{
  float weight;
  float value;
  edge outputs[];
};
typedef struct hiddenNode hiddenNode;


struct outputNode{
  float value;
  float weight;
};
typedef struct outputNode outputNode;


struct inputNode{
  float weight;
  float value;
  edge outputs[];
};
typedef struct inputNode inputNode;


//layer structures
struct outputLayer{
  int width;
  outputNode *nodes[];
};
typedef struct outputLayer outputLayer;

struct hiddenLayer{
  int width;
  hiddenNode *nodes[];
};
typedef struct hiddenLayer hiddenLayer;


struct inputLayer{
  int width;
  inputNode *nodes[];
};
typedef struct inputLayer inputLayer;

struct hiddenLayers{
  int depth;
  hiddenLayer *hiddenLayers[];
};
typedef struct hiddenLayers hiddenLayers;


//network structure
struct network{
  inputLayer *inputLayer;
  outputLayer *outputLayer;
  hiddenLayers *hiddenLayers;
};
typedef struct network network;

//data table construction functions ----------------------------------------------

//create a table datastructure to feed into network (or use as labels)
//takes an array of columns
table *compileColumns(int num, column *columns[]){
  //initialise variable list

  //allocate table memory on heap and assign depth and width and first column vals
  table *table = malloc(sizeof(table) + sizeof(column *) * num);
  table->width = num;
  column *col = columns[0];
  int lengthCheck = col->length;
  table->depth = col->length;
  table->columns[0] = col;

  //check lengths of all subsequent column and then assign to table
  for(int i = 1; i <= num - 1; i ++){
    col = columns[i];
    //if length doesnt match first column print error and return null pointer
    if(col->length != lengthCheck){
      printf("Column lengths do not match\nFirst Column %i\n%i th Column: %i", lengthCheck, i, col->length);
      return NULL;
    }
    table->columns[i] = col;
  }
  return table;
}

//creates a column struct pointer from a pointer to an array of floats
column *createColumn(int length, float *values){
  column *column = malloc(sizeof(column) + sizeof(float) * length);
  column->length = length;
  //ignoring first index as this is a lenth marker for each column
  for(int i = 1; i <= length; i ++){
    column->values[i - 1] = *(values + i);
    printf("%f\n", column->values[i - 1]);
  }
  return column;
}

//creates a table from a variable number of float arrays
//takes a variable number of pointers to arrays of floats
table *createTable(int num, ...){
  column *columns[num];
  va_list valist;
  va_start(valist, num);
  for(int i = 0; i <= num - 1; i ++){
    float *col = va_arg(valist, float *);
    //length is stored in first index in array
    int length = *col;
    printf("Column Length: %i\n", length);
    columns[i] = createColumn(length, col);
  }
  va_end(valist);
  table *table = compileColumns(num, columns);
  return table;
}

//destroys a table structure
void destroyTable(table *table){
  for(int i = 0; i <= table->width - 1; i ++){
    free(table->columns[i]);
  }
  free(table);
}

//evaluation functions ------------------------------------------------

//predict values at input layer
void predictInputLayer(inputLayer *inputLayer, float *inputs[]){
  int layerWidth = inputLayer->width;
  for(int i = 0; i <= layerWidth - 1; i ++){
    float weight = inputLayer->nodes[i]->weight;
    inputLayer->nodes[i]->value = (*inputs)[i] * weight;
  }
}

//predict values in one hidden layer
void predictHiddenLayer(hiddenLayer *hiddenLayer, float *inputs[], int previousLayerWidth){
  for(int j = 0; j <= previousLayerWidth - 1; j ++){
    for(int i = 0; i <= HIDDENWIDTH - 1; i ++){
      float weight = hiddenLayer->nodes[i]->weight;
      hiddenLayer->nodes[i]->value += ((*inputs)[(j * HIDDENWIDTH) + i]) * weight;
    }
  }
}

//predict values in all hidden layers
void predictHiddenLayers(hiddenLayers *hiddenLayers, inputLayer *inputLayer){
  //predict value of first hidden layer based on inputs from input layer
  float *inputs = malloc(sizeof(float) * INPUTWIDTH * HIDDENWIDTH);
  //for each node in the input layer
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    //for each node in the first hidden layer
    for(int j = 0; j <= HIDDENWIDTH - 1; j ++){
      inputs[(i * HIDDENWIDTH) + j] = inputLayer->nodes[i]->outputs[j].weight * inputLayer->nodes[i]->value;
      //debug input array and edge weights
      //printf("INDEX: %i    Input Node: %i  Input val: %f  Edgeweight: %f  Nodeval: %f\n", (i * HIDDENWIDTH) + j, i, inputs[i * INPUTWIDTH + j], inputLayer->nodes[i]->outputs[j].weight, inputLayer->nodes[i]->value);
    }
  }
  printf("      Predicting hidden layer 0\n");
  predictHiddenLayer(hiddenLayers->hiddenLayers[0], &inputs, INPUTWIDTH);
  free(inputs);
  float *newInputs = malloc(sizeof(float) * HIDDENWIDTH * HIDDENWIDTH);
  //predict values of all other layers based on inputs from previous layer
  //for each hidden layer
  for(int i = 1; i <= HIDDENDEPTH - 1; i ++){
    //for each node in the previous layer
    for(int j = 0; j <= HIDDENWIDTH - 1; j ++){
      //for each edge in this node
      for(int k = 0; k <= HIDDENWIDTH - 1; k ++){
        inputs[(j * HIDDENWIDTH) + k] = hiddenLayers->hiddenLayers[i - 1]->nodes[j]->outputs[k].weight * hiddenLayers->hiddenLayers[i - 1]->nodes[j]->value;
        //debug input array and edge weights
        //printf("INDEX: %i    Input Node: %i  Input val: %f  Edgeweight: %f  Nodeval: %f\n", (j * HIDDENWIDTH) + k, j, inputs[(j * HIDDENWIDTH) + k], hiddenLayers->hiddenLayers[i]->nodes[j]->outputs[k].weight, hiddenLayers->hiddenLayers[i - 1]->nodes[j]->value);
      }
    }
    printf("      Predicting hidden layer %i\n", i);
    predictHiddenLayer(hiddenLayers->hiddenLayers[i], &inputs, HIDDENWIDTH);
  }
  printf("      All hidden layers predicted\n");
  //free(newInputs);
}

//predict values in output layer
void predictOutputLayer(outputLayer *outputLayer, hiddenLayer *hiddenLayer){
  float *inputs = malloc(sizeof(float) * HIDDENWIDTH * OUTPUTWIDTH);
  //for each node in hidden layer
  for(int i = 0; i <= HIDDENWIDTH - 1; i ++){
    //for each edge in this node
    for(int j = 0; j <= OUTPUTWIDTH - 1; j ++){
      inputs[(i * HIDDENWIDTH) + j] = hiddenLayer->nodes[i]->outputs[j].weight * hiddenLayer->nodes[i]->value;
    }
  }
  //for each node in last hidden layer
  for(int i = 0; i <= HIDDENWIDTH - 1; i ++){
    //for each node in output layer
    for(int j = 0; j <= OUTPUTWIDTH - 1; j ++){
      outputLayer->nodes[j]->value += outputLayer->nodes[j]->weight * inputs[(i * HIDDENWIDTH) + j];
    }
  }
  free(inputs);
}

//predict values at all nodes in the network
void predict(network *net, float* inputs){
  printf("    Starting predict input layer function\n");
  predictInputLayer(net->inputLayer, &inputs);
  printf("    Predict input layer function exited without error\n");
  printf("    Starting predict hidden layers function\n");
  predictHiddenLayers(net->hiddenLayers, net->inputLayer);
  printf("    Predict hidden layers function exited without error\n");
  printf("    Starting predict output layer function\n");
  predictOutputLayer(net->outputLayer, net->hiddenLayers->hiddenLayers[HIDDENDEPTH - 1]);
  printf("    Predict output layer function exited without error\n");
}

//training functions -----------------------------------------------------

//a random float value from 0-1 TODO:ADD DISTRIBUTION TYPES AND MONITOR EFFECTS
float generateNoise(){
  const float MAX = 1;
  //generates a float value from 0 to 1 randomly
  float x = (float)rand()/(float)(RAND_MAX/MAX);
  return x;
}

//calculate error squared for a dataset
float calculateErrorSquared(network *net, float *label[]){
  float error = 0;
  for(int i = 0; i <= OUTPUTWIDTH - 1; i ++){
    error += power(net->outputLayer->nodes[i]->value - (*label)[i], 2);
  }
  return error;
}

//assign a output layer node weight
void assignOutputNodeWeight(network *net, int node, float value){
  net->outputLayer->nodes[node]->weight = value;
}

//assign a hidden layer edge weight
void assignHiddenEdgeWeight(network *net, int layer, int node, int edge, float value){
  net->hiddenLayers->hiddenLayers[layer]->nodes[node]->outputs[edge].weight = value;
}

//assign a hidden layer node weight
void assignHiddenNodeWeight(network *net, int layer, int node, float value){
  net->hiddenLayers->hiddenLayers[layer]->nodes[node]->weight = value;
}

//assign an input layer edge weight
void assignInputEdgeWeight(network *net, int node, int edge, float value){
  net->inputLayer->nodes[node]->outputs[edge].weight = value;
}

//assign an input node weight
void assignInputNodeWeight(network *net, int node, float value){
  net->inputLayer->nodes[node]->weight = value;
}

void backPropagate(){

}

float findErrorSquaredOfExample(network *net, table *features, table *labels){
  //put features into an array (JANKY)
  printf("hereA\n");
  float featuresArray[features->depth];
  for(int i = 0; i <= features->depth - 1; i ++){
    featuresArray[i] = features->columns[0]->values[i];
  }
  printf("hereB\n");
  //similarly put labels into an array
  float labelsArray[labels->depth];
  for(int i = 0; i <= labels->depth - 1; i ++){
    labelsArray[i] = labels->columns[0]->values[i];
  }
  printf("hereC\n");
  predict(net, featuresArray);
  printf("hereD\n");
  float errorSquared = calculateErrorSquared(net, labelsArray);
  return errorSquared;
}

// void train(network *net, table *data, table *labels,int epochs){

// }

//initialization functions --------------------------------------------------

//set parameters in input layer to be identity
void identityInputLayer(network *net){
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    //set input node weights to 1
    net->inputLayer->nodes[i]->weight = 1;
    for(int j = 0; j <= HIDDENWIDTH - 1; j ++){
      //set input node edge weights to 1
      net->inputLayer->nodes[i]->outputs[j].weight = 1;
    }
  }
}

//set parameters in hidden layers to be identity
void identityHiddenLayers(network *net){
  //set last hidden layer weights to 1 first
  for(int i = 0; i <= HIDDENWIDTH - 1; i ++){
    net->hiddenLayers->hiddenLayers[HIDDENDEPTH - 1]->nodes[i]->weight = 1;
    for(int j = 0; j <= OUTPUTWIDTH - 1; j ++){
      net->hiddenLayers->hiddenLayers[HIDDENDEPTH - 1]->nodes[i]->outputs[j].weight = 1;
    }
  }
  //set all other hidden layer weights to 1 in reverse order
  for(int i = HIDDENDEPTH - 2; i >= 0; i --){
    for(int j = 0; j <= HIDDENWIDTH - 1; j ++){
      net->hiddenLayers->hiddenLayers[i]->nodes[j]->weight = 1;
      for(int k = 0; k <= HIDDENWIDTH - 1; k ++){
        net->hiddenLayers->hiddenLayers[i]->nodes[j]->outputs[k].weight = 1;
      }
    }
  }
}

//set parameters in output layer to be identity
void identityOutputLayer(network *net){
  for(int i = 0; i <= OUTPUTWIDTH - 1; i ++){
    //set output node weights to 1
    net->outputLayer->nodes[i]->weight = 1;
  }
}

//set all parameters in network to be identity
void identityNetwork(network *net){
  identityInputLayer(net);
  identityHiddenLayers(net);
  identityOutputLayer(net);
  printf("Network initialised as identity network\n");
}

//allocate heap space for input layer
inputLayer *allocateInputLayer(){
  //input layer pointer generated with correct size allocation
  inputLayer *inputLayer = malloc(sizeof(inputLayer) + sizeof(inputNode *) * INPUTWIDTH);
  inputLayer->width = INPUTWIDTH;
  for(int i = 0; i <= INPUTWIDTH - 1; i++){
    inputLayer->nodes[i] = malloc(sizeof(inputNode) + sizeof(edge) * HIDDENWIDTH);
  }
  return inputLayer;
}

//allocate heap space for each hidden layer
hiddenLayers *allocateHiddenLayers(){
  hiddenLayers *hiddenLayers = malloc(sizeof(hiddenLayers) + sizeof(hiddenLayer *) * HIDDENDEPTH);
  hiddenLayers->depth = HIDDENDEPTH;
  //Allocate final layer of hidden layers first with |E| from each node equal to OUTPUTWIDTH
  hiddenLayers->hiddenLayers[HIDDENDEPTH - 1] = malloc(sizeof(hiddenLayer) + sizeof(hiddenNode) * HIDDENWIDTH);
  hiddenLayers->hiddenLayers[HIDDENDEPTH - 1]->width = HIDDENWIDTH;
  for(int i = 0; i <= HIDDENWIDTH - 1; i ++){
    hiddenLayers->hiddenLayers[HIDDENDEPTH - 1]->nodes[i] = malloc(sizeof(hiddenNode) + sizeof(edge) * OUTPUTWIDTH);

  }
  //Allocate all other layers in reverse with |E| from each node equal to HIDDENWIDTH
    for(int i = HIDDENDEPTH - 2; i >= 0; i --){
    hiddenLayers->hiddenLayers[i] = malloc(sizeof(hiddenLayer) + sizeof(hiddenNode *) * HIDDENWIDTH);
    hiddenLayers->hiddenLayers[i]->width = HIDDENWIDTH;
    for(int j = 0; j <= HIDDENWIDTH - 1; j ++){
      hiddenLayers->hiddenLayers[i]->nodes[j] = malloc(sizeof(hiddenNode) + sizeof(edge) * HIDDENWIDTH);
    }
  }
  return hiddenLayers;
}

//allocate heap space for output layer
outputLayer *allocateOutputLayer(){
  outputLayer *outputLayer = malloc(sizeof(outputLayer) + sizeof(outputNode *) * OUTPUTWIDTH);
  outputLayer->width = OUTPUTWIDTH;
  for(int i = 0; i <= OUTPUTWIDTH - 1; i ++){
    outputLayer->nodes[i] = malloc(sizeof(outputNode));
  }
  return outputLayer;
}

//allocates memory in heap for network
network *allocateNetwork(){
  //network struct contains pointers to layers thus allocated staticly
  network *net = malloc(sizeof(network));

  //generate input layer pointer
  inputLayer *inputLayer = allocateInputLayer();
  //generate hidden layer pointer
  hiddenLayers *hiddenLayers = allocateHiddenLayers();
  //generate outputlayer pointer
  outputLayer *outputLayer = allocateOutputLayer();

  //assign pointers to network struct
  net->inputLayer = inputLayer;
  net->hiddenLayers = hiddenLayers;
  net->outputLayer = outputLayer;

  printf("Network heap space allocated\n");
  return net;
}

//allocate memory and identity parameters for all nodes and edges
network *generateNetwork(){
  network *net = allocateNetwork();
  identityNetwork(net);

  printf("Network initialisation finished\n");
  return net;
}

//cleanup functions --------------------------------------

//free space on heap for input layer
void freeInputLayer(inputLayer *inputLayer){
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    free(inputLayer->nodes[i]);
  }
  free(inputLayer);
}

//free space on heap for hidden layers
void freeHiddenLayers(hiddenLayers *hiddenLayers){
  //free all other layers of hidden layers in reverse order
  for(int i = 0; i <= HIDDENDEPTH - 1; i ++){
    for(int j = 0; j <= HIDDENWIDTH - 1; j ++){
      free(hiddenLayers->hiddenLayers[i]->nodes[j]);
    }
    free(hiddenLayers->hiddenLayers[i]);
  }
  free(hiddenLayers);
}

//free space on heap for output layer
void freeOutputLayer(outputLayer *outputLayer){
  for(int i = 0; i <= OUTPUTWIDTH; i ++){
    free(outputLayer->nodes[i]);
  }
  free(outputLayer);
}

//free space on heap for network
void freeNetwork(network *net){
  freeInputLayer(net->inputLayer);
  freeHiddenLayers(net->hiddenLayers);
  freeOutputLayer(net->outputLayer);
  free(net);
  printf("Network destroyed\n");
}

//testing --------------------------------------------------------------

//print out the values at each node of the network
void printNodeValues(network *net){
  printf("Input Layer\n");
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    printf("%f  ", net->inputLayer->nodes[i]->value);
  }
  printf("\nHidden Layers\n");
  for(int i = 0; i <= HIDDENDEPTH - 1; i ++){
    for(int j = 0; j <= HIDDENWIDTH - 1; j ++){
      printf("%f  ", net->hiddenLayers->hiddenLayers[i]->nodes[j]->value);
    }
    printf("\n");
  }
  printf("\nOutput Layer\n");
  for(int i = 0; i <= OUTPUTWIDTH - 1; i ++){
    printf("%f  ", net->outputLayer->nodes[i]->value);
  }
  printf("\n\n");
}

//test network generation and deletion functions
void testGenerateNetwork(){
  network *net = generateNetwork();
  printf("Network generation passed test\n\n");
  freeNetwork(net);
  printf("Network deletion passed test\n\n");
}

//test random noise generator function works between bounds of 0 and 1
void testGenerateNoise(int iterations){
  float randomValue;
  for(int i = 0; i <= iterations; i ++){
    randomValue = generateNoise();
    assert(randomValue <= 1);
    assert(randomValue >= 0);
  }
  printf("Noise generattion passed %i test iterations within bounds\n\n", iterations);
}

//test the prediction functtion for the input layer
void testPredictInputLayer(){
  network *net = generateNetwork();
  float *inputs = malloc(sizeof(float) * 3);
  for(int i = 0; i <= 2; i ++){
    inputs[i] = i;
  }
  predictInputLayer(net->inputLayer, &inputs);

  printNodeValues(net);
  assert(net->inputLayer->nodes[0]->value == 0);
  assert(net->inputLayer->nodes[1]->value == 1);
  assert(net->inputLayer->nodes[2]->value == 2);
  free(inputs);
  freeNetwork(net);
  printf("Input layer predictions passed tests\n\n");
}

//test the prediction function for the hidden layers
void testPredictHiddenLayers(){
  network *net = generateNetwork();
  float *inputs = malloc(sizeof(float) * 3);
  float sum = 0;
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    inputs[i] = i * 5;
    sum += inputs[i];
  }
  predictInputLayer(net->inputLayer, &inputs);
  predictHiddenLayers(net->hiddenLayers, net->inputLayer);
  //assertions
  printNodeValues(net);
  assert(net->hiddenLayers->hiddenLayers[0]->nodes[0]->value == (float)sum);
  assert(net->hiddenLayers->hiddenLayers[1]->nodes[1]->value == (float)sum * HIDDENWIDTH);
  assert(net->hiddenLayers->hiddenLayers[2]->nodes[2]->value == (float)sum * HIDDENWIDTH * HIDDENWIDTH);

  free(inputs);
  freeNetwork(net);
  printf("Hidden layer predictions passed tests\n\n");
}

//test the predicition function for the output layer
void testPredictOutputLayer(){
  network *net = generateNetwork();
  float *inputs = malloc(sizeof(float) * 3);
  float sum = 0;
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    inputs[i] = i;
    sum += i;
  }
  predictInputLayer(net->inputLayer, &inputs);
  predictHiddenLayers(net->hiddenLayers, net->inputLayer);
  predictOutputLayer(net->outputLayer, net->hiddenLayers->hiddenLayers[HIDDENDEPTH - 1]);
  //assertions
  printNodeValues(net);
  assert(net->hiddenLayers->hiddenLayers[0]->nodes[0]->value == (float)sum);
  assert(net->hiddenLayers->hiddenLayers[1]->nodes[1]->value == (float)sum * HIDDENWIDTH);
  assert(net->hiddenLayers->hiddenLayers[2]->nodes[2]->value == (float)sum * HIDDENWIDTH * HIDDENWIDTH);
  assert(net->outputLayer->nodes[2]->value == (float)(sum * power(HIDDENWIDTH, HIDDENDEPTH)));

  free(inputs);
  freeNetwork(net);
  printf("Output layer predictions passed tests\n\n");
}

void testPredict(){
  network *net = generateNetwork();
  float *inputs = malloc(sizeof(float) * 3);
  float sum = 0;
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    inputs[i] = i;
    sum += i;
  }
  printf("  Starting predict function\n");
  predict(net, inputs);
  printf("  Predict function exited without error\n");
  printf("  Starting print node value function\n");
  printNodeValues(net);
  printf("  Node values printed successfully\n");

  printf("  Starting freeing inputs array in heap\n");
  free(inputs);
  printf("  Inputs freed sucessfully\n");
  printf("  Starting freeing network function\n");
  freeNetwork(net);
  printf("  Network freed successfully\n");
  printf("  Predict function passed tests\n");
}

//test the error calculation function
void testCalculateErrorSquared(){
  network *net = generateNetwork();
  float *inputs = malloc(sizeof(float) * INPUTWIDTH);
  for(int i = 0; i <= INPUTWIDTH - 1; i ++){
    inputs[i] = 1;
  }
  predict(net, &inputs);
  float *labels = malloc(sizeof(float) * OUTPUTWIDTH);
  for(int i = 0; i <= OUTPUTWIDTH - 1; i ++){
    labels[i] = 1 * INPUTWIDTH * power(HIDDENWIDTH, HIDDENDEPTH);
  }
  float error = calculateErrorSquared(net, &labels);
  printNodeValues(net);
  assert(error == 0);

  free(inputs);
  free(labels);
  freeNetwork(net);
  printf("Error calculation passed tests\n");
}

void testCreateTable(){
  float testData[] = {3, 0, 1, 2};
  float col2[] = {3, 2, 3, 4};
  for(int i = 1; i <= testData[0]; i ++){
    printf("Data Entry: %i  -  %f\n", i, testData[i]);
  }
  table *table = createTable(2, testData, col2);
  for(int i = 0; i <= 2; i ++){
    printf("Table Index %i: %f\n", i, table->columns[0]->values[i]);
  }
  for(int i = 0; i <= 2; i ++){
    printf("Table Index %i: %f\n", i, table->columns[1]->values[i]);
  }
  destroyTable(table);
  printf("Table creation passed tests\n");
}

void testFindErrorSquaredOfExample(){
  network *net = generateNetwork();
  float testFeatures[INPUTWIDTH + 1];
  //first entry must be column length
  testFeatures[0] = INPUTWIDTH;
  for(int i = 0; i <= INPUTWIDTH - 1; i++){
    testFeatures[i + 1] = i;
  }
  float testLabels[OUTPUTWIDTH + 1];
  //first entry must be column length
  testLabels[0] = OUTPUTWIDTH;
  for(int i = 0; i <= OUTPUTWIDTH - 1; i++){
    testLabels[i + 1] = i;
  }

  table *featureTable = createTable(1, testFeatures);
  table *labelTable = createTable(1, testLabels);

  float error = findErrorSquaredOfExample(net, featureTable, labelTable);
  printf("%f\n", error);

  destroyTable(featureTable);
  destroyTable(labelTable);
  freeNetwork(net);
}

void testPower(){
  assert(power(3, 2) == 9);
  assert(power(5, 0) == 1);
}

//main test runner
void test(){
 // testGenerateNetwork();
  //testGenerateNoise(100);
  //testPredictInputLayer();
  //testPredictHiddenLayers();
  //testPredictOutputLayer();
  //testPredict();
  testCalculateErrorSquared();
  //testCreateTable();
  //testFindErrorSquaredOfExample();
  //testPower();

  printf("All tests passed\nExiting...\n");
}

//main runner ----------------------------------------------------

int main(){
  test();
}
