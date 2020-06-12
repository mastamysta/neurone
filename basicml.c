#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

const int INPUTWIDTH = 3;
const int HIDDENDEPTH = 3;
const int HIDDENWIDTH = 3;
const int OUTPUTWIDTH = 3;

//structures ------------------------------------------------


//low level structures
struct edge{
  float weight;
};
typedef struct edge edge;


//node structures
struct hiddenNode{
  float weight;
  edge outputs[];
};
typedef struct hiddenNode hiddenNode;


struct outputNode{
  float weight;
};
typedef struct outputNode outputNode;


struct inputNode{
  float weight;
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

//methods -----------------------------------------------------------

void identityInputLayer(network *net){

}

void identityHiddenLayers(network *net){

}

void identityOutputLayer(network *net){

}

void identityNetwork(network *net){
  identityInputLayer(net);
  identityHiddenLayers(net);
  identityOutputLayer(net);
}

inputLayer *allocateInputLayer(){
  //input layer pointer generated with correct size allocation
  inputLayer *inputLayer = malloc(sizeof(inputLayer) + sizeof(inputNode *) * INPUTWIDTH);
  inputLayer->width = INPUTWIDTH;
  for(int i = 0; i > INPUTWIDTH; i++){
    inputLayer->nodes[i] = malloc(sizeof(inputNode) + sizeof(edge) * HIDDENWIDTH);
  }
  return inputLayer;
}

hiddenLayers *allocateHiddenLayers(){
  hiddenLayers *hiddenLayers = malloc(sizeof(hiddenLayers) + sizeof(hiddenLayer *) * HIDDENDEPTH);
  hiddenLayers->depth = HIDDENDEPTH;
}

network *allocateNetwork(){
  //network struct contains pointers to layers thus allocated staticly
  network *net = malloc(sizeof(network));

  //generate input layer pointer
  inputLayer *inputLayer = allocateInputLayer();

  return net;
}

network *generateNetwork(){
  network *net = allocateNetwork(INPUTWIDTH, HIDDENWIDTH, HIDDENDEPTH, OUTPUTWIDTH);
  identityNetwork(net);
}

int main(int n, char *args[n]){

}
