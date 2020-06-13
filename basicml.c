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
  for(int i = 0; i <= INPUTWIDTH - 1; i++){
    inputLayer->nodes[i] = malloc(sizeof(inputNode) + sizeof(edge) * HIDDENWIDTH);
  }
  return inputLayer;
}

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

network *allocateNetwork(){
  //network struct contains pointers to layers thus allocated staticly
  network *net = malloc(sizeof(network));

  //generate input layer pointer
  inputLayer *inputLayer = allocateInputLayer();
  //generate hidden layer pointer
  hiddenLayers *hiddenLayers = allocateHiddenLayers();

  return net;
}

network *generateNetwork(){
  network *net = allocateNetwork(INPUTWIDTH, HIDDENWIDTH, HIDDENDEPTH, OUTPUTWIDTH);
  identityNetwork(net);
}

int main(int n, char *args[n]){

}
