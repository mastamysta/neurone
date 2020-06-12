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
  outputNode nodes[];
};
typedef struct outputLayer outputLayer;

struct hiddenLayer{
  int width;
  hiddenNode nodes[];
};
typedef struct hiddenLayer hiddenLayer;


struct inputLayer{
  int width;
  inputNode nodes[];
};
typedef struct inputLayer inputLayer;


//network structure
struct network{
  inputLayer inputLayer;
  outputLayer outputLayer;
  hiddenLayer hiddenLayers[];
};
typedef struct network network;

//methods -----------------------------------------------------------

void identityInputLayer(network *net){

}

void identityHiddenLayers(network *net){

}

void identityOutputLayer(network *net){
  net->outputLayer
}

void identityNetwork(network *net){
  identityInputLayer(net);
  identityHiddenLayers(net);
  identityOutputLayer(net);
}

network *allocateNetwork(int inputLayerWidth, int hiddenLayerWitdth, int hiddenLayerDepth, int outputLayerWidth){
  network *net = malloc(sizeof(network));
  return net;
}

network *generateNetwork(){
  network *net = allocateNetwork();
  identityNetwork(net);
}

int main(int n, char *args[n]){

}
