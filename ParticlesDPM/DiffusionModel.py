import torch
import torch.nn as nn

class DiffusionLayer(nn.Module):
    """
    Para la creación de los Hidden Layers de la red neuronal.
    Todos los Layers son lineales y tienen una función de activación RELU
    """
    def __init__(self, nNeurons):
        super(DiffusionLayer, self).__init__()
        self.linear = nn.Linear(nNeurons, nNeurons) #(nInputs, nNeurons), Each block is a nxn neural network

    def forward(self, x:torch.Tensor):
        """
        Función de activación: ReLu
        """
        x = self.linear(x)
        x = nn.functional.relu(x)
        return x
    

class DiffusionModel(nn.Module):

    def __init__(self, nInputs, nLayers=2, nNeurons=64):
        """
        Para la creación del Modelo de difusión. De acuerdo al paper de Ho. et al,
        este es epsilon_theta, lo que usaremos para predecir el ruido en la imagen.
        Es un Deep Neural Network con múltiples Layers Lineales.
        """
        super(DiffusionModel, self).__init__()

        #Input Block:
        self.inputBlock = nn.Linear(nInputs+1, nNeurons)
        #Diffusion Layers:
        self.diffLayers = nn.ModuleList([DiffusionLayer(nNeurons) for i in range(nLayers)])
        #Output Block:
        self.outputBlock = nn.Linear(nNeurons, nInputs)

    def forward(self, x, t):
        """
        Función para la activación de la red neuronal. 
        x es el conjunto de datos
        t es el tiempo actual de la cadena de difusión

        retorna la predicción epsilon_theta
        """
        value = torch.hstack([x,t]) # Agrega t al arreglo de datos x
        # Forward evaluations:
        val = self.inputBlock(value)
        for diffLayer in self.diffLayers:
            val = diffLayer(val)
        val = self.outputBlock(val)

        return val