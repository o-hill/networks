'''

    Generalized neural network built in Keras to facilitate easy deep learning development.

'''

import keras
import numpy as np
import json

import os

from ipdb import set_trace as debug


class Network:
    '''Base class for all the networks.'''

    def __init__(self, name, save_location, model=None):
        self.name = name
        self.model = model
        self.save = save_location

        # Configure the paths for the model.
        self.base_path = f'{self.save}/{self.name}'
        self.model_path = self.base_path + '/model.json'
        self.weights_path = self.base_path + '/weights.h5'

        # Try to save the model to disk immediately.
        self._save()


    def _create(self):
        raise NotImplementedError


    def _load(self):
        '''Attempt to load the model from disk.'''

        # Prepare for the inevitable.
        error = RuntimeError(
                        '''> Error loading network from the files. Format must be as follows:
                        \tname: example
                        \tsave_location: ~/code/example/argos
                        Extraneous slashes will cause an issue.
                        ''')

        # Try to find the model.
        if os.path.exists(self.base_path) \
            and os.path.exists(self.model_path) \
            and os.path.exists(self.weights_path):

                try:
                    with open(self.model_path, 'r') as jsonmodel:
                        self.model = keras.models.model_from_json(jsonmodel)

                    self.model.load_weights(self.weights_path, by_name=False)

                except:
                    raise error
        else:
            raise error

        print(f'> {self.name} loaded from {self.save} successfully!')
        return self.model


    def _save(self):
        '''Save the model to disk.'''

        if self.model:
            print('hello')
            # Make sure the base path exists.
            if not os.path.exists(self.base_path):
                print(f'Creating a directory in path: {self.base_path}')
                os.makedirs(self.base_path)

            # Save the model configuration, the weights, and the hyperparameters to disk.
            with open(self.model_path, 'w') as modelfile:
                print('Writing file...')
                modelfile.write(self.model.to_json())

            self.model.save_weights(self.weights_path)

        else:
            raise RuntimeError(f'> {self.name} has not been loaded from disk yet, and cannot be saved')


    def _train(self, X: np.array = None, y: np.array = None, epochs: int = 10):
        '''Performs stochastic gradient descent on the given data.'''
        self.model.fit(x=X, y=y)


    def _evaluate(self, X: np.array = None, y: np.array = None):
        '''Evaluate the models progress.'''
        return self.model.evaluate(X, y)


    def _predict(self, X: np.array = None):
        '''Generate prediction for the model with the given data.'''
        return self.model.predict(X)



class TwoDimensionalCNN(Network):
    '''Generic 2D CNN model.'''

    def __init__(self, name = f'default_network_{ np.random.randint(10000) }',
                 save_location = '~/networks',
                 new_model = True,
                 input_sizes = [128, 128, 3],
                 output_size = 32,
                 num_conv_layers = 4,
                 num_connected_layers = 2,
                 num_starting_filters = 128,
                 filter_size = [5, 5],
                 stride_size = [2, 2],
                 activation_function = 'relu',
                 padding = 'same',
                 pooling = 'max',
                 pool_size = [2, 2],
                 loss_function = keras.losses.categorical_crossentropy,
                 optimizer = keras.optimizers.Adam(lr = 0.001)
    ):

        model = None

        if new_model:
            layer_list = self.create(locals())
            model = keras.models.Sequential(layer_list)

            model.compile(
                loss = loss_function,
                optimizer = optimizer,
                metrics = ['accuracy']
            )

        Network.__init__(self, name, save_location, model = model)

        if not new_model:
            self._load()



    def create(self, hparams):
        '''Build the network layers.'''

        # Start building the layer list.
        layer_list = [ keras.layers.Conv2D(
                filters = hparams['num_starting_filters'],
                kernel_size = hparams['filter_size'],
                strides = hparams['stride_size'],
                padding = hparams['padding'],
                activation = hparams['activation_function'],
                input_shape = hparams['input_sizes']
            )
        ]

        num_filters = hparams['num_starting_filters'] * 2
        hparams['num_conv_layers'] -= 1

        for layer in range(hparams['num_conv_layers']):

            layer_list.append(keras.layers.Conv2D(
                filters = num_filters,
                kernel_size = hparams['filter_size'],
                strides = hparams['stride_size'],
                padding = hparams['padding'],
                activation = hparams['activation_function']
            ))

            if layer < 1:
                num_filters *= 2
            else:
                num_filters /= 2
                num_filters = int(num_filters)

            if layer % 2 == 0 and layer > 0:
                if hparams['pooling'] == 'max':
                    layer_list.append(keras.layers.MaxPooling2D(hparams['pool_size']))
                elif hparams['pooling'] == 'average':
                    layer_list.append(keras.layers.AveragePooling2D(hparams['pool_size']))
                else:
                    raise ValueError(f'{hparams["pooling"]} pooling is not supported at this time! Use either max or average')

        # Flatten the convolutional layers in preparation for fully connected layers.
        layer_list.append(keras.layers.Flatten())

        # Add the fully connected layers.
        dense_units = max(2 ** (hparams['num_connected_layers'] + 1), 128)
        for layer in range(hparams['num_connected_layers'] - 1):
            layer_list.append(keras.layers.Dense(dense_units))
            layer_list.append(keras.layers.Activation(hparams['activation_function']))
            dense_units /= 2
            dense_units = int(dense_units)

        print(f'Penultimate dense layer units: {dense_units}')
        layer_list.append(keras.layers.Dense(hparams['output_size']))

        # All done!
        return layer_list


    def save(self):
        '''Save the model to disk.'''
        self._save()


    def train(self, X: np.array = None, y: np.array = None, epochs: int = 10, validation_percent: float = 0.0, required_accuracy: float = 0.95):
        '''
            Train the model using stochastic gradient descent.

            Important: this function does everything for you,
            just supply the full amount of training data.

        '''

        accuracy = 0
        iteration = 0
        validation_set = [ ]
        training_set = [ ]
        validation_indices = np.random.randint(X.shape[0], size=(int(X.shape[0] * validation_percent)))

        print(f'> {self.name} generating validation and training sets.')
        validation_indices = np.random.randint(X.shape[0])
        training_indices = list(set(range(X.shape[0])) - set(validation_indices))

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        X_training = X[training_indices]
        y_training = y[training_indices]

        print(f'> {self.name} beginning training.')
        while accuracy < required_accuracy:
            self._train(X_training, y_training, epochs=epochs)

            accuracy = self._evaluate(X_validation, y_validation)
            print(f'> {self.name} at iteration {iteration} evaluated at an accuracy of {accuracy}%.')

            if iteration % 10 == 0:
                print(f'> {self.name} saving to disk at iteration {iteration}')
                self._save()

        print(f'> {self.name} training complete! Saving...')
        self._save()


    def evaluate(self, X: np.array = None, y: np.array = None):
        '''Evaluate the current model.'''
        self._evaluate(X, y)


    def predict(self, X: np.array = None):
        '''Get predictions from the model.'''
        return self._predict(X)




if __name__ == '__main__':
    cnn = TwoDimensionalCNN()









