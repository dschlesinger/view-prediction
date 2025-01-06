import tensorflow as tf, sklearn

from utils.colors import Colors

from typing import List, Tuple, Any, Callable, Literal, Dict
from itertools import product

from IPython.display import clear_output

# Import models
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers as TFLayers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Import metrics
from sklearn.metrics import accuracy_score, roc_auc_score

class models():
    """
    Pass in data in __init__ or assign_data

    Then use functions which return model trained and print eval_metrics
    """

    def __init__(self, get_xy, random_state: int = 42) -> None:
        """
        get_xy: Call the get_xy function on the data you want to use
        """

        self.assign_data(get_xy, random_state)

    def assign_data(self, get_xy, random_state: int = 42) -> None:
        """
        get_xy: Call the get_xy function on the data you want to use
        """

        self.random_state = random_state

        self.is_loader: bool = len(get_xy) == 2

        self.if_val: bool = len(get_xy) == 6

        if not self.is_loader:

            # Gets x_train first example and finds dimensions
            example_dim: int = get_xy[0][0].shape.__len__()

            self.feature_or_images: bool = 'images' if example_dim > 1 else 'features'

        if self.is_loader:

            self.train, self.test = get_xy

        elif self.if_val:

            self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = get_xy

        else:

            self.x_train, self.x_test, self.y_train, self.y_test = get_xy

            self.x_val = self.y_val = None

    def set_random_state(self, random_state: int) -> None:

        self.random_state = random_state

    def random_forest(self, **kwargs) -> sklearn.ensemble.RandomForestClassifier:

        if self.feature_or_images != 'features': 

            raise ValueError(f"{Colors.RED.value}Random Forest requires feature data, not images!{Colors.RED.value}")

        clf = RandomForestClassifier(**kwargs, random_state=self.random_state)

        clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_test)

        print(f"Random Forest Accuracy: {accuracy_score(self.y_test, y_pred)}")

        return clf
    
    def ResNet50(self, freeze_base_model: bool = True, lr: float = 1e-2, 
                 metrics: List[str] = ['accuracy'],
                 
                 classifier = lambda base_model_output: TFLayers.Dense(1, activation='sigmoid')(
                                TFLayers.Dense(128, activation='relu')(
                                    TFLayers.GlobalAveragePooling2D()(base_model_output)
                                )
                            ),
                 
                 **kwargs
        ) -> tf.keras.models.Model:

        if self.feature_or_images != 'images': 

            raise ValueError(f"{Colors.RED.value}ResNet50 requires image data, not features!{Colors.RED.value}")

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        base_model.trainable = not freeze_base_model

        model = tf.keras.models.Model(inputs=base_model.input, outputs=classifier(base_model.output))

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=metrics)

        model.fit(self.x_train, self.y_train, **kwargs)

        model_eval = model.evaluate(self.x_test, self.y_test)

        print(
            f"""
            ResNet stats

            BinaryCrossentropy Loss: {model_eval[0]}
            {'\n'.join([f'{metric}: {value}' for metric, value in zip(metrics, model_eval[1:])])}
            """
        )

        return model
    
    def SmallCNN(self, lr: float = 1e-2, 
                metrics: List[str] = ['accuracy'],
                **kwargs) -> tf.keras.models.Model:
        
        if self.feature_or_images != 'images': 

            raise ValueError(f"{Colors.RED.value}SmallCNN requires image data, not features!{Colors.RED.value}")
        
        def build_SmallCNN():

            input_ = tf.keras.Input(shape=(224,224,3))

            x = TFLayers.Conv2D(32, (8, 8), activation='relu')(input_)

            # 56, 56, 3
            x = TFLayers.MaxPooling2D((4, 4))(x)

            x = TFLayers.Conv2D(32, (4, 4), activation='relu')(x)

            # 7, 7, 3
            x = TFLayers.MaxPooling2D((4, 4))(x)
            
            # For GradCAM
            x = TFLayers.Conv2D(32, (3, 3), activation='relu', padding='same', name='last-conv-layer')(x)

            # 147
            x = TFLayers.Flatten()(x)

            x = TFLayers.Dense(28, activation='relu')(x)

            x = TFLayers.Dense(14, activation='relu')(x)

            x = TFLayers.Dense(1, activation='sigmoid')(x)

            model = tf.keras.models.Model(inputs=input_, outputs=x)

            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=metrics)

            return model
        
        model = build_SmallCNN()

        model.fit(self.x_train, self.y_train, **kwargs)

        model_eval = model.evaluate(self.x_test, self.y_test)

        print(
            f"""
            SmallCNN stats

            BinaryCrossentropy Loss: {model_eval[0]}
            {'\n'.join([f'{metric}: {value}' for metric, value in zip(metrics, model_eval[1:])])}
            """
        )

        return model
    
    def SmallCNN_torch(self, lr: float = 1e-2, 
                metrics: List[str] = ['accuracy'],
                **kwargs) -> tf.keras.models.Model:
        
        # if self.feature_or_images != 'images': 

        #     raise ValueError(f"{Colors.RED.value}SmallCNN requires image data, not features!{Colors.RED.value}")

        class SmallCNN(nn.Module):
            def __init__(self):
                super(SmallCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding='same')
                self.pool1 = nn.MaxPool2d(kernel_size=2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding='same')
                self.pool2 = nn.MaxPool2d(kernel_size=2)
                self.conv3 = nn.Conv2d(32, 16, kernel_size=4, padding='same')
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(16 * 7 * 7, 28)
                self.fc2 = nn.Linear(28, 14)
                self.fc3 = nn.Linear(14, 10)
                self.sigmoid = nn.Softmax(dim=1)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool1(x)
                x = torch.relu(self.conv2(x))
                x = self.pool2(x)
                x = torch.relu(self.conv3(x))
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        def train_and_evaluate(model, train_data, test_data, lr=1e-2, epochs=10, batch_size=32):

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for data in train_data:
                    inputs, targets = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(test_data):.4f}")

            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                correct = 0
                total = 0
                for data in test_data:
                    inputs, targets = data
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    test_loss += loss.item()
                    predicted = (outputs.squeeze() > 0.5).float()
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            print(f"\nSmallCNN stats")
            print(f"BinaryCrossentropy Loss: {test_loss / len(test_data):.4f}")
            print(f"Accuracy: {100 * correct / total:.2f}%")

            return model

        return train_and_evaluate(SmallCNN(), self.train, self.test)

    def grid_search(self, get_model: Callable, metric: Literal['accuracy', 'auc'] = 'accuracy', param_combinations: List[Dict] = None, training_kwargs: Dict = {}, **kwargs) -> List[Tuple[Dict[str, Any], list[Any, float]]]:
        """
        Pass in:
        get_model: a function which returns the model example M.random_forest
        **kwargs: as a kwarg=[searchable params]

        returns best_model, [(params, score)]

        best model schema

        {
          "model": model,
          "metric": str,
          "metric_score": float,
          "params": Dict[str, List],
        }
        """ 

        # If param_combinations is None or {}
        if not param_combinations:
            keys, values = kwargs.keys(), kwargs.values()
            grid_combinations = [dict(zip(keys, v)) for v in product(*values)]
        else:
            grid_combinations = param_combinations

        best_model = None
        best_score = 0
        best_params = None

        save_scores = []

        num_combinations: int =  grid_combinations.__len__()

        for i, param in enumerate(grid_combinations):

            print(f"{i + 1}/{num_combinations}")

            clear_output(wait=True)

            model = get_model(**param)

            model.fit(self.x_train, self.y_train, **training_kwargs)

            y_pred = model.predict(self.x_test)

            match metric:

                case 'accuracy':

                    score = accuracy_score(self.y_test, y_pred)

                case 'auc':

                    score = roc_auc_score(self.y_test, y_pred)

                case _:

                    raise ValueError(f'Metric must be \'accuracy\' or \'auc\' not {metric}')
                
            if score > best_score:

                best_score = score

                best_model = model

                best_params = param

            save_scores.append((param, score))

        return (
            {
                "model": best_model,
                "metric": metric,
                "metric_score": best_score,
                "params": best_params,
            },
            save_scores
        )