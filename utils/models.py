import tensorflow as tf, sklearn

from utils.colors import Colors

from typing import List

# Import models
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers as TFLayers

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

        self.if_val: bool = len(get_xy) == 6

        self.feature_or_images: bool = 'images' if get_xy[0][0].shape.__len__() > 1 else 'features'

        if self.if_val:

            self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = get_xy

        else:

            self.x_train, self.x_test, self.y_train, self.y_test = get_xy

            self.x_val = self.y_val = None

    def set_random_state(self, random_state: int) -> None:

        self.random_state = random_state

    def random_forest(self, **kwargs) -> sklearn.ensemble.RandomForestClassifier:

        if self.feature_or_images != 'feature': 

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