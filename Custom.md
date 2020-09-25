## Custom Loss Functions
There are several ways to define custom loss functions
1.  If there is no parameter comes with the loss function, you can directly define a function then use it when compile the model.
```python
def huber_fn(y_true,y_pred):
    error = y_ture - y_pred
    is_small_error = tf.abs(error) <1
    squared_loss = tf.square(error) /2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error,squared_loss,linear_loss)
```
When load the model with custom objects, you need to map the names to the objects:
```python
model = keras.models.load_model("model_with_custom_loss.h5", custom_objects={"huber_fn":huber_fn})
```
The issue is when save the model, threshold will not be saved.  
 2.   **A better way is to subclassing the keras.losses.Loss class, then implementing get_config() method. This way can be used to all other component of a model**
```python
class Huberloss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```
when save the model, parameters will be saved along with it, when load the model, need to map the class name to the class itself
```python
model = keras.models.load_model("model_with_custom_loss.h5", custom_objects={"Huberloss":Huberloss})
```
**When using custom object in a model, always map the object to its' name when load the corresponding model. Using subclassing if it is possible.**  

## Custom Activation
