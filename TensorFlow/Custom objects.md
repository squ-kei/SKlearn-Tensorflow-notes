## Custom Loss Functions
There are several ways to define custom loss functions  
**1. If there is no parameter comes with the loss function, you can directly define a function then use it when compile the model.**
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
 **2.   A better way is to subclassing the keras.losses.Loss class, then implementing get_config() method. This way can be used to all other component of a model**
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

## Custom Activation, Regularizer, Constraints, Initializer
These can be done similar as custom loss. Can subclass
```python
keras.regularizers.Regularizer
keras.constraints.Constraint
keras.initializers.Initializer
keras.layers.Layer
```
**Note that must implement the call() method for losses,layers(including activation) and models, implement the __call__() method for regularizers, initializers, and constraints.**  
```python
class L1Regularizer(keras.regularizers.Regularizer):
    def __init__(self,factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}
```
## Custom Metrics
Sometimes you need a "streaming" metric which keep track of some statistics or parameters and update correspondingly after each batch. Create a subclass of the keras.metrics.Metric for this kind of task.  
```python
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def huber_fn(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```
Another implementation with sample weights:
```python
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn
    
class HuberMetric(keras.metrics.Mean):
    def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        super().__init__(name=name, dtype=dtype)
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        super(HuberMetric, self).update_state(metric, sample_weight)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```
Here in the update_state method, super(HuberMetric,self).update_state calls the update_state method from the immediate parent class of HuberMetric which is the keras.metrics.Metric.  
## Custom Layers
A simple tool is the keras.layers.Lambda layer. E.g.
```python
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
```
For a custom layer with weights, you need to create a subclass of the keras.layers.Layer class.
```python
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape) # must be at the end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
```
The init method define some setup for the layer, the build method actually construct the layer by explicitly define the kernel and bias shape and then initialize them. The call method does the actual computation and return the result of the layer.  
**Notice that the super().build method must be called in the end.**  
For layers with multiple inputs, the call() method receives tuple containing all the inputs and returns the list of output. The compute_output_shape() method receives a tuple containing each input's batch shape and returns the list of batch output shapes.
```python
class MyMultiLayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return X1 + X2, X1 * X2

    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape2]
```
For layers behave different during training and testing, must add a "training" argument to the call() method. e.g.
```python
class AddGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
```
## Losses and Metrics based on model internals
So far the custom losses and metrics were all based on the labels and the predictions. Sometimes you want to define losses based on other parts of the model, such as weights or activations of its hidden layers. e.g.
```python
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="selu",
                                          kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
```
The extra dense layer used to reconstruct the inputs must be created here because its number of units must be equal to the number of inputs, and this number is unknown before the build() method is called.  
The add_loss() method adds the extra reconstruction loss to the model's list of losses. Two different losses in total for this example.
## Custom Training Loops
