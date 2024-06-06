import numpy as np
import tensorflow as tf


class ode_cell(tf.compat.v1.nn.rnn_cell.BasicRNNCell):

    @property
    def state_size(self):
        return self._num_units, self._num_units

    def zero_state(self, batch_size, dtype):
        x_0 = tf.zeros([batch_size, self._num_units], dtype=dtype)
        v_0 = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return x_0, v_0


class pendulum_cell(ode_cell):
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.3), trainable=False)
        self.length = self.add_variable("length", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.g = self.add_variable("g", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.built = True

    def call(self, poss, vels):
        for i in range(10):
            acc = -self.g * tf.sin(poss)
            vels = vels + self.dt / 10 * acc / self.length  # calculate the angular velocity
            poss = poss + self.dt / 10 * vels
        return poss, vels


class pendulum_scale_cell(ode_cell):
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.3), trainable=False)
        self.length = self.add_variable("length", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.g = self.add_variable("g", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.f = self.add_variable("focal", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.c = self.add_variable("np_dist", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.r = self.add_variable("radius", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.built = True

    def call(self, poss, vels):
        for i in range(10):
            acc = -self.g * tf.sin(poss)
            vels = vels + self.dt / 10 * acc / self.length  # calculate the angular velocity
            poss = poss + self.dt / 10 * vels
        return poss, vels

    def get_projection(self, pos):
        # hacky (but working) way of initializing the cell before any rollouts happen
        # needed for enc_pos in conv_feedforward
        if not self.built:
            self(pos, pos)
        d = (tf.exp(self.length) + tf.exp(self.r)) * tf.math.sin(pos[:, 0])
        c_max = tf.maximum(tf.exp(self.c), d + tf.exp(self.r) + 0.1)  # ensure the projection can still work
        # notice the lack of multiplying by r below
        # this is because sigma is a scale parameter, i.e. by how much should the radius increase
        sigma = tf.exp(self.f) / ((c_max - d) ** 2 - tf.exp(self.r) ** 2) ** 0.5
        return sigma


class pendulum_intensity_cell(ode_cell):
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.3), trainable=False)
        self.length = self.add_variable("length", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.g = self.add_variable("g", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.c = self.add_variable("np_dist", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.built = True

    def call(self, poss, vels):
        for i in range(10):
            acc = -self.g * tf.sin(poss)
            vels = vels + self.dt / 10 * acc / self.length  # calculate the angular velocity
            poss = poss + self.dt / 10 * vels
        return poss, vels

    def get_intensity(self, pos):
        # hacky (but working) way of initializing the cell before any rollouts happen
        # needed for enc_pos in conv_feedforward
        if not self.built:
            self(pos, pos)
        d = tf.exp(self.length) * tf.math.sin(pos[:, 0])
        intensity = (tf.exp(self.c) - self.length) ** 2 / (tf.exp(self.c) - d) ** 2
        return intensity


class bouncing_ball_drop_cell(ode_cell):

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.3), trainable=False)
        self.g = self.add_variable("g", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        # the model has no way of backpropagating through the radius, so set it to the correct value
        self.r = self.add_variable("r", shape=[], initializer=tf.constant_initializer(3.0), trainable=False)
        self.elasticity = self.add_variable("elasticity", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.built = True

    def call(self, poss, vels):
        x, y = poss[:, 0], poss[:, 1]
        vx, vy = vels[:, 0], vels[:, 1]
        for _ in range(10):
            acc = tf.exp(self.g)
            vy = vy + self.dt / 10 * acc
            y = y + self.dt / 10 * vy
            x = x + self.dt / 10 * vx

            vy = tf.where(tf.greater(y, 32 - self.r), -vy * self.elasticity, vy)
        y = tf.clip_by_value(y, self.r, 32 - self.r)
        vels = tf.stack([vx, vy], axis=1)
        poss = tf.stack([x, y], axis=1)
        return poss, vels

class ball_throw_cell(ode_cell):

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.05), trainable=False)
        self.g = self.add_variable("g", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.r = self.add_variable("r", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.built = True

    def call(self, poss, vels):
        x, y = poss[:, 0], poss[:, 1]
        vx, vy = vels[:, 0], vels[:, 1]
        for _ in range(10):
            acc = tf.exp(self.g)
            vy = vy + self.dt / 10 * acc
            y = y + self.dt / 10 * vy
            x = x + self.dt / 10 * vx

            # indices = tf.where(y <= self.r)
            # vy = tf.tensor_scatter_nd_update(vy, indices, tf.tile(tf.convert_to_tensor([0.0]), [tf.shape(indices)[0]]))
            # vx = tf.tensor_scatter_nd_update(vx, indices, tf.tile(tf.convert_to_tensor([0.0]), [tf.shape(indices)[0]]))
            # y = tf.tensor_scatter_nd_update(y, indices, tf.tile(tf.convert_to_tensor([self.r]), [tf.shape(indices)[0]]))
        # 32 - y since (0, 0) is the top left corner
        # y = 32 - y
        # x = tf.clip_by_value(x, self.r, 32 - self.r)
        # y = tf.clip_by_value(y, self.r, 32 - self.r)
        vels = tf.stack([vx, vy], axis=1)
        poss = tf.stack([x, y], axis=1)
        return poss, vels

class sliding_block_cell(ode_cell):
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.3), trainable=False)
        self.g = self.add_variable("g", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
        self.inclination = self.add_variable("inclination", shape=[], initializer=tf.constant_initializer(0.0), trainable=True)
        self.friction = self.add_variable("friction", shape=[], initializer=tf.constant_initializer(0.0), trainable=True)
        self.built = True

    def call(self, poss, vels):
        px, py = poss[:, 0], poss[:, 1]
        vx, vy = vels[:, 0], vels[:, 1]
        for _ in range(10):
            acc = tf.exp(self.g) * (tf.math.sin(self.inclination) - self.friction * tf.math.cos(self.inclination))
            vx = vx + self.dt / 10 * acc
            px = px + self.dt / 10 * vx
            py = py + self.dt / 10 * vy
        poss = tf.stack([px, py], axis=1)
        vels = tf.stack([vx, vy], axis=1)
        return poss, vels

class bouncing_ode_cell(ode_cell):
    """ Assumes there are 2 objects """

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.3), trainable=False)
        self.built = True

    def call(self, poss, vels):
        poss = tf.split(poss, 2, 1)
        vels = tf.split(vels, 2, 1)
        for i in range(5):
            poss[0] = poss[0] + self.dt/5*vels[0]
            poss[1] = poss[1] + self.dt/5*vels[1]

            for j in range(2):
                # Compute wall collisions. Image boundaries are hard-coded.
                vels[j] = tf.where(tf.greater(poss[j]+2, 32), -vels[j], vels[j])
                vels[j] = tf.where(tf.greater(0.0, poss[j]-2), -vels[j], vels[j])
                poss[j] = tf.where(tf.greater(poss[j]+2, 32), 32-(poss[j]+2-32)-2, poss[j])  
                poss[j] = tf.where(tf.greater(0.0, poss[j]-2), -(poss[j]-2)+2, poss[j]) 

        poss = tf.concat(poss, axis=1)
        vels = tf.concat(vels, axis=1)
        return poss, vels


class spring_ode_cell(ode_cell):
    """ Assumes there are 2 objects """

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.3), trainable=False)
        self.k = self.add_variable("log_k", shape=[], initializer=tf.constant_initializer(np.log(1.0)), trainable=True)
        self.equil = self.add_variable("log_l", shape=[], initializer=tf.constant_initializer(np.log(1.0)), trainable=True)
        self.built = True

    def call(self, poss, vels):
        poss = tf.split(poss, 2, 1)
        vels = tf.split(vels, 2, 1)
        for i in range(5):
            norm = tf.sqrt(tf.abs(tf.reduce_sum(tf.square(poss[0]-poss[1]), axis=-1, keepdims=True)))
            direction = (poss[0]-poss[1])/(norm+1e-4)
            F = tf.exp(self.k)*(norm-2*tf.exp(self.equil))*direction
            vels[0] = vels[0] - self.dt/5*F
            vels[1] = vels[1] + self.dt/5*F

            poss[0] = poss[0] + self.dt/5*vels[0]
            poss[1] = poss[1] + self.dt/5*vels[1]
 
        poss = tf.concat(poss, axis=1)
        vels = tf.concat(vels, axis=1)
        return poss, vels


class gravity_ode_cell(ode_cell):
    """ Assumes there are 3 objects """

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        assert h_depth == input_depth

        self.dt = self.add_variable("dt_x", shape=[], initializer=tf.constant_initializer(0.5), trainable=False)
        self.g = self.add_variable("log_g", shape=[], initializer=tf.constant_initializer(np.log(1.0)), trainable=True)
        self.m = self.add_variable("log_m", shape=[], initializer=tf.constant_initializer(np.log(1.0)), trainable=False)
        self.A = tf.exp(self.g)*tf.exp(2*self.m)
        self.built = True

    def call(self, poss, vels):
        for i in range(5):
            vecs = [poss[:,0:2]-poss[:,2:4], poss[:,2:4]-poss[:,4:6], poss[:,4:6]-poss[:,0:2]]
            norms = [tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(vec), axis=-1, keepdims=True), 1e-1, 1e5)) for vec in vecs]
            F = [vec/tf.pow(tf.clip_by_value(norm, 1, 170), 3) for vec, norm in zip(vecs, norms)]
            F = [F[0]-F[2], F[1]-F[0], F[2]-F[1]]
            F = [-self.A*f for f in F]
            F = tf.concat(F, axis=1)
            vels = vels + self.dt/5*F
            poss = poss + self.dt/5*vels
 
        poss = tf.concat(poss, axis=1)
        vels = tf.concat(vels, axis=1)
        return poss, vels
