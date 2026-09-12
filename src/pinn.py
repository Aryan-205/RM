"""
Stage 13 -- A Physics-Informed Neural Network, written from scratch in NumPy.

WHY WRITE IT BY HAND?
=====================
Everywhere else in this project scikit-learn was the right tool, because we
were doing standard supervised regression. A PINN is not standard supervised
regression: its loss contains DERIVATIVES OF THE MODEL'S OWN OUTPUT with
respect to its own input. No scikit-learn estimator exposes that, and pulling
in PyTorch would hide the one mechanism the stage exists to explain. So the
network -- forward pass, backward pass, Adam -- is about 120 lines of NumPy
below, and every gradient in it can be checked by hand.

THE IDEA
========
An ordinary network is trained to satisfy one requirement:

    L_data  =  mean( (prediction - measurement)^2 )

which says "agree with the data we have". A physics-informed network adds a
second requirement that needs no data at all:

    L_physics  =  mean( residual of the governing equation )^2

evaluated at COLLOCATION POINTS -- arbitrary inputs where we have no
measurement, but where we nevertheless know the answer must obey the physics.
For projectile motion the governing equations are simply Newton's second law
with a constant downward force:

    d^2 x / dt^2 = 0                                                 (13.1)
    d^2 y / dt^2 = -g                                                (13.2)

and the initial conditions that pick out one trajectory from the family:

    x(0) = 0,   y(0) = y0,   x'(0) = v0 cos(th),   y'(0) = v0 sin(th)   (13.3)

Together (13.1)-(13.3) determine the solution uniquely. So a PINN can in
principle be trained with ZERO labelled examples -- and one of the experiments
in Stage 13 does exactly that, which is the cleanest possible demonstration of
what "physics-informed" buys you.

HOW WE GET THE DERIVATIVES
==========================
Modern PINN implementations use automatic differentiation to obtain
d^2(output)/d(input)^2, which requires differentiating twice through the
network -- awkward to write by hand. We use a central finite difference in t
instead:

    d^2 f / dt^2  ~=  ( f(t+h) - 2 f(t) + f(t-h) ) / h^2              (13.4)

This is not a compromise in accuracy that matters here: the truncation error
of (13.4) is O(h^2) ~ 1e-4 with h = 1e-2 s, thousands of times smaller than
the model error we are trying to drive down. Its real advantage is that the
right-hand side involves only ordinary FORWARD passes, so the gradient with
respect to the weights is obtained by ordinary backpropagation through three
evaluations of the same network. Everything stays first-order and verifiable.

The trade-off is worth stating in the report: finite differences are cheap and
transparent but scale badly to many input dimensions (one extra pair of
forward passes per derivative), whereas automatic differentiation costs the
same regardless. For a single derivative direction -- time -- finite
differences are strictly the simpler choice.
"""

import numpy as np

G = 9.81


# -------------------- THE NETWORK --------------------

class MLP:
    """
    A plain multilayer perceptron with tanh hidden layers and a linear output.

    tanh rather than ReLU on purpose. A ReLU network is piecewise linear, so
    its second derivative is zero almost everywhere -- it is structurally
    incapable of satisfying an equation like (13.2), which demands a specific
    non-zero curvature. tanh is smooth and infinitely differentiable, which is
    what a physics residual needs.

    Weights use Xavier/Glorot initialisation: variance 1/fan_in, which keeps
    the activations inside tanh's responsive region instead of saturating
    them at +-1 where the gradient vanishes.
    """

    def __init__(self, layer_sizes, seed=0, output_scale=1.0):
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.output_scale = output_scale

        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(rng.normal(0, np.sqrt(1.0 / fan_in), (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    # ---- forward

    def forward(self, Z, cache=None):
        """
        Z : (N, n_inputs), already standardised.
        Returns (N, n_outputs), multiplied by output_scale.

        If `cache` is a list it is filled with the pre- and post-activations
        needed by the backward pass.
        """
        activation = Z
        n_layers = len(self.weights)

        for i in range(n_layers):
            pre = activation @ self.weights[i] + self.biases[i]
            if cache is not None:
                cache.append((activation, pre))
            # linear on the final layer, tanh everywhere else
            activation = pre if i == n_layers - 1 else np.tanh(pre)

        return activation * self.output_scale

    # ---- backward

    def backward(self, cache, grad_output):
        """
        Backpropagate `grad_output` = dL/d(network output, after scaling).

        Returns (grad_weights, grad_biases). Standard chain rule:
          through the linear output layer : dL/dpre = dL/dout * output_scale
          through a tanh layer            : dL/dpre = dL/dact * (1 - tanh^2)
          for each layer                  : dW = act^T @ dpre,  db = sum(dpre)
        """
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]

        delta = grad_output * self.output_scale

        for i in reversed(range(len(self.weights))):
            activation, pre = cache[i]
            if i != len(self.weights) - 1:
                delta = delta * (1.0 - np.tanh(pre) ** 2)

            grad_weights[i] = activation.T @ delta
            grad_biases[i] = delta.sum(axis=0)
            delta = delta @ self.weights[i].T

        return grad_weights, grad_biases

    # ---- parameter access

    def parameters(self):
        return self.weights + self.biases

    def n_parameters(self):
        return sum(p.size for p in self.parameters())


class Adam:
    """
    Adam, written out because it is three lines of algebra and hiding it would
    undercut the point of implementing the network by hand.

    It keeps a running mean (m) and running mean-square (v) of each gradient,
    corrects both for their initialisation bias, and steps by m / (sqrt(v)+eps).
    The division makes the step size roughly independent of the gradient's
    magnitude, which is what lets one learning rate work for weights in
    different layers whose gradients differ by orders of magnitude.
    """

    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, parameters, gradients):
        self.t += 1
        for i, (p, grad) in enumerate(zip(parameters, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# -------------------- THE PROJECTILE PINN --------------------

class ProjectilePINN:
    """
    Learns (v0, theta, y0, t) -> (x, y) from any combination of
      * labelled data          (the ordinary supervised term)
      * the equations of motion (13.1)-(13.2) at collocation points
      * the initial conditions  (13.3) at collocation points

    Set physics_weight = 0 to obtain a completely ordinary neural network with
    identical architecture and identical training -- which is exactly the
    control we need to attribute any improvement to the physics and not to the
    network being bigger or trained longer.
    """

    def __init__(self, hidden=(32, 32), seed=0, lr=3e-3,
                 data_weight=1.0, physics_weight=1.0, ic_weight=1.0,
                 fd_step=1e-2, output_scale=50.0):
        self.net = MLP([4, *hidden, 2], seed=seed, output_scale=output_scale)
        self.lr = lr
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.ic_weight = ic_weight
        self.h = fd_step
        self.history = []
        self.mean_ = None
        self.scale_ = None
        # Set at fit time from the spread of the targets. Without it the data
        # term is measured in m^2 (typical value ~1e2) while the physics and
        # initial-condition terms are dimensionless and of order 1, so the
        # data term would outweigh the physics by two orders of magnitude and
        # `physics_weight` would not mean what it says. Dividing by the target
        # variance puts all three terms on the same footing, so the weights
        # express a genuine trade-off rather than an accident of units.
        self.target_scale_ = 1.0

    # ---- input standardisation

    def _fit_scaler(self, reference_inputs):
        self.mean_ = reference_inputs.mean(axis=0)
        self.scale_ = reference_inputs.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def _standardise(self, X):
        return (X - self.mean_) / self.scale_

    # ---- the three loss terms

    def _data_loss(self, X, Y):
        """Ordinary supervised term, normalised by the target scale."""
        cache = []
        prediction = self.net.forward(self._standardise(X), cache)
        residual = (prediction - Y) / self.target_scale_
        loss = float(np.mean(residual ** 2))
        grad = (2.0 / residual.size) * residual / self.target_scale_
        return loss, cache, grad

    def _physics_loss(self, X):
        """
        Enforce (13.1) and (13.2) at collocation points using the central
        difference (13.4).

        We evaluate the network at t-h, t and t+h in ONE batch of size 3N, so
        that a single forward and a single backward pass cover all three. The
        residual gradient with respect to each of the three copies follows
        directly from differentiating (13.4):

            d(residual)/d f(t-h) = +1/h^2
            d(residual)/d f(t)   = -2/h^2
            d(residual)/d f(t+h) = +1/h^2
        """
        n = len(X)
        X_minus, X_plus = X.copy(), X.copy()
        X_minus[:, 3] -= self.h
        X_plus[:, 3] += self.h

        batch = np.vstack([X_minus, X, X_plus])
        cache = []
        prediction = self.net.forward(self._standardise(batch), cache)

        f_minus, f_mid, f_plus = prediction[:n], prediction[n:2 * n], prediction[2 * n:]
        second_derivative = (f_plus - 2.0 * f_mid + f_minus) / self.h ** 2

        # target curvature: 0 for x, -g for y
        target = np.zeros_like(second_derivative)
        target[:, 1] = -G

        # normalise by g so the residual is dimensionless and O(1)
        residual = (second_derivative - target) / G
        loss = float(np.mean(residual ** 2))

        common = (2.0 / residual.size) * residual / (G * self.h ** 2)
        grad = np.vstack([common, -2.0 * common, common])
        return loss, cache, grad

    def _initial_condition_loss(self, X):
        """
        Enforce (13.3). The physics residual alone cannot pick out a
        trajectory -- (13.1) and (13.2) are satisfied by every parabola with
        the right curvature, regardless of where it starts or how fast. The
        initial conditions are what make the solution unique, and omitting
        them is the most common way a hand-written PINN silently fails.

        Positions at t = 0 are read directly. Velocities use a central
        difference about t = 0, which needs the network at t = -h; that is
        perfectly well defined even though negative times are unphysical,
        because the network is just a smooth function of its inputs.
        """
        n = len(X)
        v0, theta, y0 = X[:, 0], X[:, 1], X[:, 2]

        X_zero = X.copy()
        X_zero[:, 3] = 0.0
        X_minus, X_plus = X_zero.copy(), X_zero.copy()
        X_minus[:, 3] = -self.h
        X_plus[:, 3] = +self.h

        batch = np.vstack([X_minus, X_zero, X_plus])
        cache = []
        prediction = self.net.forward(self._standardise(batch), cache)
        f_minus, f_zero, f_plus = prediction[:n], prediction[n:2 * n], prediction[2 * n:]

        velocity = (f_plus - f_minus) / (2.0 * self.h)

        position_target = np.column_stack([np.zeros(n), y0])
        velocity_target = np.column_stack([v0 * np.cos(theta), v0 * np.sin(theta)])

        # scale both residuals to O(1): positions by y0's spread, velocities by v0's
        position_scale = max(1.0, float(np.std(y0)) + float(np.mean(np.abs(y0))))
        velocity_scale = max(1.0, float(np.mean(np.abs(v0))))

        position_residual = (f_zero - position_target) / position_scale
        velocity_residual = (velocity - velocity_target) / velocity_scale

        loss = float(np.mean(position_residual ** 2) + np.mean(velocity_residual ** 2))

        grad_position = (2.0 / position_residual.size) * position_residual / position_scale
        grad_velocity = (2.0 / velocity_residual.size) * velocity_residual / velocity_scale
        half = grad_velocity / (2.0 * self.h)

        grad = np.vstack([-half, grad_position, half])
        return loss, cache, grad

    # ---- training

    def fit(self, X_data, Y_data, X_collocation, epochs=4000, batch_size=256,
            verbose_every=None):
        """
        One optimiser step combines all three losses. Their gradients are
        accumulated -- three backward passes, one update -- rather than
        applied one after another, so that the optimiser sees the true
        gradient of the combined objective.
        """
        X_collocation = np.asarray(X_collocation, dtype=float)
        has_data = X_data is not None and len(X_data) > 0
        if has_data:
            X_data = np.asarray(X_data, dtype=float)
            Y_data = np.asarray(Y_data, dtype=float)

        reference = np.vstack([X_collocation, X_data]) if has_data else X_collocation
        self._fit_scaler(reference)
        self.target_scale_ = (
            max(1.0, float(np.std(Y_data))) if has_data else 1.0
        )

        optimiser = Adam(self.net.parameters(), lr=self.lr)
        rng = np.random.default_rng(0)

        for epoch in range(epochs):
            grads = [np.zeros_like(p) for p in self.net.parameters()]
            losses = {"data": 0.0, "physics": 0.0, "ic": 0.0}

            def accumulate(loss_value, cache, grad_output, weight, key):
                if weight == 0.0:
                    return
                gw, gb = self.net.backward(cache, grad_output * weight)
                for i, g in enumerate(gw + gb):
                    grads[i] += g
                losses[key] = loss_value

            if has_data and self.data_weight > 0.0:
                accumulate(*self._data_loss(X_data, Y_data),
                           self.data_weight, "data")

            if self.physics_weight > 0.0:
                index = rng.choice(len(X_collocation),
                                   size=min(batch_size, len(X_collocation)),
                                   replace=False)
                batch = X_collocation[index]
                accumulate(*self._physics_loss(batch), self.physics_weight, "physics")
                accumulate(*self._initial_condition_loss(batch), self.ic_weight, "ic")

            optimiser.step(self.net.parameters(), grads)

            if epoch % 50 == 0 or epoch == epochs - 1:
                self.history.append({"epoch": epoch, **losses})
            if verbose_every and epoch % verbose_every == 0:
                print(f"      epoch {epoch:>5d}  data={losses['data']:.4e}  "
                      f"physics={losses['physics']:.4e}  ic={losses['ic']:.4e}")

        return self

    def predict(self, X):
        return self.net.forward(self._standardise(np.asarray(X, dtype=float)))

    def physics_residual(self, X):
        """
        The magnitude of the violation of (13.1)-(13.2) at arbitrary points,
        in m/s^2. An honest diagnostic to report alongside the position error:
        a model can be close to the data and still describe impossible motion.
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        X_minus, X_plus = X.copy(), X.copy()
        X_minus[:, 3] -= self.h
        X_plus[:, 3] += self.h

        prediction = self.predict(np.vstack([X_minus, X, X_plus]))
        f_minus, f_mid, f_plus = prediction[:n], prediction[n:2 * n], prediction[2 * n:]
        second_derivative = (f_plus - 2.0 * f_mid + f_minus) / self.h ** 2

        target = np.zeros_like(second_derivative)
        target[:, 1] = -G
        return np.abs(second_derivative - target)
