from jax.config import config
config.update("jax_enable_x64", True)

from jax import jit, device_put, lax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.autonotebook import tqdm
from time import time
from jax.lib import xla_bridge
from dataset import ArenaDataset, SphereDataset

def main():
    print(xla_bridge.get_backend().platform)
    # resume('epoch_400.npy')
    train()

def resume(param_fn, reg=[1e-10, 0.0039, 1500*1e-10], epochs=1900, max_grad_norm=10):
    arena = iter(SphereDataset(batch_size=1000, steps=450))
    parameters = jnp.load(param_fn).astype(jnp.float64)
    
    regularization = reg
    delta = jnp.zeros_like(parameters)

    pbar = tqdm(range(1, epochs))

    # Debug overfit
    batch = next(arena)
    for epoch in pbar:
        # Print current epoch
        tqdm.write("\nEpoch: {}".format(epoch))

        # Save weights every 100 epochs
        if epoch % 10 == 0:
            jnp.save('epoch_{}.npy'.format(epoch), parameters)

        # Generate the training data for this epoch
        # batch = next(arena)

        # Run step
        parameters, regularization, delta = step(
            batch, parameters, regularization, delta, epoch
        )


def train(epochs=1900, max_grad_norm=10):
    arena = iter(ArenaDataset(batch_size=500, steps=100))
    parameters = init_params()
    
    regularization = init_regularization()
    delta = jnp.zeros_like(parameters)

    pbar = tqdm(range(1, epochs))
    batch = next(arena)
    for epoch in pbar:
        # Print current epoch
        tqdm.write("\nEpoch: {}".format(epoch))

        # Save weights every 100 epochs
        if epoch % 20 == 0:
            jnp.save('epoch_{}.npy'.format(epoch), parameters)

        # Generate the training data for this epoch
        batch = next(arena)

        # Run step
        parameters, regularization, delta = step(
            batch, parameters, regularization, delta, epoch
        )


def step(batch, parameters, regularization, delta, epoch, max_grad_norm=10):
    # Get inputs and targets from batch
    x, y = batch
    x = jnp.array(x)
    y = jnp.array(y)

    # Compute the prediction and error
    loss_mse, loss_l2, loss_h = loss_function(x, y, parameters, regularization)

    # Backpropagate the error to calculate the gradients
    loss = loss_mse + loss_l2 + loss_h
    tqdm.write('loss_l2, loss_h: {:.3f}, {:.3f}'.format(loss_l2.item(), loss_h.item()))
    # pbar.set_postfix({'loss': float(loss),
    #                   'mse': float(loss_mse)})

    # Get the gradients
    grad = div(x, y, parameters, regularization)

    # Update the regularization
    regularization = update_regularization(loss_mse, loss_l2, loss_h, regularization)

    # Rescale gradients if they are too large
    grad_norm = jnp.sqrt(jnp.sum(grad**2))

    if grad_norm > max_grad_norm:
        grad = max_grad_norm / grad_norm * grad
        grad_norm = jnp.sqrt(jnp.sum(grad**2))

    # Minimize the positive-definite quadratic approximation of the error
    delta_norm = jnp.sqrt(jnp.sum(delta**2))
    # if epoch % 10 == 0:
    #     delta = (-2*delta_norm/grad_norm) * grad
    #     delta_norm = jnp.sqrt(jnp.sum(delta**2))
    # else:
    #     delta = 0.95 * delta
    delta = 0.95 * delta

    delta_list, _ = conjugate_gradient(
        A=lambda v: G(v, x, parameters, regularization),
        b=-grad, x0=delta, max_iter=50)

    # CG-backtracking
    delta, loss_now = backtracking(parameters, delta_list, x, y, regularization)

    # Compute the reduction ratio rho
    rho = reduction_ratio(parameters, delta, x, y, grad, regularization)
    if rho > -float('inf'):
        tqdm.write('rho: {}'.format(rho.item()))
    else:
        tqdm.write('rho: -inf')

    # Backtracking line search to find the learning rate
    rate, reject_step = line_search(parameters, x, y, delta, grad, regularization)
    tqdm.write('rate: {}'.format(rate))

    # Update the dampening parameter
    regularization = update_dampening(rho, regularization)
    tqdm.write('l_l2, l_h, l_sd: {:.4f}, {:.4f}, {:.4f}'.format(*regularization))

    # Update the parameters
    if not reject_step:
        parameters = parameters + rate * delta

    # Show improvement
    tqdm.write("Initial loss, new loss: {:.4f}, {:.4f}".format(
        loss, sum(loss_function(x, y, parameters, regularization))
    ))

    return parameters, regularization, delta


def init_params(d=100, d_in=2, d_out=2, d_incoming=15):
    rng = np.random.RandomState(seed=1)
    W_hx = rng.randn(d_in, d) / np.sqrt(d_in)
    W_hx = jnp.array(W_hx)

    W_hh = rng.randn(d, d)
    u, s, v = np.linalg.svd(W_hh, full_matrices=False)
    W_hh = u.dot(v)
    W_hh = np.array(W_hh)
    W_hh = jnp.array(W_hh)

    W_yh = np.zeros((d, d_out))
    W_yh = jnp.array(W_yh)

    b_h = np.zeros(d)
    b_h = jnp.array(b_h)

    return np.concatenate((
        W_hx.reshape(-1),
        W_hh.reshape(-1),
        W_yh.reshape(-1),
        b_h.reshape(-1),
    ))


def init_regularization(l_l2=0.5, mu=1500, l_sd=0.5*1500, l_h=0.1):
    return [l_l2, l_h, l_l2 * mu]


def loss_function(x, y, parameters, regularization):
    l_l2, l_h, l_sd = regularization
    output = predict(x, parameters)

    y_hat = output[:, :, :2]
    h = output[:, :, 2:102]
    h_lin = output[:, :, 102:]

    W_hx = parameters[:200]
    W_yh = parameters[10200:10400]

    loss_h = l_h * jnp.mean(h**2) * 0.5
    loss_l2 = 0.5 * l_l2 * jnp.mean(jnp.concatenate((W_hx, W_yh))**2)

    loss_mse = jnp.mean(0.5*jnp.square(y_hat - y))
    # loss_mse = jnp.mean(jnp.square(y_hat - y))

    return loss_mse, loss_l2, loss_h


@jit
def predict(x, params, dt=1, tau=10, steps=450):
    W_hx = params[:200].reshape((2, 100))
    W_hh = params[200:10200].reshape((100, 100))
    W_yh = params[10200:10400].reshape((100, 2))
    b_h = params[10400:10500].reshape(100)

    # x = x.reshape((500, 450, 2))
    x = x.transpose((1, 0, 2))
    h_lin = jnp.zeros((x.shape[1], 100))

    def scan_f(h_lin, xs):
        h_lin = h_lin + dt/tau * (-h_lin + xs @ W_hx + jnp.tanh(h_lin) @ W_hh + b_h)
        h = jnp.tanh(h_lin)
        y_hat = h @ W_yh

        return h_lin, jnp.concatenate((y_hat, h, h_lin), axis=1)

    _, output = lax.scan(scan_f, h_lin, x, length=steps)

    return output.transpose((1, 0, 2))


@jit
def div(x, y, params, reg):
        return jax.grad(lambda params: sum(loss_function(x, y, params, reg)))(params)


@jit
def G(v, x, params, reg, loss='mse'):
    l_l2, l_h, l_sd = reg
    output = predict(x, params)
    batch_sz, steps, _ = x.shape
    
    _, Jv = jax.jvp(lambda params: predict(x, params), (params,), (v,))
    if 'mse' in loss:
        sigmoid = jax.nn.sigmoid(output[:, :, 102:])
        HJv = jnp.concatenate((
            Jv[:, :, :102], Jv[:, :, 102:] * sigmoid * (1 - sigmoid)
        ), axis=2)
    elif 'cross-entropy' in loss:
        softmax = jax.nn.softmax(output[:, :, :10])
        tanh = jnp.tanh(output[:, :, 110:])
        HJv = jnp.concatenate((
            softmax * Jv[:, :, :10] - softmax * np.sum(softmax * Jv[:, :, :10], axis=2, keepdims=True),
            Jv[:, :, 10:110], Jv[:, :, 110:] * (1-tanh**2)
        ), axis=2)
        
    diag = jnp.concatenate((jnp.ones(2) * 2 / (2 * batch_sz * steps), 
                            jnp.ones(100) * l_h / (100 * batch_sz * steps), 
                            jnp.ones(100) * l_sd / (100 * batch_sz * steps)))
    HJv = diag[None, None, :] * HJv
    _, vjp = jax.vjp(lambda params: predict(x, params), params)
    JHJv = vjp(HJv)

    l2_v = l_l2 / (400) * \
        jnp.concatenate((v[:200], jnp.zeros(10000),
                         v[10200:10400], jnp.zeros(100)))

    return JHJv[0] + l2_v


def backtracking(parameters, deltas, x, y, regularization):
    delta = deltas[-1]
    new_parameters = parameters + delta
    error = sum(loss_function(x, y, new_parameters, regularization))

    for d in reversed(deltas[:-1]):
        new_parameters = parameters + d
        error_new = sum(loss_function(x, y, new_parameters, regularization))

        if float(error) < float(error_new):
            break

        delta = d.copy()
        error = error_new

    return delta, error


def line_search(parameters, x, y, delta, grad, regularization):
    reject_step = False
    rate = 1
    c = 1e-2

    loss_old = sum(loss_function(x, y, parameters, regularization))

    new_parameters = parameters + delta
    loss_new = sum(loss_function(x, y, new_parameters, regularization))

    for i in range(60):
        if loss_new <= loss_old + c * rate * delta.dot(grad):
            break
        else:
            rate *= 0.8

        new_parameters = parameters + rate*delta
        loss_new = sum(loss_function(x, y, new_parameters, regularization))

        if i == 59:
            reject_step = True
            break

    return rate, reject_step


def update_dampening(rho, regularization):
    l_l2, l_h, l_sd = regularization

    if rho > 0.75:
        l_l2 *= 2/3
        l_h *= 2/3
    elif rho < 0.25:
        l_l2 *= 3/2
        l_h *= 3/2

    return l_l2, l_h, l_sd


def update_regularization(loss_mse, loss_l2, loss_h, regularization):
    l_l2, l_h, l_sd = regularization

    if loss_l2 > 1/3 * loss_mse:
        l2 = 2/3 * l_l2
    else:
        l2 = l_l2

    if loss_h > 1/3 * loss_mse:
        h = 2/3 * l_h
    else:
        h = l_h

    sd = l2 * 1500

    return [l2, h, sd]


def reduction_ratio(parameters, delta, x, y, grad, regularization):
    loss_old = sum(loss_function(x, y, parameters, regularization))

    new_parameters = parameters + delta
    loss_new = sum(loss_function(x, y, new_parameters, regularization))

    Gv = G(delta, x, parameters, [0, 0, 0])
    loss_quadratic = 0.5 * delta.dot(Gv) + delta.dot(grad)

    rho = (loss_new - loss_old) / loss_quadratic

    return rho if loss_new <= loss_old else -float('inf')


def conjugate_gradient(A, b, x0, M=None, max_iter=50):
    phi = np.zeros(max_iter)
    i_next = 5
    i_mult = 1.3

    i_store = []
    x_store = []

    r = b - A(x0)
    d = r
    delta_new = r.dot(d)
    x = x0

    i = 1
    while i < (max_iter + 1):
        Ad = A(d)
        dAd = d.dot(Ad)

        if dAd < 0:
            raise Exception('Gauss-Newton matrix has negative curvature')

        alpha = delta_new / dAd
        x = x + alpha * d
        r = r - alpha * Ad
        s = r
        delta_old = delta_new.copy()
        delta_new = r.dot(s)

        beta = delta_new / delta_old
        d = s + beta*d

        if i == np.ceil(i_next):
            i_store.append(i)
            x_store.append(x.copy())
            i_next = i_next * i_mult

        k = max(10, np.ceil(0.1 * i))
        phi[i-1] = 0.5 * (-b - r).dot(x)
        # if i > k and phi[i-k-1] < 0 and (phi[i-1] - phi[i-k-1])/phi[i-1] < 1e-4:
        if i > k and phi[i-k-1] < 0 and (phi[i-1] - phi[i-k-1])/phi[i-1] < 1e-4:
            break

        i += 1

    if i != np.ceil(i_next):
        i_store.append(i)
        x_store.append(x.copy())

    return x_store, i_store


if __name__ == '__main__':
    main()
