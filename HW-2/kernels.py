import numpy as np
from scipy.signal import convolve2d

def box_blur_kernel(radius: int) -> np.ndarray:
    """
    Returns a blox blur kernel.
    """
    if radius < 0:
            raise ValueError("radius must be positive")
    
    size = 2*radius + 1 
    return np.ones((size, size), dtype=np.float32) / (size * size)

def gaussian_blur_kernel(radius: int, sigma: float = None) -> np.ndarray:
    """
    Returns a gaussian blur kernel. 
    Standard deviation defaults to a third of the radius.
    """
    if radius < 0:
        raise ValueError("radius must be positive")
    
    if sigma is None:
        sigma = radius / 3
    
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    
    return kernel.astype(np.float32)

def gaussian_derivative_kernels(
    radius: int, 
    sigma: float = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the x and y derivatives of a 2D gaussian.
    Standard deviation defaults to a third of the radius.
    """
    if radius < 0:
        raise ValueError("radius must be positive")
        
    if sigma is None:
        sigma = radius / 3
        
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    
    gaussian = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    gaussian /= gaussian.sum()
    
    dx = -(xx / sigma**2) * gaussian
    dy = -(yy / sigma**2) * gaussian
    
    return dx.astype(np.float32), dy.astype(np.float32)        

def sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the horizontal and vertical Sobel kernels.
    """    
    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    return kx, ky

def prewitt_kernels() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the horizontal and vertical Prewitt kernels.
    """    
    kx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)
    
    return kx, ky

def roberts_kernels() -> tuple[np.ndarray, np.ndarray]:
    """
    Return the two Roberts kernels.
    """
    kx = np.array([
        [ 0, 1],
        [-1, 0]
    ], dtype=np.float32)

    ky = np.array([
        [1,  0],
        [0, -1]
    ], dtype=np.float32)

    return kx, ky

def gb_kernels(radius, epsilon=1.0, sigma=None):
    """
    Construct the two Generalized Boundaries (Gb) filters.
    """
    if radius < 1:
        raise ValueError("radius must be at least 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if sigma is None:
        sigma = radius / 2

    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coords, coords)

    # Gaussian weights g.
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    g /= g.sum()
    
    # Projection pi_epsilon onto a disk of radius epsilon.
    distance = np.sqrt(x**2 + y**2)
    scale = np.minimum(1.0, epsilon / np.maximum(distance, 1e-12))

    projected_x = x * scale
    projected_y = y * scale

    alpha = np.sum((g * projected_x)**2)
    
    hx = g**2 * projected_x / alpha
    hy = g**2 * projected_y / alpha

    return hx, hy

def generalized_boundaries(layers, radius, epsilon=1.0, sigma=None):
    layers = np.asarray(layers, dtype=np.float64)

    if layers.ndim == 2:
        layers = layers[..., None]
        
    if layers.ndim != 3:
        raise ValueError("layers must have shape (H, W) or (H, W, K)")

    hx, hy = gb_kernels(radius, epsilon, sigma)

    height, width, n_layers = layers.shape

    # Jx and Jy store the two filter responses for each layer.
    jx = np.empty((height, width, n_layers), dtype=np.float64)
    jy = np.empty_like(jx)
    
    for k in range(n_layers):
        jx[..., k] = convolve2d(layers[..., k], hx, mode="same", boundary="fill")
        jy[..., k] = convolve2d(layers[..., k], hy, mode="same", boundary="fill")

    # M = J J^T, computed at every pixel.
    mxx = np.sum(jx * jx, axis=2)
    myy = np.sum(jy * jy, axis=2)
    mxy = np.sum(jx * jy, axis=2)

    # Eigenvalues of the symmetric 2x2 matrix M.
    discriminant = np.sqrt(np.maximum((mxx - myy)**2 + 4 * mxy**2, 0))
    eigenvalue = np.maximum(0, 0.5 * (mxx + myy + discriminant))

    strength = np.sqrt(eigenvalue)

    return strength
