import math
from numpy import unique, array
import numpy as np

i = complex(0, 1)
e = np.e
pi = np.pi
tau = 2 * np.pi

def instance(cls):
    return cls()

@instance
class RGB:
    """
    class for shorthands
    """
    R = array([255, 0, 0])
    r = 0xff0000
    G = array([0, 255, 0])
    g = 0x00ff00
    B = array([0, 0, 255])
    b = 0x0000ff

    def hex(self, x):
        return str(x)[2:]

def row(*x):
    arr = array(x)
    if len(arr.shape) == 1:
        return arr.reshape(1, -1)
    return arr


def col(*x):
    arr = array(x)
    if len(arr.shape) == 1:
        return arr.reshape(-1, 1)
    return arr

def flatten2d(arr, d = 2):
    """

    :param arr:
    :type arr:
    :param d: final dimen
    :type d:
    :return:
    :rtype:
    """
    return arr.reshape((-1, arr.shape[-1]))

def c2r2(z):
    """

    :param z: array of complex numbers of shape S
    :return: (2 x S) array of points in R^2
    """
    # let shape(z) = S

    # (S) -> (2 x S.T)
    rect = array([z.real, z.imag])

    # (2 x S.T) --> (S x 2) = (S x 2) yay!
    arr = rect#.T
    #print(z, (arr := arr.reshape((2,-1))))
    return arr

def int_(arr):
    return np.vectorize(int)(arr)

def round_(arr):
    return np.vectorize(round)(arr)


def uint(n, a=0, b=1, *, exc=True, inc=True):
    """
    gives n points in [frame, b];
    default n pts in the Unit INTerval.
    returns (1 x n) matrix
    """

    # teeny tiny epsilon diff for drawing
    eps = 0  #0.0001
    r = row(np.unique(np.linspace(a - eps, b + eps, round(n), exc)[not inc:]))
    return r


def unit(n, a=0, b=1, *, exc=True, inc=True):
    """
    n points in the UNIT circle (C)
    """

    return exp2ni(uint(n, a, b, exc=exc, inc=inc))


def exp2ni(t):
    """n kinda looks liek pi.... good enough.

    e^0pi

    :param t:
    :type t:
    :return:
    :rtype:
    """
    return np.exp(tau * i * t)

def udisc(m, n, a=0, b=1, c=0, d=1, *, exc=True, inc=True):
    """

    :param m: radial samples
    :type m:
    :param n: angular samples
    :type n:
    :param a: inner radial bound
    :type a:
    :param b: outer radial bound
    :type b:
    :param c: lower angular bound
    :type c:
    :param d: greater angular bound
    :type d:
    :param exc:
    :type exc:
    :param inc:
    :type inc:
    :return:
    :rtype:
    """
    rad = uint(m, a, b)  # (1 x M)
    ang = unit(n, c, d)  # (1 X N)
    return rad.T @ ang

def c2z2(arr, f=round_):
    """
    the same as unique(int(arr)).reshape((2, -1)) -- to set coords of an array!
    :param: arr
    :type:
    :param: f
    :type: function from R -> Z (default round_; see int_). this is composed with c2r2 to make c2z2 : C -> Z^2
    :return: unique(int(arr))
    :rtype:
    """
    return f(c2r2(arr)).reshape((2, -1))