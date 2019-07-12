import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import load_model
import random
import time
from fenics import *

def boundary1(x, on_boundary):
    return on_boundary and not near(x[0], 1)

def boundary2(x, on_boundary):
    return near(x[0], 1) and on_boundary

import random


def disc(s,n,r):
    if(s==-1):
        a = n/2-r-r
        b = n/2-r-r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array = np.zeros((n, n))
        array[:] += 63
        array[mask] += 63
        a = n/2-r-r
        b = n/2+r+r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = n/2+r+r
        b = n/2-r-r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = n/2+r+r
        b = n/2+r+r
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
    else:
        random.seed(s)
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array = np.zeros((n, n))
        array[:] += 63
        array[mask] += 63
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
        a = random.randint(r,n-r)
        b = random.randint(r,n-r)
        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        array[mask] += 63
    return array




def stats(scores_file, samples_file, z_file, noise):
    scores = np.load(scores_file)
    samples = np.load(samples_file)
    zs = np.load(z_file)
    x_hat = 0
    z_hat = 0
    s_hat = 0
    for i in range(len(scores)):
        x_hat += samples[i] * np.exp(-1 * scores[i] / (2*noise*noise))
        z_hat += np.exp(-1 * scores[i] / (2*noise*noise))
        s_hat += samples[i]**2 * np.exp(-1 * scores[i] / (2*noise*noise))
    x_mean = x_hat / z_hat
    s = s_hat / z_hat
    variance = s - x_mean**2

    return x_mean, variance

def gen_plots(model_file, scores_file, samples_file, z_file, noise, n, r):
    model = load_model(model_file)
    for i in range(16):
        a = plt.figure()
        plt.imshow(disc(i, n, r))
        plt.colorbar()
        plt.savefig('plots/talk/real_image_%d.png' % (i+1))

    np.random.seed(int(time.time()))
    noise = np.random.normal(0, 1, (16, 128))
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    for i in range(16):
        fake_image = gen_imgs[i, :,:,0]*252+63
        b = plt.figure()
        plt.imshow(fake_image)
        plt.colorbar()
        plt.savefig('plots/talk/fake_image_%d.png' % (i+1))

    x = plt.figure()
    plt.imshow(disc(-1, n, r))
    plt.colorbar()
    plt.savefig('plots/talk/x_ref.png')

    mesh = UnitSquareMesh(63, 63)
    V = FunctionSpace(mesh, 'P', 1)
    u_D1 = Expression('0', degree=1)
    u_D2 = Expression('1', degree=1)
    bc1 = DirichletBC(V, u_D1, boundary1)
    bc2 = DirichletBC(V, u_D2, boundary2)
    u = TrialFunction(V)
    v = TestFunction(V)
    k = Constant(1)
    k = interpolate(k, V)
    f = Constant(0)
    f = interpolate(f, V)
    v2d = vertex_to_dof_map(V)
    k.vector()[v2d[:]] = disc(-1, n, r).flatten()
    a = k*dot(grad(u), grad(v))*dx
    L = f*v*dx
    u = Function(V)
    solve(a == L, u, bcs=[bc1, bc2])

    uu = plt.figure()
    uu = plot(u)
    plt.colorbar(uu)
    plt.savefig('plots/talk/u_no_noise.png')

    #rnoise = u.vector()[:]*0.10
    rnoise = np.random.normal(0, 1, np.shape(u))
    u.vector()[:] = u.vector()[:] + rnoise
    uu = plt.figure()
    uu = plot(u)
    plt.colorbar(uu)
    plt.savefig('plots/talk/u_noise.png')

#    mean, variance = stats(scores_file, samples_file, z_file, 0.1*np.reshape(u.vector()[:], (64, 64)))
    mean, variance = stats(scores_file, samples_file, z_file, 0.1)
    m = plt.figure()
    plt.imshow(mean)
    plt.colorbar()
    plt.savefig('plots/talk/mean.png')

    v = plt.figure()
    plt.imshow(variance)
    plt.colorbar()
    plt.savefig('plots/talk/variance.png')

    scores = np.load(scores_file)
    samples = np.load(samples_file)

    l = np.argsort(scores)
    for i in range(16):
        xx = plt.figure()
        plt.imshow(samples[l[i]])
        plt.colorbar()
        plt.savefig('plots/talk/x_map_%d.png' % i)
        mesh = UnitSquareMesh(63, 63)
        V = FunctionSpace(mesh, 'P', 1)
        u_D1 = Expression('0', degree=1)
        u_D2 = Expression('1', degree=1)
        bc1 = DirichletBC(V, u_D1, boundary1)
        bc2 = DirichletBC(V, u_D2, boundary2)
        u = TrialFunction(V)
        v = TestFunction(V)
        k = Constant(1)
        k = interpolate(k, V)
        f = Constant(0)
        f = interpolate(f, V)
        v2d = vertex_to_dof_map(V)
        k.vector()[v2d[:]] = samples[l[i]].flatten()
        a = k*dot(grad(u), grad(v))*dx
        L = f*v*dx
        u = Function(V)
        solve(a == L, u, bcs=[bc1, bc2])

gen_plots('wgangp_resnet.h5', 'scores.npy', 'samples.npy', 'zs.npy', 0.1, 64, 10)

