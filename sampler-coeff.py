import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import math
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

def sampler(model_name, u_hat, noise=np.random.normal(0, 1, (1, 128)), batch_size=100):
    model = load_model(model_name)
    samples = np.shape(noise)[0]
    best_samples = []
    best_scores = []
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    cnt = 0
    for i in range(samples//batch_size+1):
        # noise = np.random.normal(0, 1, (1, 128))
        min_score = math.inf
        curr_sample = gen_imgs[0, :,:,0]
        for j in range(batch_size):
            if cnt >= samples:
                break
            fake_image = gen_imgs[cnt, :,:,0]*252+63
            cnt+=1
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
            k.vector()[v2d[:]] = (fake_image).flatten()
            a = k*dot(grad(u), grad(v))*dx
            L = f*v*dx
            u = Function(V)
            solve(a == L, u, bcs=[bc1, bc2])
            diff = project(u-u_hat, V)
            ret = assemble(diff*diff*dx)
            if ret < min_score:
                min_score = ret
                curr_sample = fake_image
        best_samples.append(curr_sample)
        best_scores.append(min_score)
  #       print("Diff, f, u, u_hat: ")
  #       plt.figure()
  #       plot(diff)
  #       plt.figure()
  #       plot(f)
  #       plt.figure()
  #       plot(u)
  #       plt.figure()
  #       plot(u_hat)
  #       plt.show()
    np.save('samples.npy', best_samples)
    np.save('scores.npy', best_scores)
    np.save('zs.npy', noise)
#     np.save('/content/gdrive/My Drive/samples2.npy', best_samples)
#     np.save('/content/gdrive/My Drive/scores2.npy', best_scores)
    return best_samples, best_scores


mesh = UnitSquareMesh(63, 63)
V = FunctionSpace(mesh, 'P', 1)
u_D1 = Expression('0', degree=1)
u_D2 = Expression('1', degree=1)
bc1 = DirichletBC(V, u_D1, boundary1)
bc2 = DirichletBC(V, u_D2, boundary2)
u_hat = TrialFunction(V)
v = TestFunction(V)
k = Constant(1)
k = interpolate(k, V)
f = Constant(0)
f = interpolate(f, V)
v2d = vertex_to_dof_map(V)
k.vector()[v2d[:]] = ((disc(-1, 64, 10))).flatten()
a = k*dot(grad(u_hat), grad(v))*dx
L = f*v*dx
u_hat = Function(V)
solve(a == L, u_hat, bcs=[bc1, bc2])
#noise = u_hat.vector()[:]*0.10
noise = 0.1 * np.random.normal(0, 1, np.shape(u_hat))
u_hat.vector()[:] = u_hat.vector()[:] + noise

import time
np.random.seed(int(time.time()))
best_samples, best_scores = sampler('wgangp_resnet.h5', u_hat, np.random.normal(0, 1, (99999, 128)), 1) #samples=99999 in this case

l = np.argsort(np.asarray(best_scores))
for i in range(0,100):      #how many best samples to plot?
    plt.figure()
    print("Loss = ", best_scores[l[i]])
    plt.imshow(best_samples[l[i]], cmap='gray')
