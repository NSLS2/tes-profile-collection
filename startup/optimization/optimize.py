import scipy as sp
import numpy as np

from numpy import linalg as la
from skimage import measure

import ophyd

def get_concave_area(contour):

    total_area = 0
    tri = sp.spatial.qhull.Delaunay(contour)
    for pts in contour[tri.simplices]:

        p0 = pts.mean(axis=0)
        if not cv2.pointPolygonTest(contour.astype(np.float32).reshape(-1,1,2), p0, measureDist=False) > 0: continue
        total_area += .5 * la.det(np.c_[pts[:,0],pts[:,1],np.ones(3)])

    return total_area

def get_beam_stats(im, thresh=np.exp(-2), area_method='convex'):

    nx, ny = im.shape
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    from skimage import measure
    contours = measure.find_contours(im, thresh * im.max())
    contour  = contours[np.argmax([len(_) for _ in contours])]
    cx, cy   = contour.mean(axis=0)
    wx, wy   = contour.ptp(axis=0)
    peak     = im.max()
    hull     = sp.spatial.ConvexHull(contour)
    if area_method == 'concave':
        area = get_concave_area(contour)
    else:
        area = hull.volume

    I, J = np.meshgrid(np.arange(int(np.maximum(cx-0.5*wx,0)),
                                 int(np.minimum(cx+0.5*wx,nx-1))),
                       np.arange(int(np.maximum(cy-0.5*wy,0)),
                                 int(np.minimum(cy+0.5*wy,ny-1))), indexing='ij')

    in_contour = True
    for eq in hull.equations:
        in_contour &= (eq[:-1][:,None,None]*np.r_[I[None],J[None]]).sum(axis=0) + eq[-1] <= 1e-12
    flux = im[I,J][in_contour].sum()

    return cx, cy, wx, wy, peak, flux, area, contour


def get_some_beam_stats(im):

    nx, ny = im.shape

    W = im - im.min()

    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    ci = np.sum(W * I) / np.sum(W)
    cj = np.sum(W * J) / np.sum(W)

    wi = np.sqrt(np.sum(W * np.square(I - ci)) / np.sum(W))
    wj = np.sqrt(np.sum(W * np.square(J - cj)) / np.sum(W))

    return ci, cj, wi, wj



def get_image(_positions):

    # Move the motors to these positions

    # for motor, position in zip(motors, _positions):
    #     motor.move(position)

    RE(bps.mv(*[_ for motor, position in zip(motors, _positions) for _ in [motor, position]]))


    # Collect the data
    # image = ...\

    uid, = RE(bp.count([vstream] + motors))
    hdr  = db[uid]

    image  = sp.ndimage.gaussian_filter(np.array(list(hdr.data("vstream_image")))[0], sigma=1)
    image -= image.min()

    return image

def get_loss(_positions):

    image = get_image(_positions)

    nx, ny = image.shape

    cx, cy, wx, wy, peak, flux, area, contour = get_beam_stats(image, thresh=1/np.e, area_method='convex')

    #cx, cy, wx, wy = get_some_beam_stats(image)

    loss = (np.abs(wx) + np.abs(wy)) / flux # + 1e-1 * (np.abs(cx - nx/2) + np.abs(cy - ny/2))

    stat_string = ' | '.join([f'{lab} = {val:.01f}' for lab, val in zip(['cx','cy','wx','wy'], [cx, cy, wx, wy])])

    print(f'{loss = :.01e} | {stat_string}')

    return loss



def get_fitness(image):

    # Draw a contour, evaluate beam position and shape


    return fitness



motors = [kbh.dsh, kbh.ush, kbv.dsh, kbv.ush]

#bounds = [[-0.15, 0.15], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.25]]

rel_bounds = np.array([[-0.10, 0.10], [-0.10, 0.10], [-0.10, 0.10], [-0.10, 0.10]]) + 1e-2 * np.random.standard_normal(size=4)[:,None]

start_positions = np.array([3.33999983, 2.29500126, 4.27499621, 2.4947932 ])#np.array([motor.user_readback.get() for motor in motors])

bounds = start_positions[:,None] + rel_bounds

#print(get_beam_stats(get_image(start_positions, motors))[:-1])

positions = np.array([np.random.uniform(low=b[0],high=b[1]) for b in bounds])

#print(start_positions)

print(bounds)
print(start_positions)
print(positions)
#print(get_loss(start_positions, motors))

print(get_loss(start_positions))
print(get_loss(positions))

#cx, cy, wx, wy, peak, flux, area, contour = get_beam_stats(image)

#plt.imshow(image.T, cmap='gray_r')
#plt.plot(*contour.T, c='r')

#for motor, position in zip(motors, positions):
#    motor.move(position)


# set bounds manually
# get readback from motors, initialize optimizers

def DE_func(iter_mps):

    vals = []
    for p in iter_mps:
        vals.append(get_loss(p))

    return np.array(vals)[:,None]

#print(DE_func(start_positions + np.random.uniform(low=-1e-2,high=1e-2,size=10)[:,None]))



try:

    print('starting optimization...')
    #pass
    #res = sp.optimize.minimize(get_loss, x0=positions, bounds=bounds, options={'maxfun' : 64}, tol=1e-9) # classic
    #res = sp.optimize.basinhopping(get_loss, x0=positions, bounds=bounds, options={'maxfun' : 64}, tol=1e-9)
    #res = sp.optimize.dual_annealing(get_loss, x0=positions, bounds=bounds, options={'maxfun' : 64}, tol=1e-9)
    #res = sp.optimize.differential_evolution(DE_func, x0=start_positions[None,:] + np.random.uniform(low=-1e-2,high=1e-2,size=4)[:,None],
    #bounds=bounds, maxiter=10, popsize=4, tol=0.01)
    res = sp.optimize.shgo(get_loss, bounds=bounds, options={'maxfun' : 4}) # no gradient



    print(res)
    #
except Exception as e:

    print(e)

print(get_loss(positions))
print(get_loss(res.x))
print(get_loss(start_positions))


#sp.optimize.differential_evolution(get_loss, x0=x0, bounds=bounds) # genetic

#sp.optimize.basinhopping(get_loss, x0=x0, bounds=bounds) # good for bumpy GD

## this is probably not the best method

#sp.optimize.direct(get_loss, x0=x0, bounds=bounds) # no gradient

for motor, position in zip(motors, start_positions):

    motor.move(position)
