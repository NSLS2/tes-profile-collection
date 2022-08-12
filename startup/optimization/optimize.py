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
    hull     = sp.spatial.qhull.ConvexHull(contour)
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





def get_loss(motor_positions):

    # Move the motors to these positions 
    # move_motors()

    # Collect the data
    # image = ...

    return -get_fitness(image)



def get_fitness(image):

    # Draw a contour, evaluate beam position and shape

    nx, ny = image.shape

    cx, cy, wx, wy, peak, flux, area, contour = get_beam_stats(image)

    fitness = - (np.abs(wx) + np.abs(wy) + np.abs(cx - nx/2) + np.abs(cy - ny/2))

    return fitness

# set bounds manually
# get readback from motors, initialize optimizers

sp.optimize.minimize(get_loss, x0=x0, bounds=bounds, method='L-BFGS-B') # classic

sp.optimize.differential_evolution(get_loss, x0=x0, bounds=bounds) # genetic 

sp.optimize.basinhopping(get_loss, x0=x0, bounds=bounds) # good for bumpy GD

sp.optimize.dual_annealing(get_loss, x0=x0, bounds=bounds) # this is probably not the best method 

sp.optimize.direct(get_loss, x0=x0, bounds=bounds) # no gradient 