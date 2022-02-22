"""
@author: congzlwag
"""
import numpy as np
import scipy.sparse as sps

def foldQuadrantSps(mat_coo, x0=None, y0=None, quadrant_filter=[1,1,1,1]):
	"""
	Fold sparse matrix and sum quadrants. 

	Parameters
	----------
	mat_coo : scipy.sparse.coo_matrix
		It has to be a single matrix
	x0, y0 : int,int
		Coordinates of the central pixel
	quadrant_filter : bool(4)
		Whether to count in each quadrant or not. Following Elio's quadrant 
		package's convention, when plt.imshow this matrix
		quadrant  |  0		  |  1		 |  2		 |  3		  |
		location  |Upper-right  | Upper-left | Lower-left | Lower-right |
	
	Returns
	-------
	Sum of the folded quadrants
	"""
	sy, sx = mat_coo.shape
	if x0 is None:
		x0 = int(sx//2)
	if y0 is None:
		y0 = int(sy//2)
	
	lx, ly = ([], [])
	if quadrant_filter[1] or quadrant_filter[2]:
		lx.append(x0+1)
	if quadrant_filter[0] or quadrant_filter[3]:
		lx.append(sx-x0)
	if quadrant_filter[0] or quadrant_filter[1]:
		ly.append(y0+1)
	if quadrant_filter[2] or quadrant_filter[3]:
		ly.append(sy-y0)
	if len(lx)>0 and len(ly)>0:
		lx = min(lx)
		ly = min(ly)
	else:
		raise ValueError("No quadrant is selected. All 0")
	
	data, row, col = (mat_coo.data, mat_coo.row, mat_coo.col)
	new_data, new_row, new_col = ([], [], [])
	qsigns = np.array([[-1,1],[-1,-1],[1,-1],[1,1]])
	for q in range(4):
		qsy, qsx = qsigns[q]
		if quadrant_filter[q]:
			flags = ((row-y0)*qsy>=0) & ((col-x0)*qsx>=0) & (ly+qsy*(y0-row)>0) & (lx+qsx*(x0-col)>0)
			new_data.append(data[flags])
			new_row.append((row[flags]-y0)*qsy)
			new_col.append((col[flags]-x0)*qsx)
	new_data = np.concatenate(new_data)
	new_row = np.concatenate(new_row)
	new_col = np.concatenate(new_col)
	return sps.coo_matrix((new_data, (new_row, new_col)), shape=(ly,lx))

def resizeFoldedSps(mat_coo, Rmax):
	sy,sx = mat_coo.shape 
	data, row, col = (mat_coo.data, mat_coo.row, mat_coo.col)
	flags = np.ones(data.size, dtype=bool)
	if Rmax < sy:
		flags = flags & (row<Rmax)
		sy = Rmax
	if Rmax < sx:
		flags = flags & (col<Rmax)
		sx = Rmax
	return sps.coo_matrix((data[flags], (row[flags], col[flags])), shape=(Rmax,Rmax))