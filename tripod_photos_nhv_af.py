# on Rich's laptop, use 32 bit EPD (with cv2)

from pylab import *
import cv2
from common import anorm
from scipy import optimize
from PIL import Image, ImageFont, ImageDraw
import os
import glob
import sys
import unsharp
import time
from multiprocessing import Pool,cpu_count

NUM_PROCESSES = cpu_count()
#NUM_PROCESSES = 1
#BASEDIR = "/Volumes/Expansion Drive/Tripod_166"
BASEDIR = "/RPS/bottom_photos"


print "tripod_photos"	 

def annotate_frame(im,iframe,dat,tim):
	draw = ImageDraw.Draw(im)
#        imshow(im)
	# write date/time from spreadsheet onto image

	datstr='%2.2d/%2.2d/%2.2d' % (dat[2],dat[0],dat[1])
	timstr='%2.2d:%2.2d:%2.2d' % (tim[0],tim[1],tim[2])
	frmstr='%3.3d' % (iframe)
	font = ImageFont.truetype('arial.ttf',36)
	#font = ImageFont.truetype(os.path.join(BASEDIR,'python','fonts','arial.ttf'),36)
	#font = ImageFont.load_default()
	draw.text((95,820),datstr,font=font,fill=255)
	draw.text((130,860),timstr,font=font,fill=255)
	draw.text((100,1000),frmstr,font=font,fill=255)
	
def read_frame(fname):
#	print "read_frame( %s )"%(fname)
	img = cv2.imread(fname)
#	img = Image.open(fname)
	img = 255 - cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY)
#	img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY)
#	img = cv2.equalizeHist(img)  #Norm, you probably want to remove this
	return img

def plot_points(img1,img2,u1,u2,ind,p,iframe):
	#plot 1st image
	fitfunc1 = lambda a, b, x1, y1: b+x1*np.cos(a)+y1*np.sin(a) # Target function
	fitfunc2 = lambda a, b, x1, y1: b-x1*np.sin(a)+y1*np.cos(a) # Target function
	ang=p[0]
	x0=p[1]
	y0=p[2]
	plt.imshow(img1,cmap=plt.cm.gray)
	plt.show()
	#plot 2nd image with 50% transparency
	plt.imshow(img2,cmap=plt.cm.gray, alpha=0.5)
	# x'  =		ang	  x0	 x		   y
	x3=fitfunc1(ang,x0,u1[ind,0],u1[ind,1])
	# y'  =		ang	  y0	 x		   y
	y3=fitfunc2(ang,y0,u1[ind,0],u1[ind,1])
	plt.plot(u1[ind,0],u1[ind,1],'ro',
		u2[ind,0],u2[ind,1],'go',
		x3,y3,'w.')
	plt.title('frame %d' % iframe)
#	 time.sleep(1)
	plt.draw()
#	 plt.savefig('c:/rps/python/image/frames/frames%3.3d' % iframe)
#	plt.savefig('/g/nhv/python/image/frames/frames%3.3d' % iframe)	   
	plt.savefig(os.path.join(BASEDIR,'frames','frames%3.3d') % iframe)	   
	plt.clf()
	  
def affine_lsq(u1,u2):
	# Target functions
	#x'=x0+x*cos(a)+y*sin(a)
	#y'=y0-x*sin(a)+y*cos(a)	   
	fitfunc1 = lambda a, b, x1, y1: b+x1*np.cos(a)+y1*np.sin(a) # Target function
	fitfunc2 = lambda a, b, x1, y1: b-x1*np.sin(a)+y1*np.cos(a) # Target function
	
	# Initial guess for the first set's parameters
	p1 = np.r_[0.]
	# Initial guess for the second set's parameters
	p2 = np.r_[0.]
	# Initial guess for the common angle
	a = 0.0
	# Vector of the parameters to fit, it contains all the parameters of the problem,  
	# and the period of the oscillation is not there twice !
	p = np.r_[a, p1, p2]
	# Cost function of the fit, compare it to the previous example.
	errfunc = lambda p, x1, y1, x2, y2: np.r_[
					fitfunc1(p[0], p[1], x1, y1) - x2,
					fitfunc2(p[0], p[2], x1, y1) - y2
				]
	# This time we need to pass the two sets of data, there are thus four "args".
	p,success = optimize.leastsq(errfunc, p, 
									args=(u1[:,0], u1[:,1], u2[:,0], u2[:,1]))
	
	ang=p[0]  # rotation angle
	x0=p[1]	  # x offset
	y0=p[2]	  # y offset

	# If rotation angle is more than angmax degrees, don't use it
	# and just calculate mean offset instead
	angmax=5.0
	if np.abs(ang*180./np.pi) >= angmax:
		# 2x3 affine matrix M
		x0=np.median(u2[:,0]-u1[:,0])
		y0=np.median(u2[:,1]-u1[:,1])
		ang=0.0

	# 2x3 affine matrix M
	M=np.array([np.cos(ang),np.sin(ang),x0,-np.sin(ang),np.cos(ang),y0]).reshape(2,3)
	return M,p

def match_bruteforce(desc1, desc2, r_threshold = 0.75):
	res = []
	for i in xrange(len(desc1)):
		dist = anorm( desc2 - desc1[i] )
		n1, n2 = dist.argsort()[:2]
		r = dist[n1] / dist[n2]
		if r < r_threshold:
			res.append((i, n1))
	return np.array(res)

def enhance_image(img,cmap):
    im=Image.fromarray(img)
    im=unsharp.usm(im,50,20,1.7,0.02)
    im=im.convert("L")
    im.putpalette(cmap)
    img = asarray(im)
    return img
    
def enhance_im(im,cmap):
    im=unsharp.usm(im,50,20,1.7,0.02)
    im=im.convert("L")
    im.putpalette(cmap)
    return im

def make_cpt(file):
    cmap=np.genfromtxt(file,delimiter='\t',skip_header=2,skip_footer=3,
            names=['pv1','r1','g1','b1','pv2','r2','g2','b2'],dtype=None)
    gmap = np.array([cmap[i][1] for i in xrange(len(cmap))])/255.
    b5=np.asarray([gmap, gmap, gmap]).T
    gmap2=matplotlib.colors.ListedColormap(b5)  # matplotlib colormap
    cmap=b5.reshape(768,1)*255   # PIL paletteim.putpalette(b6)
    return cmap
    


# generator of image list and arguments to feed multi processing pool
def gen_images(af,idir,odir):
        #create custom color map
    	cmap = make_cpt('c:/rps/usgs_photo/python/b166b.cpt')

	# read 1st frame
	filename=af['raw'][0]
	iframe=int(filename[9:12])	  # frame number, eg, 0001, 0002
	#frame_name = images[0]
	frame_name=os.path.join(idir,filename) # full path name to image

	img1 = read_frame(frame_name)

	print frame_name
	"""
	for img in images:
		yield  {'image':img,
				'img1':img1,
				'mask':mask,
				'fmask':fmask,
				'hessianThreshold':hessianThreshold,
				'odir':odir }
	"""
	print "Entering Loop"
	for img in af:
		filename=img['raw']	      #filename 
		if filename[12] == '.':   #process frames with .tif, not _b.tif:
#			print 'gen_images( %s )'% filename
			dat=[int(x) for x in img['date'].split('/')]
			tim=[int(x) for x in img['time'].split(':')]
			args = {'img1':img1,
					'idir':idir,
					'odir':odir, 
					'filename':filename,
					'dat':dat,
					'tim':tim,
					'cmap':cmap}
#			print args
                        print filename
			yield args
		
def process(args):
#	print "process\n",args
	img1 = args['img1']
	idir = args['idir']
	odir = args['odir']
	dat = args['dat']
	tim = args['tim']
	filename = args['filename']
	cmap = args['cmap']
	

	iframe=int(filename[9:12])	             #frame number [001,002,...]
	frame_name=os.path.join(idir,filename)	 #full path filename

	print 'process( %s )'% frame_name

	# create a mask that processes only the central region
	mask = zeros(img1.shape,dtype=uint8)
	mask[20:1040,140:1500] = 1

	# mask for black border on saved image
	fmask = zeros(img1.shape,dtype=uint8)	# black
	fmask[20:1050,90:1680] = 1	#y,x

	# create an OpenCV SURF (Speeded Up Robust Features) object
	hessianThreshold = 1200

	img2 = read_frame(frame_name)
 #       img2 = enhance_image(img2,cmap)
	detector = cv2.SURF(hessianThreshold)
	kp1, desc1 = detector.detectAndCompute(img1, mask)
	kp2, desc2 = detector.detectAndCompute(img2, mask)

	desc1.shape = (-1, detector.descriptorSize())
	desc2.shape = (-1, detector.descriptorSize())
	r_threshold = 0.75
	m = match_bruteforce(desc1, desc2, r_threshold)
	u1 = np.array([kp1[i].pt for i, j in m])
	u2 = np.array([kp2[j].pt for i, j in m])
	H, status = cv2.findHomography(u1, u2, cv2.RANSAC, 1.0)
	M=H[0:2,:]
		
		# this affine matrix from findHomography is not as constrained as doing
		# least-squares to allow affine only, so only use the 'status' 
		# from findHomography to indicate
		# "good pairs of points" to use in least-squares fit below

	ind=where(status)[0]
			
	M,p = affine_lsq(u1[ind],u2[ind])
		# invert the affine matrix so we can register image 2 to image 1
	Minv = cv2.invertAffineTransform(M)
	img2b = cv2.warpAffine(img2,Minv,np.shape(img2.T))

	# transform successive frames rather than just the 1st frame
	# comment this out to just difference on the 1st frame
	#img1=img2b
		
		# fill in left side with grey from single pixel
	img2b[0:624,90:152] = img2b[625,152] 
	img2b[624:,90:109] = img2b[625,152]
		
	# create PIL image
	# overlay registered image with black border to eliminate
	# annoying jumping of ragged border regions when animating

	im = Image.fromarray(img2b*fmask)
	
	im = enhance_im(im,cmap)

        
	# annotate frame with date and time
	print filename,dat,tim
	annotate_frame(im,iframe,dat,tim)
	
  
        
	plotpoints=False
	if plotpoints:
		plot_points(img1,img2,u1,u2,ind,p,iframe)
        
	im.save(('%s/%3.3d.png' % (odir,iframe)),'png')
   

def main():
	# read input frames from spreadsheet, process frames listed in last column
	# and read date and time (typed in by Erin after looking at each frame)

	#directory with raw frames
	idir=os.path.join(BASEDIR,'Tripod_166','Original Raw Transfers')
#	idir = "/Users/Shared/USGS/Tripod_166/Original Raw Transfers/"
	print "idir = %s"%(idir)

#	idir = "/Users/Shared/USGS/nhv/Processed"
#	images = glob.glob(os.path.join(idir,"*.tif"))
#	print images

	spreadsheet = os.path.join(BASEDIR,'Tripod_166','framelist166_fixed.xls')
	af=np.genfromtxt(spreadsheet,delimiter='\t',
					 skiprows=4,names=['file','date','time','stab','raw'],dtype=None)

	#directory for output frames
	odir=os.path.join(BASEDIR,'nhv','frames','affine_af')
	if not os.path.exists(odir):
		os.makedirs(odir)
	print "odir = %s"%(odir)

	# create our process pool
	pool = Pool(processes=NUM_PROCESSES)
	results = pool.map_async(process, gen_images(af,idir,odir))
	pool.close()
	pool.join()

if __name__ == "__main__":
	main()
	
