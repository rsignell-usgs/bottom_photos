# we need PythonMagick for its unsharp mask functions
#

from PIL import Image
from PythonMagick import Image as PMImage #distinguish between PIL and PythonMagick 'Image' object
from PythonMagick import Blob 

#Convert Magick image to PIL (Python Imageing Library) image
#taken from ftp://ftp.imagemagick.net/pub/ImageMagick/python/README.txt (which contains an error:
#Image.data does not exist
#see Magick++ documentation for the PythonMagick functions

def convertMGtoPIL(magickimage):
    'works with grayscale and color'
    img = PMImage(magickimage)  # make copy
    img.depth = 8        #  this takes 0.04 sec. for 640x480 image
    img.magick = "RGB"
    w, h = img.columns(), img.rows()
    blb=Blob()
    img.write(blb)
    data = blb.data
    
    # convert string array to an RGB Pil image
    pilimage = Image.fromstring('RGB', (w, h), data)
    return pilimage

#Convert PIL image to Magick image

def convertPILtoMG(pilimage):
    'returns RGB image'
    if pilimage == None:
        return None
    if pilimage.mode == 'L':
        pilimage = pilimage.convert("RGB")
    size = "%sx%s" % (pilimage.size[0], pilimage.size[1])
    data = pilimage.tostring('raw','RGB')
    mdata = Blob(data)
    img = PMImage(mdata, size, 8, 'RGB')
    return img

#unsharp mask function, see ImageMagick docs for explanation of radius, sigma, amount and threshold
def usm(pilimage,radius,sigma,amount,threshold):
	img=convertPILtoMG(pilimage)
	img.unsharpmask(radius,sigma,amount,threshold)
	pilimage=convertMGtoPIL(img)
	return pilimage