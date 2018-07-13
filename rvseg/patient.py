#!/usr/bin/env python

from __future__ import division, print_function

import os, glob, re
import dicom
import numpy as np
from PIL import Image, ImageDraw
import json
import cv2
import skimage.draw
from keras import utils


def maybe_rotate(image):
    # orient image in landscape
    height, width = image.shape
    return np.rot90(image) if width < height else image

class PatientData(object):
    """Data directory structure (for patient 01):
    directory/
      P01dicom.txt
      P01dicom/
        P01-0000.dcm
        P01-0001.dcm
        ...
      P01contours-manual/
        P01-0080-icontour-manual.txt
        P01-0120-ocontour-manual.txt
        ...
    """
    def __init__(self, directory):
        self.directory = os.path.normpath(directory)

        # load all data into memory
        self.load_images()
        '''
        try:
            self.load_masks()
        except FileNotFoundError:
            pass
        '''

    @property
    def images(self):
        return self.all_images
    @property
    def masks(self):
        return self.all_masks

    @property
    def dicoms(self):
        assert 0
        return [self.all_dicoms[i] for i in self.labeled]

    @property
    def dicom_path(self):
        assert 0
        return os.path.join(self.directory, "P{:02d}dicom".format(self.index))

    def load_images(self):
        print(self.directory)
        annotations = json.load(open(os.path.join(self.directory, "plaque_b_train.json")))
        trainsize=annotations['size']
        traindatas=annotations['datas']
        self.all_images = []
        self.all_masks=[]
        for a in traindatas:
            image_id=a['imagename']
            path=self.directory+'/'+a['imagename']
            image_info = {
                    "id": image_id,
                    "width": a['width'],
                    "height": a['height'],
                    "path":path,
                    "polygons":a['imageshapes']
                    }
            mask_info = {
                    "id": image_id,
                    "width": a['width'],
                    "height": a['height'],
                    "path":path,
                    "polygons":a['imageshapes']
                    }
            self.all_images.append(image_info)
            self.all_masks.append(mask_info)

    def load_a_image(self,imageinfo):
        imagepath=imageinfo['path']
        return np.expand_dims(cv2.imread(imagepath,0),axis=2)

    def load_a_mask(self,maskinfo):
        polygons=maskinfo['polygons']
        masktemp = np.zeros([maskinfo["height"], maskinfo["width"], len(maskinfo["polygons"])], dtype=np.uint8)
        for i, p in enumerate(polygons):
            xy = list(map(tuple, p['points']))
            xy=np.array(xy,dtype=np.int32)
            rr, cc = skimage.draw.polygon(xy[0:,1], xy[0:,0])
            masktemp[rr, cc, i] = 1
        masktemp=np.sum(masktemp,axis=2)
        masktemp=np.expand_dims(masktemp,axis=2)
        #one-hot
        masktemp=utils.to_categorical(masktemp).reshape([maskinfo["height"], maskinfo["width"],3])
        return masktemp

    '''
    def load_contour(self, filename):
        # strip out path head "patientXX/"
        match = re.search("patient../(.*)", filename)
        path = os.path.join(self.directory, match.group(1))
        x, y = np.loadtxt(path).T
        if self.rotated:
            x, y = y, self.image_height - x
        return x, y

    def contour_to_mask(self, x, y, norm=255):
        BW_8BIT = 'L'
        polygon = list(zip(x, y))
        image_dims = (self.image_width, self.image_height)
        img = Image.new(BW_8BIT, image_dims, color=0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        return norm * np.array(img, dtype='uint8')

    def load_masks(self):
        with open(self.contour_list_file, 'r') as f:
            files = [line.strip() for line in f.readlines()]

        inner_files = [path.replace("\\", "/") for path in files[0::2]]
        outer_files = [path.replace("\\", "/") for path in files[1::2]]

        # get list of frames which have contours
        self.labeled = []
        for inner_file in inner_files:
            match = re.search("P..-(....)-.contour", inner_file)
            frame_number = int(match.group(1))
            self.labeled.append(frame_number)

        self.endocardium_contours = []
        self.epicardium_contours = []
        self.endocardium_masks = []
        self.epicardium_masks = []
        for inner_file, outer_file in zip(inner_files, outer_files):
            inner_x, inner_y = self.load_contour(inner_file)
            self.endocardium_contours.append((inner_x, inner_y))
            outer_x, outer_y = self.load_contour(outer_file)
            self.epicardium_contours.append((outer_x, outer_y))

            inner_mask = self.contour_to_mask(inner_x, inner_y, norm=1)
            self.endocardium_masks.append(inner_mask)
            outer_mask = self.contour_to_mask(outer_x, outer_y, norm=1)
            self.epicardium_masks.append(outer_mask)
            
    def write_video(self, outfile, FPS=24):
        import cv2
        image_dims = (self.image_width, self.image_height)
        video = cv2.VideoWriter(outfile, -1, FPS, image_dims)
        for image in self.all_images:
            grayscale = np.asarray(image * (255 / image.max()), dtype='uint8')
            video.write(cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR))
        video.release()
    '''

