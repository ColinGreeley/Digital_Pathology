# generatetiles.py
#
# Two ways to run:
#   main1(): python3 generatetiles.py original_image.jpg [original_image_dots.csv]
#   main2(): python3 generatetiles.py (assumes presence of image_lists_cropped.py file)
#
# The _dots.csv file contains lines each of the format: color,x,y. The color refers to
# the type of disease pathology, but is ignored. There should be no header line in the
# file.
#
# Generates tile images of size TILE_SIZE x TILE_SIZE based on the original image.
# The tiles are taken every TILE_INCREMENT horizontally and vertically within the
# original image. If any part of the tile is white, then it is ignored. If the
# original_image_dots.csv file is present (either given for main1, or in the same
# directory as the image for main2), then the tile must also overlap with a location
# in the _dots.csv file. This also means that the tile is a diseased tile. Otherwise,
# the tile is assumed to be a healthy tile. The _dots.csv file may be present, but
# empty, which means it's a diseased image, but no dots are available and no tiles
# will be generated.
# 
# Tiles are output to files named as <original_image>_<diseased|non_diseased>_tile_NNNN.jpg
# in the same directory as the original image. NNNN starts are 0001 and increments by 1.

import cv2
import numpy as np
import sys
import os
import csv

TILE_SIZE = 256 # 256x256 image about the size of one cell
TILE_INCREMENT = 128

def overlapsDot(x1, y1, x2, y2, dotLocations):
    """Returns True if at least one dot location is contained within given rectangle."""
    for location in dotLocations:
        x = location[0]
        y = location[1]
        if (x > x1) and (x < x2) and (y > y1) and (y < y2):
            return True
    return False

def containsWhite(image):
    """Return True if any pixel is white."""
    h,w,c = image.shape
    image1 = (image > 250)
    image2 = image1.reshape(h*w,c)
    imagelist = image2.tolist()
    return any(x == [True,True,True] for x in imagelist)

def containsTooMuchBackground(image):
    """Return True if image contains more than 30% background color (off white)."""
    h,w,c = image.shape
    threshold = 0.3 * h * w
    image1 = (image > 200)
    image2 = image1.reshape(h*w,c)
    imagelist = image2.tolist()
    numbk = sum(1 for x in imagelist if x == [True,True,True])
    return (numbk > threshold)

def getDotLocations(dotLocationsFileName):
    dotLocations = []
    with open(dotLocationsFileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dotLocations.append([int(row[1]), int(row[2])])
    return dotLocations                                

def processDiseasedImage(imageFileName, dotLocationsFileName):
    global TILE_SIZE, TILE_INCREMENT
    dotLocations = getDotLocations(dotLocationsFileName)
    if not dotLocations:
        return 0
    fileRoot = os.path.splitext("../Generated_Images/")[0] + "_diseased_tile_"
    image = cv2.imread(imageFileName, cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    x1 = y1 = 0
    x2 = y2 = TILE_SIZE # yes, TILE_SIZE, not (TILE_SIZE - 1)
    tiles = []
    while y2 <= height: # yes, <=, not <
        while x2 <= width:
            tileImage = image[y1:y2, x1:x2]
            if overlapsDot(x1, y1, x2, y2, dotLocations) and (not containsWhite(tileImage)) and (not containsTooMuchBackground(tileImage)):
                tileFileName = fileRoot + str(numTiles).zfill(4) + ".jpg"
                #cv2.imwrite(tileFileName, tileImage)
                tiles.append(tileImage)
            x1 += TILE_INCREMENT
            x2 += TILE_INCREMENT
        x1 = 0
        x2 = TILE_SIZE
        y1 += TILE_INCREMENT
        y2 += TILE_INCREMENT   
    return tiles             

def processNonDiseasedImage(imageFileName):
    global TILE_SIZE, TILE_INCREMENT
    #fileRoot = os.path.splitext("../Generated_Images/")[0] + "_non_diseased_tile_"
    image = cv2.imread(imageFileName, cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    x1 = y1 = 0
    x2 = y2 = TILE_SIZE # yes, TILE_SIZE, not (TILE_SIZE - 1)
    tiles = []
    locs = []
    while y2 <= height: # yes, <=, not <
        while x2 <= width:
            tileImage = image[y1:y2, x1:x2]
            if (not containsWhite(tileImage)) and (not containsTooMuchBackground(tileImage)):
                #tileFileName = fileRoot + str(numTiles).zfill(4) + ".jpg"
                #cv2.imwrite(tileFileName, tileImage)
                tiles.append(tileImage)
                locs.append((x1, y1, x2, y2))
            x1 += TILE_INCREMENT
            x2 += TILE_INCREMENT
        x1 = 0
        x2 = TILE_SIZE
        y1 += TILE_INCREMENT
        y2 += TILE_INCREMENT
    return np.asarray(tiles), np.asarray(locs), image
  
def main1():
    # Get file names from command line
    origImageFileName = sys.argv[1] # "original.jpg"
    if len(sys.argv) > 2:
        dotLocationsFileName = sys.argv[2]
        numTiles = processDiseasedImage(origImageFileName, dotLocationsFileName)
        print("Wrote " + str(numTiles) + " diseased tiles")
    else:
        numTiles = processNonDiseasedImage(origImageFileName)
        print(numTiles)
        print(type(numTiles))
        print("Wrote " + str(numTiles) + " non-diseased tiles")

#import image_lists_cropped # comment out if not available (i.e., using main1)
 
def main2():
    # Generate tiles from diseased images
    diseased_images = image_lists_cropped.diseased_high_res_with_marked
    num_images = len(diseased_images)
    numTiles = 0
    for i in range(0,num_images):
        origImageFileName = diseased_images[i]
        fileRoot = os.path.splitext(origImageFileName)[0]
        dotLocationsFileName = fileRoot + "_dots.csv" # _dots.csv file assumed to exist; run finddots2 to generate these
        numTiles += processDiseasedImage(origImageFileName, dotLocationsFileName)
    print("Wrote " + str(numTiles) + " diseased tiles")
    # Generate tiles from non-diseased images (include last list?)
    non_diseased_images = image_lists_cropped.non_diseased + image_lists_cropped.non_diseased_high_res_no_marked #+ image_lists_cropped.non_diseased_high_res_with_marked
    num_images = len(non_diseased_images)
    numTiles = 0
    for i in range(0,num_images):
        origImageFileName = non_diseased_images[i]
        numTiles += processNonDiseasedImage(origImageFileName)
    print("Wrote " + str(numTiles) + " non-diseased tiles")

if __name__ == '__main__':
    main1()
