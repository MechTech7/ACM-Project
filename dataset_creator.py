import numpy as np
import io
from webmercator import *
from util import *
import math
import cv2
from PIL import Image

OUTPUT_IMAGE_WIDTH = 224
def imageFile(x, y):
    return "imagery/14_"+str(x)+"_"+str(y)+".jpg"

def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))

def pre_process(image):
    #image is a [256, 256, 3] numpy array
    op = cv2.resize(image, dsize=(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
    op = (op / 255)
    return op

def import_process(jpg):
    im = Image.open(jpg)
    array = np.array(im)
    return pre_process(array)

def pre_process2(image):
    #image is a [256, 256, 3] numpy array
    op = cv2.resize(image, dsize=(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
    return op

def import_process2(jpg):
    im = Image.open(jpg)
    array = np.array(im)
    return pre_process2(array)
    

class ZipCode:
    def __init__(self,zipcode,latCoord,lonCoord):
        self.zipcode = zipcode
        self.tiles = []
        self.lat = latCoord
        self.lon = lonCoord
    def __str__(self):
        if (hasattr(self,"population")):
            return "ZIPCODE - zipcode: %s, # of tiles: %s, lat: %s, lon: %s, moneyPerPerson: %s, money: %s, population: %s" % (self.zipcode, len(self.tiles), self.lat, self.lon, self.moneyPerPop, self.money, self.population)
        return "ZIPCODE - zipcode: %s, # of tiles: %s, lat: %s, lon: %s" % (self.zipcode, len(self.tiles), self.lat, self.lon)

class Tile:
    def __init__(self,zipcode,xCoor,yCoor):
        self.zipCode = zipcode
        self.x = xCoor
        self.y = yCoor
        self.img = imageFile(self.x,self.y)
    def __str__(self):
        return "TILE - zipcode: %s, X: %s, Y: %s, img path: %s" % (self.zipCode.zipcode, self.x, self.y, self.img)

zipCodesLocationData = np.genfromtxt("ziplatlon.csv", skip_header=1, dtype=np.float32, delimiter=";")
zipCodesCodes = zipCodesLocationData[:, 0]
zipCodesLats = zipCodesLocationData[:, 3]
zipCodesLongs = zipCodesLocationData[:, 4]
zipCodes = {}
tiles = {}

zipCodesMoneyData = np.genfromtxt("16zpallnoagi.csv", skip_header=2, dtype=np.float32, delimiter=",")
zipCodesCodes2 = zipCodesMoneyData[:, 2]
zipCodesPopulation = zipCodesMoneyData[:, 4]
zipCodesMoney = zipCodesMoneyData[:, 20]
print("zipcode: "+str(zipCodesCodes2[0])+", population: "+str(zipCodesPopulation[0])+", money: "+str(zipCodesMoney[0]))


#iterate over the zip codes
print(zipCodesCodes.shape[0])

count = 0

for i in range(zipCodesCodes.shape[0]):
    latCoord = zipCodesLats[i]
    lonCoord = zipCodesLongs[i]
    zipcodeCode = int(zipCodesCodes[i])
    xCoord = int(round(x(lonCoord,14)))
    yCoord = int(round(y(latCoord,14)))
    if xCoord >= 2794 and xCoord < 2839 and yCoord >= 6528 and yCoord < 6572:
        newZip = ZipCode(zipcodeCode, latCoord, lonCoord)
        for j in range(zipCodesCodes2.shape[0]): 
            if (int(zipCodesCodes2[j]) == zipcodeCode):
                newZip.population = zipCodesPopulation[j]
                newZip.money = zipCodesMoney[j]
                assert(zipCodesMoney[j] > 0)
                newZip.moneyPerPop = zipCodesMoney[j]/zipCodesPopulation[j]
                break
        if (hasattr(newZip,"population")):
            zipCodes[zipcodeCode] = newZip
            newTile = Tile(newZip,xCoord,yCoord)
            tiles[newTile.img] = newTile
            newZip.tiles.append(newTile)
        else:
            #print(str(newZip) + " X: "+str(x(newZip.lon,14))+" Y: "+str(y(newZip.lat,14)))
            count += 1

print(count)

print(len(zipCodes.keys()))

for xCoord in range(2794, 2839):
    for yCoord in range(6528, 6572):
        lonCoord = lon(xCoord,14)
        latCoord = lat(yCoord,14)
        elev = getElevation(latCoord, lonCoord)
        imgCode = imageFile(xCoord,yCoord)
        if (imgCode not in tiles and elev > 0): 
            minI = 0
            minDist = float("inf")
            for i in zipCodes.keys():
                currDist = dist(x(zipCodes[i].lon,14), y(zipCodes[i].lat,14), xCoord, yCoord)
                if (currDist < minDist):
                    minDist = currDist
                    minI = i
            newTile = Tile(zipCodes[minI],xCoord,yCoord)
            zipCodes[minI].tiles.append(newTile)
            tiles[newTile.img] = newTile

count = 0
for i in zipCodes.keys():
    print(zipCodes[i])
    count += 1
    for j in zipCodes[i].tiles:
        print(j)
    print("")
    if (count >= 20):
        break

print(len(tiles))
nptiles = np.empty([len(tiles), 225, 224, 3])

idx = 0
for i in tiles.keys():
    current = tiles[i]
    npImageArray = import_process(current.img)
    extraLine = np.empty([1, 224, 3])

    if (idx < 4):
        npImageArray2 = import_process2(current.img)
        img = Image.fromarray(npImageArray2, 'RGB')
        newName = tiles[i].img[:tiles[i].img.find(".")] + "_2" + tiles[i].img[tiles[i].img.find("."):]
        print(tiles[i].zipCode.moneyPerPop)
        print(tiles[i].img)
        print(newName+"\n")
        img.save(newName)

    if (not hasattr(current.zipCode,"moneyPerPop")):
        print(current.zipCode)
    extraLine[0][0][0] = current.zipCode.moneyPerPop
    npImageArray = np.append(npImageArray, extraLine, axis=0)
    nptiles[idx] = npImageArray
    idx += 1

print(nptiles[0][224][0][0])
print(nptiles[1][224][0][0])
print(nptiles[2][224][0][0])
print(nptiles[3][224][0][0])
print(nptiles[4][224][0][0])
np.save("compiledData.npy", nptiles)

