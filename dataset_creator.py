import numpy as np
import io
from webmercator import *
from util import *
import math

def imageFile(x, y):
    return "imagery/14_"+str(x)+"_"+str(y)+".jpg"

def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))

class ZipCode:
    def __init__(self,zipcode,latCoord,lonCoord):
        self.zipcode = zipcode
        self.tiles = []
        self.lat = latCoord
        self.lon = lonCoord
    def __str__(self):
        if (hasattr(self,"population")):
            return "ZIPCODE - zipcode: %s, # of tiles: %s, lat: %s, lon: %s, moneyPerPerson: %s" % (self.zipcode, len(self.tiles), self.lat, self.lon, self.moneyPerPop)
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

#iterate over the zip codes
print(zipCodesCodes.shape[0])

for i in range(zipCodesCodes.shape[0]):
    latCoord = zipCodesLats[i]
    lonCoord = zipCodesLongs[i]
    zipcodeCode = int(zipCodesCodes[i])
    xCoord = int(round(x(lonCoord,14)))
    yCoord = int(round(y(latCoord,14)))
    if xCoord >= 2794 and xCoord < 2839 and yCoord >= 6528 and yCoord < 6572:
        newZip = ZipCode(zipcodeCode, latCoord, lonCoord)
        newTile = Tile(newZip,xCoord,yCoord)
        tiles[newTile.img] = newTile
        newZip.tiles.append(newTile)
        zipCodes[zipcodeCode] = newZip

removeList = []
for i in zipCodes.keys():
    currZipCode = i
    for j in range(zipCodesCodes2.shape[0]): 
        if (int(zipCodesCodes2[j]) == currZipCode):
            zipCodes[i].population = zipCodesPopulation[j]
            zipCodes[i].money = zipCodesMoney[j]
            zipCodes[i].moneyPerPop = zipCodesMoney[j]/zipCodesPopulation[j]
            break
    if (not hasattr(zipCodes[i],"population")):
        removeList.append(i)

for i in removeList:
    del zipCodes[i]

print(len(removeList))

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

