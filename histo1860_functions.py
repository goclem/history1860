#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the ARTHISTO project
@author: Clement Gorin
@contact: gorin@gate.cnrs.fr
@date: January 2020
'''

#%% PACKAGES

import argparse
import numpy
import matplotlib.pyplot as plt
import os
import pandas
import re
import socket
import shutil
import scipy.ndimage
import skimage.morphology

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

#%% PATHS

# Sets Paths for the project
def makePaths(project=None):
    if socket.gethostname() == 'PSE-CALC-MAPHIS':
        os.environ['USERPROFILE']="F:\cgorin"
    source = os.path.join(os.path.expanduser('~'), 'Dropbox', 'data') 
    base   = os.path.join(os.path.expanduser('~'), 'Dropbox', 'research', 'arthisto')
    paths  = argparse.Namespace(
        base    = base,
        scem    = os.path.join(source, 'ign_scem'),
        ohs     = os.path.join(source, 'ign_ohs'),
        code    = os.path.join(base,   'code_1860'),
        data    = os.path.join(base,   'data_1860'),
        destat  = os.path.join(base,   'data_1860', 'destat'),
        tile    = os.path.join(base,   'data_1860', 'tiles'),
        mask    = os.path.join(base,   'data_1860', 'mask'),
        tmp     = os.path.join(os.path.expanduser('~'), 'tmp'),
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop'),
        france  = os.path.join(base, 'data_project', 'fr15_buffer.gpkg')
    )
    if project == 'buildings':
        data   = os.path.join(base, 'data_1860', 'buildings')
        paths2 = argparse.Namespace(
            y      = os.path.join(paths.data, 'y'),
            X      = os.path.join(paths.data, 'X'),
            styles = os.path.join(paths.data, 'styles'),
            Xy1    = os.path.join(data, 'Xy1'),
            y2     = os.path.join(data, 'y2'),
            X1     = os.path.join(data, 'X1'),
            X2     = os.path.join(data, 'X2'),
            yh1    = os.path.join(data, 'yh1'),
            yh2    = os.path.join(data, 'yh2'),
            yh3    = os.path.join(data, 'yh3'),
            fh1    = os.path.join(data, 'fh1'),
            fh2    = os.path.join(data, 'fh2')
        )
        vars(paths).update(vars(paths2))
    if project == 'landuse':
        data   = os.path.join(base, 'data_1860', 'landuse')
        paths2 = argparse.Namespace(
            y      = os.path.join(paths.data, 'y'),
            X      = os.path.join(paths.data, 'X'),
            Xdat   = os.path.join(data, 'Xdat'),
            ydat   = os.path.join(data, 'ydat'),
            yh     = os.path.join(data, 'yh'),
            fh     = os.path.join(data, 'fh'),
            seg    = os.path.join(data, 'seg'),
            ids    = os.path.join(data, 'ids'),
            hull   = os.path.join(data, 'hull'),
            styles = os.path.join(paths.data, 'styles')
        )
        vars(paths).update(vars(paths2))
    return(paths)

#%% INPUT OUTPUT OPERATIONS

# Raster to array
def raster2array(srcRstPath, dataType=None):
    # import gdal
    srcRst = gdal.Open(srcRstPath)
    nBands = srcRst.RasterCount
    outArr = [None] * nBands
    for srcBnd in range(nBands):
        outArr[srcBnd] = srcRst.GetRasterBand(srcBnd + 1).ReadAsArray()
    outArr = numpy.dstack((outArr))
    if nBands == 1:
       outArr = outArr[:,:,0]
    if dataType is not None:
        outArr = outArr.astype(dataType)
    return(outArr)

# Array to raster
def array2raster(srcArr, srcRstPath, outRstPath, driver:str='GTiff', noDataValue=0, dataType=gdal.GDT_Byte):
    '''
    Description:
        Converts a numpy array to a raster file
    
    Parameters:
        srcArr      (obj): Source numpy array
        srcRstPath  (str): Path to source raster file
        outRstPath  (str): Path to output raster file
        driver      (str): GDAL driver
        noDataValue (int): Values labelled as missing
        dataType    (int): GDAL data type of the output raster
        
    Returns:
        Raster file
    '''
    # import gdal
    nBands = 1 if len(srcArr.shape) == 2 else srcArr.shape[2]
    srcRst = gdal.Open(srcRstPath)
    rstDrv = gdal.GetDriverByName(driver)
    outRst = rstDrv.Create(outRstPath, srcRst.RasterXSize, srcRst.RasterYSize, nBands, dataType)
    outRst.SetGeoTransform(srcRst.GetGeoTransform())
    outRst.SetProjection(srcRst.GetProjection())
    if nBands == 1:
        outBnd = outRst.GetRasterBand(1)
        outBnd.SetNoDataValue(noDataValue)
        outBnd.WriteArray(srcArr)
    else:
        for i in range(nBands):
            outBnd = outRst.GetRasterBand(i + 1)
            outBnd.SetNoDataValue(noDataValue)
            outBnd.WriteArray(srcArr[:, :, i])
    del(srcRst, outRst, outBnd)
    return(outRstPath)

# Raster to vector
def raster2vector(srcRstPath:str, outVecPath:str, driver:str='ESRI Shapefile', bandIndex:int=1, connected:int=4, layerName:str='vectorised', fieldName:str=None, dataType:int=ogr.OFTInteger, fieldIndex:int=-1):
    '''
    Description:
        Converts a raster file to a vector file
    
    Parameters:
        srcRstPath (str): Path to source raster file 
        outVecPath (str): Path to output vector file
        driver     (str): OGR driver
        connected  (int): Connectedness of raster cells (4 or 8)
        bandIndex  (int): Index of the raster band
        layerName  (str): Name of the output vector layer
        filedName  (str): Name of the output vector field (optional)
        fieldIndex (int): Default value when no field is created
        dataType   (int): Type of the output vector field
        
    Returns:
        Polygonised raster file in vector format
    '''
    # import gdal, ogr, osr
    srcRst = gdal.Open(srcRstPath)
    srcBnd = srcRst.GetRasterBand(bandIndex)
    vecDrv = ogr.GetDriverByName(driver)
    if os.path.exists(outVecPath):
        vecDrv.DeleteDataSource(outVecPath)
    outVec = vecDrv.CreateDataSource(outVecPath)
    ourSrs = osr.SpatialReference(srcRst.GetProjection())
    outLay = outVec.CreateLayer(layerName, ourSrs)    
    if fieldName is not None:
        fieldIndex = 0
        outField   = ogr.FieldDefn(fieldName, dataType)
        outLay.CreateField(outField)
    gdal.Polygonize(srcBnd, srcBnd, outLay, fieldIndex, ['8CONNECTED=' + str(connected)], callback=None)
    outVec.Destroy()

# Vector to raster
def vector2raster(srcVecPath:str, srcRstPath:str, outRstPath:str, driver:str='GTiff', burnField:str=None, burnValue:int=1, noDataValue:int=0, dataType:int=gdal.GDT_Byte):
    '''
    Description:
        Converts a vector file to a raster file
    
    Parameters:
        srcVecPath  (str): Path to source vector file 
        srcRstPath  (str): Path to source raster file
        outRstPath  (str): Path to output raster file
        driver      (str): GDAL driver
        burnField   (str): Field to burn (overrides burnValue)
        burnValue   (int): Fixed value to burn
        noDataValue (int): Missing value in raster
        dataType    (int): GDAL data type
        
    Returns:
        Rasterised vector file in raster format
    '''
    # import gdal, ogr
    srcRst = gdal.Open(srcRstPath, gdal.GA_ReadOnly)
    srcVec = ogr.Open(srcVecPath)
    srcLay = srcVec.GetLayer()
    rstDrv = gdal.GetDriverByName('GTiff')
    outRst = rstDrv.Create(outRstPath, srcRst.RasterXSize, srcRst.RasterYSize, 1, dataType)
    outRst.SetGeoTransform(srcRst.GetGeoTransform())
    outRst.SetProjection(srcRst.GetProjection())
    outBnd = outRst.GetRasterBand(1)
    outBnd.Fill(noDataValue)
    outBnd.SetNoDataValue(noDataValue)
    if burnField is not None:
        gdal.RasterizeLayer(outRst, [1], srcLay, options = ['ATTRIBUTE=' + burnField])
    else:
        gdal.RasterizeLayer(outRst, [1], srcLay, burn_values = [burnValue])
    outRst = None
    
# Sieve a raster
def sieveRaster(srcRstPath, outRstPath, threshold=None, connectedness=4, noDataValue=-1, dataType=gdal.GDT_Float32):
   # import gdal
   srcRst = gdal.Open(srcRstPath)
   srcBnd = srcRst.GetRasterBand(1)
   driver = gdal.GetDriverByName('GTiff')
   outRst = driver.Create(outRstPath, srcRst.RasterXSize, srcRst.RasterYSize, 1, dataType)
   outRst.SetGeoTransform(srcRst.GetGeoTransform())
   outRst.SetProjection(srcRst.GetProjection())
   outBnd = outRst.GetRasterBand(1)
   outBnd.Fill(noDataValue)
   outBnd.SetNoDataValue(noDataValue)
   gdal.SieveFilter(srcBnd, None, outBnd, threshold, connectedness)
   del(srcRst, srcBnd, outRst, outBnd)

# Masks a raster
def maskRaster(srcRstPath, mskRstPath, outRstPath, maskValue=0, updateValue=0, noDataValue=-1, dataType=gdal.GDT_Float32):
    srcRst = raster2array(srcRstPath)
    mskRst = raster2array(mskRstPath)
    srcRst[mskRst == maskValue] = updateValue
    array2raster(srcRst, srcRstPath, outRstPath, noDataValue, dataType)

# Dataframe to raster
def variables2rasters(srcDf, srcSegPath, outPath):
   srcSeg   = raster2array(srcSegPath)
   outPaths = [os.path.join(outPath + '_' + varName + '.tif') for varName in srcDf.columns.values.tolist()]
   for i in range(1, len(outPaths)):
       var = srcDf.iloc[:, i].values[srcSeg.flatten()].reshape(srcSeg.shape[0], srcSeg.shape[1])
       array2raster(var, srcSegPath, outPaths[i])

# Raster2xy
def raster2xy(srcRstPath):
    srcRst = gdal.Open(srcRstPath)
    xmin, xres, xskew, ymax, yskew, yres = srcRst.GetGeoTransform()
    xsize = srcRst.RasterXSize
    ysize = srcRst.RasterYSize
    xmax  = xmin + xsize * xres
    ymin  = ymax - ysize * yres
    xy    = numpy.dstack(numpy.meshgrid(numpy.arange(xmin + xres / 2, xmax, xres), numpy.arange(ymin + yres / 2, ymax, yres)))
    return(xy)

# summarise
def summarise(srcArr, srcIdx, function=None):
    nr, nc, nb = srcArr.shape
    srcIdx = srcIdx.flatten()
    srcIdx = numpy.unique(srcIdx, return_inverse=True)[1].reshape(srcIdx.shape)
    srcArr = numpy.reshape(srcArr, (numpy.prod([nr, nc]), nb))
    offset = srcIdx.max() + 1
    index  = numpy.stack([srcIdx + i * offset for i in range(nb)], axis = -1)
    outArr = function(srcArr, labels=index, index=range(offset * nb)).reshape(nb, offset).T
    return(outArr)

def table(array):
    # import pandas as pd
    print(pandas.value_counts(array.flatten()))

def mapValues(array, values, replacement):
    output = numpy.zeros(values.max() + 1, dtype=replacement.dtype)
    output[values] = replacement
    output = output[array]
    return(output)    

#%% Plots
    
def plot(array, figsize=(10, 10), dpi=100):
    # import matplotlib.pyplot as plt
    fig = plt.figure(dpi=dpi, figsize=figsize)
    axs = fig.add_subplot()
    axs.imshow(array, cmap='gray', vmin=numpy.min(array), vmax=numpy.max(array))
    axs.axis('off')
    plt.show()

def plot_compare(array1, array2):
    fig, axs = plt.subplots(ncols=2, figsize=(15, 30), dpi=100)
    axs[0].imshow(array1, cmap='gray', vmin=numpy.min(array1), vmax=numpy.max(array1))
    axs[1].imshow(array2, cmap='gray', vmin=numpy.min(array2), vmax=numpy.max(array2))
    axs[0].axis('off')
    axs[1].axis('off')    
    plt.show()

#%% Utilities

def try_wrapper(fun, arg):
    expr = fun.__name__ + '(\'%s\')'%(arg)
    print('Trying ' + expr)
    try:
        eval(expr)
        print(expr + ' done')
    except:
        print(expr + ' fail')
        pass

# Confirms
def confirm():
    check = str(input('Confirm ? (Y/N): ')).lower().strip()
    if check[0] == 'y':
        return True
    elif check[0] == 'n':
        return False
    else:
        print('Invalid Input')
        return(confirm())

# Move files
def move_files(root_src_dir, root_tar_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_tar_dir)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.move(src_file, dst_dir)
        shutil.rmtree(root_src_dir)

# List files
def listFiles(path, pattern='.', fullPath=True, extension=True):
    # import re, os
    fileList = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if fullPath == True:
                fileList.append(os.path.join(root, file))
            else:
                fileList.append(file)
    fileList = list(filter(re.compile(pattern).search, fileList))
    if extension == False:
        fileList = [os.path.splitext(os.path.basename(file))[0] for file in fileList]
    fileList.sort() 
    return(fileList)

# List folders
def listFolds(path, pattern='.', fullPath=True):
    # import re, os
    folds = list(filter(re.compile(pattern).search, os.listdir(path)))
    if fullPath == True:
        folds = [os.path.join(path, fold) for fold in folds]
    return(folds)

# Rests a folder without confirmation
def resetFolder(path, remove=False):
    if remove == True:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    if remove == False:
        if not os.path.exists(path):
            os.mkdir(path)

# Reset a folder
def resetFolderConfirm(path, remove=False):
    if remove == True:
        check = confirm()
        if os.path.exists(path) and check == True:
            shutil.rmtree(path)
            os.mkdir(path)
            print('Folder reseted')
        if check == False:
            print('Aborted')
    if remove == False:
        if not os.path.exists(path):
            os.mkdir(path)
            print('Folder created')

def fileName(path, noExt=True):
    return(os.path.splitext(os.path.basename(path))[0])

# OGR layer definition
def ogrLaydef(srcLay):
    layDef = srcLay.GetLayerDefn()
    for i in range(layDef.GetFieldCount()):
        fieldName     = layDef.GetFieldDefn(i).GetName()
        fieldataTypeCode = layDef.GetFieldDefn(i).GetType()
        fieldataType     = layDef.GetFieldDefn(i).GetFieldataTypeName(fieldataTypeCode)
        fieldWidth    = layDef.GetFieldDefn(i).GetWidth()
        GetPrecision  = layDef.GetFieldDefn(i).GetPrecision()
        print('name: ' + fieldName + ' - type: ' + fieldataType + ' - width: ' + str(fieldWidth) + ' - precision: ' + str(GetPrecision))

def getTiles(folder1, folder2=None, pattern1='.tif$', pattern2='.tif$', operator='difference', prefix1='_', suffix1=None, prefix2='_', suffix2=None):
    tiles = listFiles(folder1, pattern1, fullPath=False, extension=False)
    if prefix1 is not None:
        tiles = [file[file.find(prefix1) + 1:] for file in tiles]
    if suffix1 is not None:
        tiles = [file[:file.rfind(suffix1)] for file in tiles]
    if folder2 is not None:
        filt = listFiles(folder2, pattern2, fullPath=False, extension=False)
        if prefix2 is not None:
            filt = [file[file.find(prefix2) + 1:] for file in filt]
        if suffix2 is not None:
            filt = [file[:file.rfind(suffix2)] for file in filt]
        if operator == 'difference': 
            tiles = list(set(tiles) - set(filt))
        if operator == 'intersection':
            tiles = list(set(tiles) & set(filt))
    tiles.sort()
    print(str(len(tiles)) + ' tiles')
    return(tiles)

#%% LOG

def entry(tile, wks, seg, var):
    return('%s;%d;%d;%d\n' % (tile, wks, seg, var))

def log_init(logPath, tiles):
    open(logPath, 'w').close()
    entries = ['tile;computer;segment;variables\n']
    entries.extend([entry(tile, 0, 0, 0) for tile in tiles])
    with open(logPath, 'a') as log:
        log.write(''.join(map(str, entries)))

def log_processed(logPath, tile, var = 'computer'):
    log = pandas.read_csv(logPath, sep = ';')
    return(log.loc[log.tile == tile, var].iloc[0] != 0)

def log_update(logPath, tile, wks = 0, seg = 0, var = 0):
    with open(logPath, 'r') as log:
        entries = log.read()
    if wks != 0:
        entries = entries.replace(entry(tile, 0, 0, 0), entry(tile, wks, 0, 0))
    if seg == 1:
        entries = entries.replace(entry(tile, wks, 0, 0), entry(tile, wks, 1, 0))
    if var == 1:
        entries = entries.replace(entry(tile, wks, 1, 0), entry(tile, wks, 1, 1))
    with open(logPath, 'w') as log:
        log.write(entries)

#%% SPATIAL

# Solidity
def solidity(srcSegPath, srcHulPath, varName='FID'):
    driver  = ogr.GetDriverByName('ESRI Shapefile')
    srcVec  = driver.Open(srcSegPath, 1)
    srcHul  = driver.Open(srcHulPath, 1)
    vecLay  = srcVec.GetLayer()
    hulLay  = srcHul.GetLayer()
    vecArea = []
    hulArea = []
    fid     = []
    for feat in vecLay:
      vecArea.append(feat.GetGeometryRef().GetArea())
      fid.append(feat.GetField(varName))
    for feat in hulLay:
      hulArea.append(feat.GetGeometryRef().GetArea())
    vecArea = numpy.bincount(fid, weights=vecArea)
    hulArea = numpy.bincount(fid, weights=hulArea)
    numpy.seterr(divide='ignore', invalid='ignore')
    solidity = numpy.divide(vecArea, hulArea)
    solidity = solidity[numpy.argsort(fid)]
    return(solidity)

# Smoothness
def smoothness(srcSegPath, srcHulPath, varName = 'IDS'):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    srcVec = driver.Open(srcSegPath, 1)
    srcHul = driver.Open(srcHulPath, 1)
    vecLay = srcVec.GetLayer()
    hulLay = srcHul.GetLayer()
    vecLen = []
    fid    = []
    for feat in vecLay:
      vecLen.append(feat.GetGeometryRef().Boundary().Length())
      fid.append(feat.GetField(varName))
    hulLen = []
    for feat in hulLay:
      hulLen.append(feat.GetGeometryRef().Boundary().Length())
    vecLen = numpy.bincount(fid, weights = vecLen)
    hulLen = numpy.bincount(fid, weights = hulLen)
    numpy.seterr(divide='ignore', invalid='ignore')
    smoothness = numpy.divide(hulLen, vecLen)
    smoothness = smoothness[numpy.argsort(fid)]
    return(smoothness)

# Regularity
def regularity(srcRstPath):
    srcRst = raster2array(srcRstPath)
    xy     = raster2xy(srcRstPath).reshape(srcRst.shape[0], srcRst.shape[1], 2, order='F')
    xySd   = summarise(xy, srcRst, scipy.ndimage.standard_deviation)
    minSd  = xySd.min(axis=1)
    maxSd  = xySd.max(axis=1)
    regularity = numpy.sqrt(numpy.divide(minSd, maxSd, out=numpy.ones_like(minSd), where=maxSd != 0)) # Check that zero
    return(regularity)

# Size
def frequency(srcRstPath):
    # import numpy
    srcRst = raster2array(srcRstPath)
    size   = numpy.unique(srcRst, return_counts=True)
    return(size)

# Buffer
def buffer(srcVecPath, outVecPath, bufferDist, layerName='buffer', driver='ESRI Shapefile'):
    # import ogr, os
    vecDrv = ogr.GetDriverByName(driver)
    srcVec = ogr.Open(srcVecPath)
    srcLay = srcVec.GetLayer()
    if os.path.exists(outVecPath):
        vecDrv.DeleteDataSource(outVecPath)
    outVec = vecDrv.CreateDataSource(outVecPath)
    outLay = outVec.CreateLayer(layerName, geom_type=ogr.wkbPolygon, srs=srcLay.GetSpatialRef())
    # Creates fields in output layer
    layDef  = srcLay.GetLayerDefn()
    fieDefs = [layDef.GetFieldDefn(i) for i in range(layDef.GetFieldCount())]
    for fieDef in fieDefs:
        outLay.CreateField(fieDef)
    # Iterates over features
    for srcFeat in srcLay:
        srcGeom = srcFeat.GetGeometryRef()
        outGeom = srcGeom.Buffer(bufferDist)
        outFeat = ogr.Feature(layDef)
        outFeat.SetGeometry(outGeom)
        # Adds values for each field
        for fieDef in fieDefs:
            fieName = fieDef.GetNameRef()
            outFeat.SetField(fieName, srcFeat.GetField(fieName))
        # Needs to be at the end
        outLay.CreateFeature(outFeat)
    outVec = None
    
# Convex hulls
def convexHulls(srcVecPath, outVecPath, layerName='hulls', driver='ESRI Shapefile'):
    vecDrv = ogr.GetDriverByName(driver)
    srcVec = ogr.Open(srcVecPath)
    srcLay = srcVec.GetLayer()
    if os.path.exists(outVecPath):
        vecDrv.DeleteDataSource(outVecPath)
    outVec  = vecDrv.CreateDataSource(outVecPath)
    outLay  = outVec.CreateLayer(layerName, geom_type=ogr.wkbPolygon, srs=srcLay.GetSpatialRef())
    layDef  = srcLay.GetLayerDefn()
    fieDefs = [layDef.GetFieldDefn(i) for i in range(layDef.GetFieldCount())]
    for fieDef in fieDefs:
        outLay.CreateField(fieDef)
    for srcFeat in srcLay:
        srcGeom = srcFeat.GetGeometryRef()
        outGeom = srcGeom.ConvexHull()
        outFeat = ogr.Feature(layDef)
        outFeat.SetGeometry(outGeom)
        # Adds values for each field
        for fieDef in fieDefs:
            fieName = fieDef.GetNameRef()
            outFeat.SetField(fieName, srcFeat.GetField(fieName))
        outLay.CreateFeature(outFeat)
    outDatSrc = None

def rasterBbox(srcRstPath):
    # import gdal, ogr
    srcRst = gdal.Open(srcRstPath)
    xmin, xres, xskew, ymax, yskew, yres = srcRst.GetGeoTransform()
    xsize = srcRst.RasterXSize
    ysize = srcRst.RasterYSize
    xmax  = xmin + xsize * xres
    ymin  = ymax + ysize * yres
    geom  = ogr.Geometry(ogr.wkbLinearRing)
    geom.AddPoint(xmin, ymax)
    geom.AddPoint(xmin, ymin)
    geom.AddPoint(xmax, ymax)
    geom.AddPoint(xmax, ymin)
    geom.AddPoint(xmin, ymax)
    bbox  = ogr.Geometry(ogr.wkbPolygon)
    bbox.AddGeometry(geom)
    return(bbox)

def gdalSrs(srcRstpath):
    # import gdal, osr
    srcRst = gdal.Open(srcRstpath)
    srcSrs = srcRst.GetProjection()
    wkt2proj4 = osr.SpatialReference() 
    wkt2proj4.ImportFromWkt(srcSrs)
    outSrs = wkt2proj4.ExportToProj4()
    return(outSrs)

def border(array, class_value=1, kernel_size=3):
    subset = numpy.isin(array, class_value).astype(int)
    inner  = subset - skimage.morphology.erosion(subset, skimage.morphology.square(kernel_size))
    outer  = skimage.morphology.dilation(subset, skimage.morphology.square(kernel_size)) - subset
    inside = (subset - inner).astype(bool)
    border = (inner + outer).astype(bool)
    return(inside, border, inner, outer)

#%% Depreciated

# # Computes the intersection between geometries of two layers
# def ogrIntersection(srcVecPath1, srcVecPath2):
#     # Note: Not yet functional
#     # import ogr
#     driver  = ogr.GetDriverByName('ESRI Shapefile')
#     srcVec1 = driver.Open(srcVecPath1)
#     srcVec2 = driver.Open(srcVecPath2)
#     srcLay1 = srcVec1.GetLayer()
#     srcLay2 = srcVec2.GetLayer()
#     outArea = []
#     for feat1 in srcLay1:
#         geom1 = feat1.GetGeometryRef()
#         for feat2 in srcLay2:
#             geom2 = feat2.GetGeometryRef()
#             if geom2.Intersects(geom1):
#                 outArea.append(geom2.Intersection(geom1).GetArea())
#         srcLay2.reset_reading() # Have to declare it to iterate over features again
#     return(outArea)

# def gitlab_push(script):
#     # import os
#     # !git remote add arthisto 'https://gitlab.com/goclem/arthisto.git'
#     !git remote
#     os.chdir(paths.code)
#     !git add script
#     !git commit -m 'Auto update'
#     !git push arthisto