#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Prepares data for the Arthisto1860 project
@author: Clement Gorin
@contact: gorinclem@gmail.com
@date: February 2021
'''

#%% MODULES

# Modules
import argparse
import concurrent.futures
import functools
import cv2
import histo1860_functions as foo
import skimage.exposure
import pandas as pd
import numpy as np
import os

from osgeo import gdal
from osgeo import ogr

# Paths
paths = foo.makePaths('buildings')
tile  = '0880_6260'

#%% FUNCTIONS

# Builds file paths
def makeFiles(tile, label=''):
    files = argparse.Namespace(
        scem      = os.path.join(paths.scem, 'scem_{}.tif'),
        mask      = os.path.join(paths.mask, 'mask_{}.tif'),
        X         = os.path.join(paths.X,    'X_{}.tif'),
        y         = os.path.join(paths.y,    'y_{}.tif'),
        tmp       = os.path.join(paths.tmp, label + '_{}.tif'),
        ybloc     = os.path.join(paths.tmp,  'ybloc_{}.tif'),
        ybuilding = os.path.join(paths.tmp,  'ybuilding_{}.tif'),
        ylanduse  = os.path.join(paths.tmp,  'ylanduse_{}.tif'),
        yroad     = os.path.join(paths.tmp,  'yroad_{}.tif'),
        yrail     = os.path.join(paths.tmp,  'yrail_{}.tif'),
        yriver    = os.path.join(paths.tmp,  'yriver_{}.tif')
    )
    files = dict((k, v.format(tile)) for k, v in vars(files).items())
    files = argparse.Namespace(**files)
    return(files)

# Computes training rasters
def computeRasters(srcPath, tiles, label, value=None, dataType=gdal.GDT_Byte):
    # Loads shapefile (memory)
    srcDs   = ogr.GetDriverByName('GPKG').Open(srcPath, 0)
    memDs   = ogr.GetDriverByName('Memory').CreateDataSource('memDs')
    memDs.CopyLayer(srcDs.GetLayer(), 'memLay', ['OVERWRITE=YES'])
    memLay  = memDs.GetLayer('memLay')
    # Converts classes
    memLay.CreateField(ogr.FieldDefn('y', ogr.OFTReal))
    if value is not None:
        for feature in memLay:
            feature.SetField('y', value)
            memLay.SetFeature(feature)
    else: # (!) time
        for feature in memLay:
            # print(feature.GetFID()) # Debug only
            clsID = ohs2cls.loc[ohs2cls.ohs_id == feature.GetField('THEME')].iloc[0].cls_id
            feature.SetField('y', clsID)
            memLay.SetFeature(feature)
    # Exports raster
    for tile in tiles:
        files   = makeFiles(tile, label)
        srcRst  = gdal.Open(files.X, gdal.GA_ReadOnly)
        outPath = files.tmp
        outRst  = gdal.GetDriverByName('GTiff').Create(outPath, srcRst.RasterXSize, srcRst.RasterYSize, 1, dataType)
        outRst.SetGeoTransform(srcRst.GetGeoTransform())
        outRst.SetProjection(srcRst.GetProjection())
        outBnd  = outRst.GetRasterBand(1)
        outBnd.Fill(0)
        outBnd.SetNoDataValue(0)
        gdal.RasterizeLayer(outRst, [1], memLay, options = ['ATTRIBUTE=y', 'ALL_TOUCHED=FALSE'])
        outRst = outBnd = None
        # Removes empty raster
        if np.count_nonzero(foo.raster2array(outPath)) == 0:
            os.remove(outPath)

# Computes masks
def computeMasks(tile):
    files   = makeFiles(tile)
    srcRst  = gdal.Open(files.scem, gdal.GA_ReadOnly)
    outRst  = gdal.GetDriverByName('GTiff').Create(files.mask, srcRst.RasterXSize, srcRst.RasterYSize, 1, gdal.GDT_Byte)
    outRst.SetGeoTransform(srcRst.GetGeoTransform())
    outRst.SetProjection(srcRst.GetProjection())
    outBnd  = outRst.GetRasterBand(1)
    outBnd.Fill(0)
    outBnd.SetNoDataValue(0)
    gdal.RasterizeLayer(outRst, [1], memLay, burn_values = [1])
    outRst = outBnd = None
    # Removes empty rasters
    if np.count_nonzero(foo.raster2array(outPath)) == 0:
        os.remove(outFile)

# Merges training classes
def mergeRasters(tile):
    files  = makeFiles(tile)
    ybase  = np.zeros((5000, 5000)).astype(np.int)
    if os.path.exists(files.yland):
        ylanduse = foo.raster2array(files.ylanduse)
        ybase[ylanduse != 0] = ylanduse[ylanduse != 0]
        del(ylanduse)
    if os.path.exists(files.yriver):
        yriver = foo.raster2array(files.yriver)
        ybase[yriver != 0] = 8
        del(yriver)
    if os.path.exists(files.yroad):
        yroad = foo.raster2array(files.yroad)
        ybase[yroad != 0] = 0
        del(yroad)
    if os.path.exists(files.yrail):
        yrail = foo.raster2array(files.yrail)
        ybase[yrail != 0] = 0
        del(yrail)
    if os.path.exists(files.ybuild):
        ybuilding = foo.raster2array(files.ybuilding)
        ybase[ybuilding != 0] = 1
        del(ybuilding)
    if os.path.exists(files.ybloc):
        ybloc = foo.raster2array(files.ybloc)
        ybase[ybloc != 0] = 2
        del(ybloc)
    # Removes empty rasters
    if np.count_nonzero(ybase) > 0:
        foo.array2raster(ybase, files.scem, files.y, noDataValue=0, dataType = gdal.GDT_Byte)

# Contrast-enhancement
def clahe(X):
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (10, 10))
    X_clahe = []
    for channel in cv2.split(X):
        X_clahe.append(clahe.apply(channel))
    X_clahe = cv2.merge(X_clahe)
    return(X_clahe)

def maskArray(array, mask, maskValue = 0, updateValue = 0):
    mask   = mask.flatten()
    masked = array.reshape(np.prod(array.shape[:2]), 1, -1)
    masked[mask == maskValue, ...] = updateValue
    masked = masked.reshape(array.shape)
    return(masked)

# Pre-processes rasters
def preprocess(tile):
    files = makeFiles(tile)
    scem  = foo.raster2array(files.scem, dataType=np.uint8)
    X     = skimage.color.rgb2lab(scem)
    X     = skimage.exposure.rescale_intensity(X)
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (10, 10))
    X[...,0] = clahe.apply(X[...,0])
    X     = clahe(X)
    foo.array2raster(X, files.scem, files.X, dataType=gdal.GDT_Byte)
    
# Safety wrapper for eventual failure
def safetyWrapper(function, arguments):
    expression = function.__name__ + '(\'%s\')'%(arguments)
    print('Trying {} ... '.format(expression), end='')
    try:
        eval(expression)
        print(' success')
    except:
        print(' failure')
        pass

#%% FORMATS SCEM RASTERS
# Single file, byte type

tiles = foo.getTiles('/Volumes/IGN/Scan_EM_40K/1_DONNEES_LIVRAISON_2015-03-00276/SCEM40K_TIF_LAMB93_FRANCE', suffix1='_')

for tile in tiles:
    print(tile)
    srcFile = os.path.join('/Volumes/IGN/Scan_EM_40K/1_DONNEES_LIVRAISON_2015-03-00276/SCEM40K_TIF_LAMB93_FRANCE', 'SCEM40K_' + tile + '_L93.tif')
    outFile = os.path.join(paths.scem, 'scem_' + tile + '.tif')
    scem    = foo.raster2array(srcFile)
    foo.array2raster(scem, srcFile, outFile, dataType=gdal.GDT_Byte)

#%% COMPUTE TRAINING RASTERS

# Resets folder
# foo.resetFoldConfirm(paths.tmp, remove = True)

# Maps OHS classes to our classes
ohs2cls = pd.read_csv(os.path.join(paths.data, 'ohs2cls.csv'), ';', dtype = {'cls_id': np.float64}) # Shapefiles do not accept integers

# Lists training tiles
tmpDs   = ogr.GetDriverByName('GPKG').Open(os.path.join(paths.tile, 'ohs_tiles.gpkg'), 0)
tmpLay  = tmpDs.GetLayer()
tiles   = []
for feature in tmpLay:
    tiles.append(feature.GetField('tile'))
tiles = list(set(tiles) - set(foo.getTiles(paths.tmp, pattern1 = 'yrail')))

# Computes training rasters
computeRasters(os.path.join(paths.ohs, 'batiment_bloc.gpkg'), tiles, 'ybloc', 1)
computeRasters(os.path.join(paths.ohs, 'batiment.gpkg'), tiles, 'ybuilding', 1)
computeRasters(os.path.join(paths.ohs, 'troncon_de_route.gpkg'), tiles, 'yroad', 1)
computeRasters(os.path.join(paths.ohs, 'troncon_de_voie_ferree.gpkg'), tiles, 'yrail', 1)
computeRasters(os.path.join(paths.ohs, 'troncon_de_cours_d_eau.gpkg'), tiles, 'yriver', 1)
computeRasters(os.path.join(paths.ohs, 'limite_administrative.gpkg'), tiles, 'yborder', 1)
computeRasters(os.path.join(paths.ohs, 'ocs_ancien_sans_bati.gpkg'), tiles, 'ylanduse')

#%% COMPUTE PREDICTION FIXES

# Lists fixes tiles
cities = pd.read_csv(os.path.join(paths.data, 'buildings', 'fixes', 'cities.csv'))
tiles  = list(cities.tile)
computeRasters(os.path.join(paths.data, 'buildings', 'fixes', 'yh_fixes.gpkg'), tiles, 'yhfix', 1)

#%% MERGE TRAINING RASTERS

# Resets folder
# foo.resetFoldConfirm(paths.y, remove=True)

tiles = foo.getTiles(paths.tmp, pattern1='^y.*\\.tif$')
tiles = list(set(tiles)) # Unique values
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    executor.map(mergeRasters, tiles)
    
#%% MASKS

# Resets folder
# foo.resetFoldConfirm(paths.mask, remove = True)

# Loads vectors in memory
srcPath = paths.france
srcDs   = ogr.GetDriverByName('GPKG').Open(srcPath, 0)
memDs   = ogr.GetDriverByName('Memory').CreateDataSource('memDs')
memDs.CopyLayer(srcDs.GetLayer(), 'memLay', ['OVERWRITE=YES'])
memLay  = memDs.GetLayer('memLay')

# Computes masks
tiles = foo.getTiles(paths.scem, '.tif$')
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    executor.map(computeMasks, tiles)

#%% PREPROCESSING

# Resets folder
# foo.resetFoldConfirm(paths.X, remove = True)

tiles = foo.getTiles(paths.mask, '.tif$', paths.x, '.tif$')
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    executor.map(preprocess, tiles)

#%% POSTPROCESSING FOR YH1

cities  = pd.read_csv(os.path.join(paths.data, 'buildings', 'fixes', 'cities.csv'))
tiles   = list(cities.tile)
srcPath = os.path.join(paths.data, 'buildings', 'fixes', 'yh_fixes.gpkg')
computeRasters(srcPath, tiles, 'yh1fix', 1)

#%% FIX SINGLE TILE

srcPath = os.path.join(paths.data, 'buildings', 'fixes', 'yh_fixes.gpkg')
tiles   = ['0920_6900', '0900_6460']
computeRasters(srcPath, tiles, 'yhfix', 1)

def augment(tile):
    print(tile)
    files = argparse.Namespace(
        yh1   = os.path.join(paths.yh1, 'yh1_{}.tif'.format(tile)),
        yh2   = os.path.join(paths.yh2, 'yh2_{}.tif'.format(tile)),
        yh    = os.path.join(paths.data, 'landuse', 'yh', 'yh_{}.tif'.format(tile)),
        yhfix = os.path.join(paths.tmp, 'yhfix_{}.tif'.format(tile)))
    if os.path.exists(files.yhfix):
        fix = foo.raster2array(files.yhfix).astype(bool)
        fix = np.where(fix)
        yh1 = foo.raster2array(files.yh1)
        yh2 = foo.raster2array(files.yh2)
        yh  = foo.raster2array(files.yh)
        yh1[fix[0], fix[1]] = 1
        yh2[fix[0], fix[1]] = 1
        yh[fix[0],  fix[1]] = 1
        foo.array2raster(yh1, files.yh1, files.yh1, noDataValue=0, dataType=gdal.GDT_Byte)
        foo.array2raster(yh2, files.yh2, files.yh2, noDataValue=0, dataType=gdal.GDT_Byte)
        foo.array2raster(yh,  files.yh,  files.yh,  noDataValue=0, dataType=gdal.GDT_Byte)
        
for tile in tiles:
    augment(tile)
