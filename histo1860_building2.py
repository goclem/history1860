#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Predicts buildings from Etat-Major maps (step 2)
@author: Clement Gorin
@contact: gorinclem@gmail.com
@date: April 2021
'''

#%% MODULES AND PATHS

# Modules
import argparse
import concurrent.futures
from operator import invert
import histo1860_functions as foo
import joblib
import itertools
import numpy as np
import os
import pandas as pd
import scipy
import shutil
import skimage
import skimage.color
import skimage.exposure
import skimage.feature
import skimage.morphology
import sklearn
import sklearn.ensemble
import sklearn.impute
import sklearn.model_selection
import time

from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr

# Paths and utilities
paths  = foo.makePaths('buildings')
tile   = '0880_6260'  # Testing only
X2vars = ['L_median', 'A_median', 'B_median', 'L_variance', 'A_variance', 'B_variance', 'texture1_median', 'texture2_median', 'texture3_median', 'texture4_median', 'texture1_variance', 'texture2_variance', 'texture3_variance', 'texture4_variance', 'size', 'regularity', 'solidity', 'smoothness', 'probability_median', 'probability_variance']

#%% FUNCTIONS

# Builds file paths
def makeFiles(tile):
    files = argparse.Namespace(
        scem = os.path.join(paths.scem, 'scem_{}.tif'),
        mask = os.path.join(paths.mask, 'mask_{}.tif'),
        X    = os.path.join(paths.X,    'X_{}.tif'),
        y    = os.path.join(paths.y,    'y_{}.tif'),
        yh1  = os.path.join(paths.yh1,  'yh1_{}.tif'),
        ids  = os.path.join(paths.tmp,  'tmp_{}_yh1.tif'),
        X2   = os.path.join(paths.X2,   'X2_{}.npy'),
        y2   = os.path.join(paths.y2,   'y2_{}.npy'),
        yh2  = os.path.join(paths.yh2,  'yh2_{}.tif'),
        yh3  = os.path.join(paths.yh3,  'yh3_{}.tif'),
        yh1p = os.path.join(paths.yh1,  'yh1p_{}.tif'),
        yh2p = os.path.join(paths.yh2,  'yh2p_{}.tif'),
        y2r  = os.path.join(paths.y2,   'y2_{}.tif'),
        idsv = os.path.join(paths.tmp,  'tmp_{}_yh1.shp'),
        hull = os.path.join(paths.tmp,  'tmp_{}_hull.shp'),
        yhfix = os.path.join(paths.tmp, 'yhfix_{}.tif'),
        tmp  = os.path.join(paths.tmp,   'tmp_{}')
    )
    files = dict((k, v.format(tile)) for k, v in vars(files).items())
    files = argparse.Namespace(**files)
    return(files)

# Computes identifiers and convex hull
def computeData(tile, hull=True):
    files = makeFiles(tile)
    foo.raster2vector(srcRstPath=files.yh1, outVecPath=files.idsv, driver='ESRI Shapefile')
    # Recode FID to avoid 0 indexing
    driver = ogr.GetDriverByName('ESRI Shapefile')
    srcVec = ogr.Open(files.idsv, 1)
    srcLay = srcVec.GetLayer()
    for feature in srcLay:
        feature.SetField('FID', feature.GetField('FID') + 1)
        srcLay.SetFeature(feature)
    srcVec.Destroy()
    foo.vector2raster(srcVecPath=files.idsv, srcRstPath=files.yh1, outRstPath=files.ids, burnField='FID', dataType=gdal.GDT_Int32)
    foo.convexHulls(srcVecPath=files.idsv, outVecPath=files.hull)

# Computes inputs
def computeX2(tile):
    start = time.time()
    files = makeFiles(tile)
    observation = foo.raster2array(files.ids).astype(int)
    if np.count_nonzero(observation) == 0: # Empty tiles
        return
    X = foo.raster2array(files.X)
    # Colour
    colour = skimage.color.rgb2lab(X)
    colour = np.concatenate((
        foo.summarise(colour, observation, scipy.ndimage.median), 
        foo.summarise(colour, observation, scipy.ndimage.variance)), 
        axis = 1)
    colour = np.delete(colour, 0, axis = 0)
    # Texture
    texture = list()
    gray    = skimage.color.rgb2gray(X)
    for argument in [(8, 1), (16, 2), (24, 3), (32, 4)]:
        texture.append(skimage.feature.local_binary_pattern(gray, *argument, 'ror'))
    texture = np.dstack(texture)
    texture = np.concatenate((
        foo.summarise(texture, observation, scipy.ndimage.median), 
        foo.summarise(texture, observation, scipy.ndimage.variance)), axis = 1)
    texture = np.delete(texture, 0, axis = 0)
    # Shape
    size       = foo.frequency(files.ids)[1]
    size       = np.delete(size, 0)
    regularity = foo.regularity(files.ids)
    regularity = np.delete(regularity, 0)
    solidity   = foo.solidity(files.idsv, files.hull, varName = 'FID')
    smoothness = foo.smoothness(files.idsv, files.hull, varName = 'FID')
    shape      = np.column_stack((size, regularity, solidity, smoothness))
    # Probability
    probability = foo.raster2array(files.yh1p)
    if os.path.exists(files.yhfix):
        fix = foo.raster2array(files.yhfix).astype(bool)
        fix = np.logical_and(fix, observation > 0)
        probability = np.where(fix, 1, probability)
    probability = probability.reshape(5000, 5000, 1)
    probability = np.concatenate((
        foo.summarise(probability, observation, scipy.ndimage.median), 
        foo.summarise(probability, observation, scipy.ndimage.variance)), axis = 1)
    probability = np.delete(probability, 0, axis = 0)
    # Aggregation
    X2 = np.column_stack((colour, texture, shape, probability))
    X2 = pd.DataFrame(X2, columns = X2vars)
    np.save(files.X2, X2)
    print('%.2f' % ((time.time() - start) / 60))

# Computes response
def computeY2(tile, args):
    print(tile)
    files = makeFiles(tile)
    # Computes mask
    mask = foo.raster2array(files.y)
    mask = np.where(mask > 0, 1, mask).astype(bool)
    # Recodes y
    y2   = foo.raster2array(files.y)
    y2   = np.where(np.isin(y2, [1, 2]), 1, y2)
    y2   = np.where(np.isin(y2, [3, 4, 5, 6, 7, 8]), 0, y2)
    y2   = np.where(np.invert(mask), -1, y2)
    # Computes observations
    yh1  = foo.raster2array(files.ids).astype(int)
    idx  = np.where(yh1 > 0, 1, yh1).astype(bool)
    tab  = pd.DataFrame({'yh1': yh1[idx], 'y2': y2[idx]})
    tab  = pd.crosstab(tab.yh1, tab.y2)
    tab  = tab.reindex(columns=[-1, 0, 1], fill_value=0)
    # Encodes response
    y2   = np.repeat(-1, tab.shape[0])
    tot  = np.sum(tab, axis = 1)
    cnd0 = (tab[1]  / tot) == 0   # If no building pixel
    cnd1 = (tab[1]  / tot) >= args.share
    cndn = (tab[-1] / tot) >= 0.1 # Buildings crossed by main roads
    # Observations with builtup areas below 10% are not used
    y2 = np.where(cnd0,  0, y2)
    y2 = np.where(cnd1,  1, y2)
    y2 = np.where(cndn, -1, y2)
    np.save(files.y2 , y2)
    if args.raster == True:
        y2r   = pd.Series(y2, tab.index)
        yh1   = foo.raster2array(files.ids).astype(int)
        mask0 = np.in1d(yh1, y2r[y2r ==  0].index).reshape(yh1.shape)
        mask1 = np.in1d(yh1, y2r[y2r ==  1].index).reshape(yh1.shape)
        maskn = np.in1d(yh1, y2r[y2r == -1].index).reshape(yh1.shape)
        y2r   = np.zeros((5000, 5000), dtype=int)
        y2r   = np.where(mask1, 1, y2r)
        y2r   = np.where(mask0, 2, y2r)
        y2r   = np.where(maskn, 3, y2r)
        foo.array2raster(y2r, files.y, files.y2r, noDataValue=0, dataType=gdal.GDT_Byte)

# Computes estimated response
def computeYh2(tile):
    print(tile)
    files = makeFiles(tile)
    ids   = foo.raster2array(files.ids)
    X2    = np.load(files.X2)
    # Imputes data
    imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X2)
    X2      = imputer.transform(X2)
    # Predicts probabilities
    yh2p = fh2.predict_proba(X2)
    # Reshapes results
    yh2p = np.append(0, yh2p[:,1])
    yh2p = yh2p[ids]
    yh2p = np.where(ids == 0, -1, yh2p)
    foo.array2raster(yh2p, files.ids, files.yh2p, noDataValue=-1, dataType=gdal.GDT_Float32)

# Computes borders
def computeBorder(array, class_value=1, kernel_size=3):
    subset = np.isin(array, class_value).astype(int)
    inner  = subset - skimage.morphology.erosion(subset, skimage.morphology.square(kernel_size))
    outer  = skimage.morphology.dilation(subset, skimage.morphology.square(kernel_size)) - subset
    inside = (subset - inner).astype(bool)
    border = (inner + outer).astype(bool)
    return(inside, border, inner, outer)

# Computes post-processing
def postProcessYh2(tile, args):
    print(tile)
    files = makeFiles(tile)
    # Data
    yh2p = foo.raster2array(files.yh2p)
    yh2  = yh2p > args.threshold
    # Fixes predictions
    if args.augment == True and (os.path.exists(files.y) or os.path.exists(files.yhfix)):
        # Aggregate fixes
        fix = np.zeros((5000, 5000), dtype=bool)
        if os.path.exists(files.yhfix):
            tmp = foo.raster2array(files.yhfix)
            fix = np.logical_or(fix, tmp) 
        if os.path.exists(files.y):
            tmp = foo.raster2array(files.y)
            tmp = np.logical_and(np.isin(tmp, [1, 2]), yh2p > 0) # IGN mistakes
            fix = np.logical_or(fix, tmp)
        del(tmp)
        # Computes categories
        inside, _, inner, _ = computeBorder(fix, True)
        # Fixes inside
        inside = np.where(inside) # Low probabilities are not fixed
        index  = np.arange(len(inside[0]))
        sample = np.random.choice(index, size=int(0.95 * len(index)), replace=False)
        yh2[inside[0][sample],inside[1][sample]] = 1
        # Fixes inner border
        inner  = np.where(inner) # Low probabilities are not fixed
        index  = np.arange(len(inner[0]))
        sample = np.random.choice(index, size=int(0.95 * len(index)), replace=False)
        yh2[inner[0][sample],inner[1][sample]] = 1
        # Fills holes and removes small aggregates
    if args.morphology:
        yh2 = yh2.astype(bool)
        yh2 = skimage.morphology.remove_small_holes(yh2, 25)
        yh2 = skimage.morphology.remove_small_objects(yh2, 5)
    foo.array2raster(yh2, files.yh2p, files.yh2, noDataValue=0, dataType=gdal.GDT_Byte)

# Training tiles identifiers
def getOHStiles():
    ohsDS  = ogr.GetDriverByName('GPKG').Open(os.path.join(paths.tile, 'ohs_tiles.gpkg'), 0)
    ohsLay = ohsDS.GetLayer()
    tiles  = [feature.GetField('tile') for feature in ohsLay]
    return(tiles)

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

# Aggregates all tiles
def aggregateTiles():
    # Files
    files = argparse.Namespace(
        vrt = os.path.join(paths.tmp,  'buildings1860.vrt'),
        ref = os.path.join(paths.base, 'data_project', 'ca.tif'),
        out = os.path.join(paths.data, 'buildings1860.tif'))
    # Source
    opt = gdal.BuildVRTOptions(allowProjectionDifference=True)
    vrt = gdal.BuildVRT(files.vrt, foo.listFiles(paths.tmp, pattern='agg.tif$'), options = opt)
    vrt = None
    src = gdal.Open(files.vrt)
    ref = gdal.Open(files.ref)
    # Output
    driver = gdal.GetDriverByName('GTiff')
    out    = driver.Create(files.out, ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Float32)
    out.SetGeoTransform(ref.GetGeoTransform())
    out.SetProjection(ref.GetProjection())
    # Reproject
    print('Reprojecting')
    gdal.ReprojectImage(src, out, src.GetProjection(), ref.GetProjection(), gdalconst.GRA_Average)
    del out
    # Formats output
    out = foo.raster2array(files.out)
    ref = foo.raster2array(files.ref)
    out[ref < 0] = -1
    foo.array2raster(out, files.out, files.out, noDataValue=-1, dataType=gdal.GDT_Float32)

#%% COMPUTES TRAINING SAMPLES

# Resets temporary folder
# foo.resetFoldConfirm(paths.tmp, remove=True)

# Computes data
tiles  = foo.getTiles(paths.yh1, paths.tmp, pattern1='yh1_', pattern2='^tmp_.*_yh1.shp$', suffix2='_', operator='difference')
trials = ((computeData, tile) for tile in tiles)
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    executor.map(lambda trial: safetyWrapper(*trial), trials)

# Computes input (! time)
tiles  = foo.getTiles(paths.tmp, paths.X2, pattern1='^tmp_.*_yh1.shp$', pattern2='.npy$', suffix1='_', operator='difference')
trials = ((computeX2, tile) for tile in tiles)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(lambda trial: safetyWrapper(*trial), trials)

# Fixes some overlapping geometries du to gdal.Polygonize hole problem (geometry in shapefile but not in raster)
""" 
def fix_geometries(tile):
    files  = makeFiles(tile)
    missid = foo.frequency(files.ids)[0]
    missid = list(set(np.arange(np.max(missid)+1)) - set(missid))
    print('tile: {} - missid: {}'.format(tile, missid))

for tile in tiles:
    fix_geometries(tile)
 """
# Computes response
tiles  = foo.getTiles(paths.y, paths.y2, pattern1='^ybuild.*tif$', pattern2='.npy$') # Excludes fixes
tiles  = getOHStiles() # Quick metrics
args   = argparse.Namespace(share=.5, raster=False)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(computeY2, tiles, itertools.repeat(args))

#%% FORMATS TRAINING SAMPLES

# 0160_6760 tile has no response islands lost in the sea without buildings
tiles = foo.getTiles(paths.y2, paths.X2, pattern1='npy$', pattern2='npy$', operator='intersection')
X2    = np.vstack([np.load(os.path.join(paths.X2, 'X2_{}.npy'.format(tile))) for tile in tiles])
y2    = np.hstack([np.load(os.path.join(paths.y2, 'y2_{}.npy'.format(tile))) for tile in tiles])

# Drops observations with undefined response
idx    = np.where(y2 == -1)
ytrain = np.delete(y2, idx)
Xtrain = np.delete(X2, idx, axis = 0)
del(tiles, X2, y2, idx)

# Saves training data
np.save(os.path.join(paths.fh2, 'ytrain.npy'), ytrain)
np.save(os.path.join(paths.fh2, 'Xtrain.npy'), Xtrain)

#%% FITS RANDOM FOREST

# Loads training data
ytrain = np.load(os.path.join(paths.fh2, 'ytrain.npy'))
Xtrain = np.load(os.path.join(paths.fh2, 'Xtrain.npy'))

# Impute a few missing values
np.sum(np.isnan(Xtrain))
imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(Xtrain)
Xtrain  = imputer.transform(Xtrain)

# Estimates model
start = time.time() # 16 mins
args  = argparse.Namespace(n_estimators=None, criterion='gini', max_features='sqrt', bootstrap=True, oob_score=True, n_jobs=-1, random_state=1, verbose=1, warm_start=True)
fh2   = sklearn.ensemble.RandomForestClassifier(**vars(args))
oobs  = list()

for n_estimators in np.arange(100, 600, 100, dtype=int):
    fh2.set_params(n_estimators=n_estimators)
    fh2.fit(Xtrain, ytrain)
    print('Trees: {} - OOB: {}'.format(n_estimators, fh2.oob_score_))
    oobs.append(fh2.oob_score_)

joblib.dump(fh2, os.path.join(paths.fh2, 'fh2.sav'))
print('%.2f' % ((time.time() - start) / 60))
del(start, args, Xtrain, ytrain)

# Diagnostics
fh2.oob_score_
importance = pd.Series(fh2.feature_importances_, index=X2vars)
importance = importance.sort_values(ascending=True)
importance.plot(kind='barh', figsize=(10, 10))
del(importance)

#%% PREDICTS TILES
# Rerun all tiles with the 25% threshold

# Loads model
fh2 = joblib.load(os.path.join(paths.fh2, 'fh2.sav'))
fh2.set_params(n_jobs=4, verbose=0)

# Computes estimated response
# ['0260_6900', '0280_6660'] are empty tiles
tiles = foo.getTiles(paths.X2, paths.yh2, pattern1='npy$', pattern2='yh2p_.*tif$', operator='difference')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(computeYh2, tiles)

#%% POST PROCESSES TILES

tiles  = foo.getTiles(paths.yh2, paths.yh2, pattern1='^yh2p_.*tif$', pattern2='^yh2_.*tif$', operator='difference')
cities = pd.read_csv(os.path.join(paths.data, 'buildings', 'fixes', 'cities.csv'))
tiles  = list(cities.tile)
args   = argparse.Namespace(threshold=0.25, augment=True, morphology=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(postProcessYh2, tiles, itertools.repeat(args))
del(tiles, args)

#%% AGGREGATION
def noData(tile):
    print(tile)
    files = makeFiles(tile)
    yh2   = foo.raster2array(files.yh2)
    foo.array2raster(yh2, files.yh2, files.tmp + "_agg.tif", noDataValue=-1, dataType=gdal.GDT_Byte)

tiles = foo.getTiles(paths.yh2, pattern1='^yh2_.*tif$')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(noData, tiles)

tiles = foo.getTiles(paths.tmp, pattern1='agg')
aggregateTiles()

