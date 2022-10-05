#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Extracts land-use classes from Etat-Major historical maps
@author: Clement Gorin
@contact: gorinclem@gmail.com
@date: August 2021
'''

#%% Modules

import argparse
import concurrent.futures
import cv2
import histo1860_functions as foo
import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
import sklearn.cluster
import sklearn.impute
import sklearn.preprocessing
import skimage.exposure
import skimage.segmentation
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import subprocess
import time

from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr

# Paths
paths = foo.makePaths('landuse')
tile  = '0880_6260'
Xvars = [
    'id',
    'L_median', 'A_median', 'B_median', 
    'L_variance', 'A_variance', 'B_variance', 
    'texture1_median', 'texture2_median', 'texture3_median', 
    'texture4_median', 'texture1_variance', 'texture2_variance', 'texture3_variance', 'texture4_variance',
    'size', 'regularity', 'solidity', 'smoothness']
#%% FUNCTIONS

# Creates file paths
def makeFiles(tile):
    files = argparse.Namespace(
        scem = os.path.join(paths.scem, 'scem_{}.tif'),
        mask = os.path.join(paths.mask, 'mask_{}.tif'),
        seg  = os.path.join(paths.seg,  'seg_{}.tif'),
        yh2  = os.path.join(paths.data, 'buildings', 'yh2', 'yh2_{}.tif'),
        ids  = os.path.join(paths.ids,  'ids_{}.tif'),
        hull = os.path.join(paths.hull, 'hull_{}.shp'),
        idsv = os.path.join(paths.ids,  'ids_{}.shp'),
        X    = os.path.join(paths.X,    'X_{}.tif'),
        y    = os.path.join(paths.y,    'y_{}.tif'),
        Xdat = os.path.join(paths.Xdat, 'X_{}.pkl'),
        ydat = os.path.join(paths.ydat, 'y_{}.pkl'),
        yh   = os.path.join(paths.yh,   'yh_{}.tif'),
        yhv  = os.path.join(paths.yh,   'yh_{}.gpkg'),
        yhb  = os.path.join(paths.data, 'buildings', 'yh2', 'yh2_{}.tif')
    )
    files = dict((k, v.format(tile, tile)) for k, v in vars(files).items())
    files = argparse.Namespace(**files)        
    return(files)

# Creates temporary file
def makeTempFile(tile, tmplab="tmp", tmpext="tif"):
    folder = os.path.join(paths.tmp, tile)
    file   = os.path.join(folder, "{}_{}.{}".format(tmplab, tile, tmpext))
    return(folder, file)

# Segmentation
def segment(tile, args):
    files   = makeFiles(tile)
    X       = foo.raster2array(files.X)
    segment = skimage.segmentation.quickshift(X, ratio=args.ratio, kernel_size=args.kernel_size, max_dist=args.max_dist, sigma=args.sigma, convert2lab=args.convert2lab, random_seed=args.random_seed)
    foo.array2raster(segment, files.X, files.seg, dataType=args.dataType)

# Sieving
def sieve(tile, args):
    files = makeFiles(tile)
    foo.sieveRaster(files.seg, files.seg, threshold=args.threshold, connectedness=args.connectedness, dataType=args.dataType)

# Computes identifier vectors and convex hulls
def computeIdentifiers(tile, args):
    # args = argparse.Namespace(makeValid=True, dataType=gdal.GDT_UInt32)
    print(tile)
    files = makeFiles(tile)
    # Data
    segment = foo.raster2array(files.seg)
    mask    = foo.raster2array(files.mask).astype(bool)
    # Masks buildings and excluded pixels
    if os.path.exists(files.yh2):
        yh2  = foo.raster2array(files.yh2).astype(bool)
        mask = np.logical_or(np.invert(mask), yh2)
    ids = np.where(mask, 0, segment)
    if np.count_nonzero(ids) > 0:
    # Normalises identifiers
        rank = scipy.stats.rankdata(ids.flatten(), method='dense') - 1
        ids  = rank.reshape(ids.shape)
        # Saves identifier raster and shapefile
        tmpFold, tmpFile = makeTempFile(tile, "ids", "shp")
        foo.resetFolder(tmpFold)
        foo.array2raster(ids, files.seg, files.ids, noDataValue=0, dataType=args.dataType)
        foo.raster2vector(files.ids, tmpFile, fieldName='ids', dataType=ogr.OFTInteger, fieldIndex=0)
        # Converts polygons to multipolygons
        command = "ogr2ogr {} {} -dialect sqlite -sql \"SELECT ST_Union(geometry), ids FROM {} GROUP BY ids\""
        command = command.format(files.idsv, tmpFile, foo.fileName(tmpFile)) # Output, Input, Layer
        status  = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, error = status.communicate()
        # Validates geometries if error
        if error.decode('utf-8')[:10] == 'GEOS error':
            command = "Rscript histo1860_validate.R --srcVecPath={} --outVecPath={}"
            command = command.format(tmpFile, files.idsv)
            os.system(command)
        # Computes convex hulls
        foo.convexHulls(files.idsv, files.hull)
        foo.resetFolder(tmpFold, remove=True)

# Computes variables
def computeX(tile):
    print(tile)
    start = time.time()
    files = makeFiles(tile)
    # Data
    ids = foo.raster2array(files.ids).astype(int)
    X   = foo.raster2array(files.X)
    # Shape
    solidity    = foo.solidity(files.idsv, files.hull, 'ids')
    smoothness  = foo.smoothness(files.idsv, files.hull, 'ids')
    idvar, size = foo.frequency(files.ids)
    regularity  = foo.regularity(files.ids)
    # Colour
    colour = skimage.color.rgb2lab(X)
    colour = np.concatenate((
        foo.summarise(colour, ids, scipy.ndimage.median), 
        foo.summarise(colour, ids, scipy.ndimage.variance)), axis=1)
    # Texture
    texture = list()
    gray    = skimage.color.rgb2gray(X)
    for argument in [(8, 1), (16, 2), (24, 3), (32, 4)]:
        texture.append(skimage.feature.local_binary_pattern(gray, *argument, 'ror'))
    texture = np.dstack(texture)
    texture = np.concatenate((
        foo.summarise(texture, ids, scipy.ndimage.median), 
        foo.summarise(texture, ids, scipy.ndimage.variance)), axis=1)
    # Removes variables from the 0 pixels (computed from raster)
    if np.any(ids == 0):
        colour     = np.delete(colour, 0, axis=0)
        texture    = np.delete(texture, 0, axis=0)
        idvar      = np.delete(idvar, 0)
        size       = np.delete(size, 0)
        regularity = np.delete(regularity, 0)
    # Aggregation
    shape = np.column_stack((size, regularity, solidity, smoothness))
    Xdat  = np.column_stack((idvar, colour, texture, shape))
    Xdat  = pd.DataFrame(Xdat, columns=Xvars)
    Xdat.to_pickle(files.Xdat)
    print('%.2f' % ((time.time() - start) / 60))

def computeY(tile, args):
    # args  = argparse.Namespace(share=0.5, shareWater=0.25, undefined=1, raster=True)
    print(tile)
    files = makeFiles(tile)
    ids   = foo.raster2array(files.ids)
    y     = foo.raster2array(files.y)
    mask  = np.where(ids != 0)
    tab   = pd.DataFrame({'ids': ids[mask], 'y': y[mask]})
    tab   = pd.crosstab(tab.ids, tab.y)
    tab   = tab.apply(lambda row: row/row.sum(), axis=1)
    tab   = tab.reindex(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8], fill_value=0)
    y = np.where(tab.max(axis=1) > args.share, tab.idxmax(axis=1), args.undefined).astype(int)
    y = np.where(tab[8] > args.shareWater, 8, y)           # Lowers threshold for linear entities
    y = np.where(np.isin(y, [0, 1, 2]), args.undefined, y) # Removes buildings
    y = pd.DataFrame(y, index=tab.index, columns=['y'])
    y.to_pickle(files.ydat)
    if args.raster:
        outRst  = np.append(0, y.y)[ids]
        outFile = files.ydat.replace('pkl', 'tif')
        foo.array2raster(outRst, files.ids, outFile, noDataValue=0, dataType=gdal.GDT_Byte)
    
def computeYh(tile, args):
    print(tile)
    files = makeFiles(tile)
    # Data
    X   = pd.read_pickle(files.Xdat)
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    ids = foo.raster2array(files.ids)
    # Prediction
    imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    X   = imputer.fit_transform(X)
    yh  = fh.predict(X)
    yh  = np.append(0, yh)[ids]
    if args.addBuildings and os.path.exists(files.yhb):
        yhb = foo.raster2array(files.yhb)
        yh  = yh + yhb
    foo.array2raster(yh, files.ids, files.yh, dataType=gdal.GDT_Byte)
    
def subset(tile, args):
    print(tile)
    files = makeFiles(tile)
    yh = foo.raster2array(files.yh)
    yh = np.where(yh == args.value, 1, 0)
    tempFold, tempFile = makeTempFile(tile, args.label)
    foo.array2raster(yh, files.yh, tempFile, noDataValue=-1, dataType=gdal.GDT_Byte)

# Aggregates all tiles
def aggregateTiles(tiles, args):
    # Files
    files = argparse.Namespace(
        vrt = os.path.join(paths.tmp,  '{}1860.vrt'.format(args.label)),
        ref = os.path.join(paths.base, 'data_project', 'ca.tif'),
        out = os.path.join(paths.data, '{}1860.tif'.format(args.label)))
    # Source
    opt = gdal.BuildVRTOptions(allowProjectionDifference=True)
    vrt = [os.path.join(paths.tmp, tile, '{}_{}.tif'.format(args.label, tile)) for tile in tiles]
    vrt = gdal.BuildVRT(files.vrt, vrt, options = opt)
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

#%% COMPUTES SEGMENTATION

# Segmentation
tiles = foo.getTiles(paths.X, paths.seg)
# args = argparse.Namespace(ratio=2, kernel_size=10, max_dist=30, sigma=2, convert2lab=True, random_seed=1, dataType=gdal.GDT_UInt32) # Previous version
args = argparse.Namespace(ratio=1, kernel_size=5,  max_dist=10, sigma=1, convert2lab=True, random_seed=1, dataType=gdal.GDT_UInt32)
with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    executor.map(segment, tiles, itertools.repeat(args))
del(tiles, args)

# Sieving
tiles = foo.getTiles(paths.seg)
args  = argparse.Namespace(threshold=10, connectedness=4, dataType=gdal.GDT_UInt32)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(sieve, tiles, itertools.repeat(args))
del(tiles, args)

#%% COMPUTES IDENTIFIERS

# Identifiers
# 276 tiles with invalid geometries were fixed using R before creating the multipolygons and the convex hulls
tiles = foo.getTiles(paths.seg, paths.ids, pattern1='seg.*tif$', pattern2='ids.*shp$')
args  = argparse.Namespace(dataType=gdal.GDT_UInt32)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(computeIdentifiers, tiles, itertools.repeat(args))
del(tiles)

#%% COMPUTES INPUT

# Input (! time)
tiles  = foo.getTiles(paths.ids, paths.Xdat, pattern2='pkl$', operator='difference')
trials = ((computeX, tile) for tile in tiles)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(lambda trial: safetyWrapper(*trial), trials)
del(tiles, trials)

#%% COMPUTES RESPONSE

# foo.resetFolderConfirm(paths.ydat, remove=True)
tiles = foo.getTiles(paths.y, paths.ydat, pattern2='pkl$', operator='difference')
args  = argparse.Namespace(share=0.75, shareWater=0.25, undefined=1, raster=False)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(computeY, tiles, itertools.repeat(args))
del(tiles, args)

#%% FORMATS TRAINING SAMPLES

# Loads training samples
tiles  = foo.getTiles(paths.Xdat, paths.ydat, '.pkl$', '.pkl$', 'intersection')
ytrain = pd.concat([pd.read_pickle(makeFiles(tile).ydat) for tile in tiles])
ytrain = ytrain.y.to_numpy()

Xtrain = pd.concat([pd.read_pickle(makeFiles(tile).Xdat) for tile in tiles])
Xtrain = Xtrain.drop(['id'], axis=1)
Xtrain = Xtrain.to_numpy()

# Drops observations with undefined response (coded 1)
idx    = np.invert(np.isin(ytrain, 1))
Xtrain = Xtrain[idx]
ytrain = ytrain[idx]
del(tiles, idx)

# Eventually remaps classes
# ytrain = np.where(ytrain==5, 4, ytrain)

#%% FITS RANDOM FOREST

# Impute a few missing values
np.sum(np.isnan(Xtrain))
imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
Xtrain  = imputer.fit_transform(Xtrain)

# Estimation
start = time.time()
args  = argparse.Namespace(n_estimators=100, criterion='gini', max_features='sqrt', bootstrap=True, class_weight='balanced', oob_score=True, n_jobs=-1, random_state=1, verbose=1, warm_start=False)
fh    = sklearn.ensemble.RandomForestClassifier(**vars(args))
fh.fit(Xtrain, ytrain)
# joblib.dump(fh, os.path.join(paths.fh, 'fh.sav'))
print('%.2f' % ((time.time() - start) / 60))
del(start, args)

# Diagnostics
fh = joblib.load(os.path.join(paths.fh, 'fh.sav'))
fh.oob_score_
importance = pd.Series(fh.feature_importances_, index=Xvars[1:])
importance = importance.sort_values(ascending=True)
importance.plot(kind='barh', figsize=(10, 10))
del(importance)

#%% PREDICTS TILES

# Loads model
# fh = joblib.load(os.path.join(paths.fh, 'fh.sav'))
fh.set_params(verbose=0)

tiles = foo.getTiles(paths.Xdat, pattern1='.pkl$')
args  = argparse.Namespace(addBuildings=True)
for tile in tiles:
    computeYh(tile, args)

#%% AGGREGATE CLASSES AND TILES

# Creates temporary folders
tiles = foo.getTiles(paths.yh)
for tile in tiles:
    print(tile)
    tmpFold, tempFile = makeTempFile(tile)
    foo.resetFolder(tmpFold, remove=True)

# Aggregate tiles
argslist = [
    argparse.Namespace(value=1, label='buildings'),
    argparse.Namespace(value=3, label='crops'),
    argparse.Namespace(value=4, label='meadows'),
    argparse.Namespace(value=5, label='pastures'),
    argparse.Namespace(value=6, label='specialised'),
    argparse.Namespace(value=7, label='forests'),
    argparse.Namespace(value=8, label='water')]

for args in argslist:
    print(args)
    for tile in tiles:
        subset(tile, args)
    aggregateTiles(tiles, args)
    
#%% VECTORISES PREDICTIONS

tiles = foo.getTiles(paths.yh)

for tile in tiles:
    print(tile)
    files = makeFiles(tile)
    foo.raster2vector(files.yh, files.yhv, driver='GPKG', layerName='vectorised', fieldName='class', dataType=ogr.OFTInteger, fieldIndex=0)
    # tempFold, tempFile = makeTempFile(tile, tmplab='valid', tmpext='gpkg')
    # command = 'ogr2ogr -f \"GPKG\" {} {} -dialect sqlite -sql \"select ST_MakeValid(geom) as geom, * from vectorised\"'.format(tempFile, files.yhv)
    # os.system(command)
    
#%% NOTES

# QUICKSHIFT
# - CHARACTERISTICS
# - Superpixels of different sizes
# - Precise
# - Slow
# - ARG: RATIO
#   - Range [0.5 - 2]
#   - Larger values increases the number of segments and gives a more precise segmentation
#   - Ratio = 0, superpixels do not capture any pattern
#   - Ratio > 2, superpixels seem too precise
#   - Increasing ratio ? computing time
# - KERNEL SIZE
#   - Range [3 - 10]
#   - Smaller values capture more detail but creates *many* more superpixels!
#   - Larger values allow for superpixels (i.e. max_dist)
#   - Increasing kernel_size increases computing time
# - ARG: MAX DIST
#   - Range [10 - 30]
#   - Controls the maximum size of the superpixels
#   - Seem to merge fields without decrease in accuracy on buildings
#   - Increasing max_dist increases computing time
#   - max_dist = 20 give good results
# - ARG: SIGMA
#   - Range []
#   - Larger values simplify segmentation by removing noisy edges
#   - Increasing sigma increases computing time
#   - 5 doesn't capture any pattern
#  - PREFERED
#   - Ratio = 2: Precise segmentation
#   - Kernel = 9: Allow for superpixels in large entities
#   - Distance = 30: Allow for large superpixels
#   - Sigma = 2: Removes noisy edges

#%% DEPRECIATED

"""
# Temporary converts numpy arrays to pandas 
tiles = foo.getTiles(paths.Xdat, pattern1='pkl.npy$',  suffix1='.pkl')
for tile in tiles:
    print(tile)
    tmp = np.load(os.path.join(paths.Xdat, 'X_{}.pkl.npy'.format(tile)))
    tmp = pd.DataFrame(tmp, index=(tmp[:,0]).astype(int), columns=Xvars)
    tmp.index.name = 'ids'
    tmp.pop('id')
    tmp.to_pickle(os.path.join(paths.Xdat, 'X_{}.pkl'.format(tile)))
"""
  
"""
# Fix invalid geometries in R
pacman::p_load(sf, dplyr, future.apply)
tiles <- dir("F:/cgorin/tmp")
plan(multiprocess, workers = availableCores() - 1)

future_lapply(tiles, function(tile) {
  srcFile <- sprintf("F:/cgorin/tmp/%s/ids_%s.shp", tile, tile)
  outFile <- sprintf("F:/cgorin/Dropbox/research/arthisto/data_1860/landuse/ids/ids_%s.shp", tile)
  geoms   <- st_read(srcFile, quiet = T)
  geoms   <- st_make_valid(geoms)
  geoms   <- geoms %>% group_by(ids) %>% summarise(.groups = "drop")
  st_write(geoms, outFile, delete_dsn = T, quiet = T)
})
"""  

"""
oobs  = list()
for n_estimators in np.arange(100, 200, 100, dtype=int):
    fh.set_params(n_estimators=n_estimators)
    fh.fit(Xtrain, ytrain)
    print('Trees: {} - OOB: {}'.format(n_estimators, fh.oob_score_))
    oobs.append(fh.oob_score_)
"""

"""
# Tuning
from sklearn.model_selection import RandomizedSearchCV

grid = {
    'criterion':        ['gini', 'entropy'],
    'n_estimators':     [100, 500, 1000, 1500, 2000],
    'max_features':     ['auto', 'sqrt', 'log2', None],
    'bootstrap':        [True, False],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'class_weight':     ['balanced', 'balanced_subsample', None]
    }

search = RandomizedSearchCV(sklearn.ensembles.RandomForestClassifier(), param_distributions=grid, n_iter=50, cv=3, n_jobs=-1, refit=1)
search.fit(X, y)

search.best_params_
fh = random_search.best_estimator_
rf_mod.set_params(**params)
print(rf_mod.get_params())

pd.DataFrame(random_search.cv_results_).loc[:, :'std_score_time'].round(4)
rf_tune = pd.DataFrame(random_search.cv_results_['params'])
rf_tune.to_csv(tunePath, sep = ',')
"""

"""
# Optimises segmentation
args0 = argparse.Namespace(ratio=1, kernel_size=5,  max_dist=10, sigma=0, convert2lab=True, random_seed=1) # Default
args1 = argparse.Namespace(ratio=2, kernel_size=10, max_dist=30, sigma=2, convert2lab=True, random_seed=1) # Current
args2 = argparse.Namespace(ratio=1, kernel_size=10, max_dist=10, sigma=0, convert2lab=True, random_seed=1) # New
args3 = argparse.Namespace(ratio=2, kernel_size=5,  max_dist=10, sigma=1, convert2lab=True, random_seed=1) # New
args4 = argparse.Namespace(ratio=2, kernel_size=5,  max_dist=20, sigma=1, convert2lab=True, random_seed=1) # New
args5 = argparse.Namespace(ratio=1, kernel_size=5,  max_dist=10, sigma=1, convert2lab=True, random_seed=1) # Default

args  = args5
argid = 5
files = argparse.Namespace(
    X      ='/Users/clementgorin/Dropbox/research/arthisto/data_1860/X/X_0560_6280.tif',
    segment='/Users/clementgorin/Desktop/segment{}.tif'.format(argid),
    sieve  ='/Users/clementgorin/Desktop/sieve{}.tif'.format(argid),
    vector ='/Users/clementgorin/Desktop/vector{}.shp'.format(argid)
)

X     = foo.raster2array(files.X)
start = time.time()
segment = skimage.segmentation.quickshift(X, ratio=args.ratio, kernel_size=args.kernel_size, max_dist=args.max_dist, sigma=args.sigma, convert2lab=args.convert2lab, random_seed=args.random_seed)
print('%.2f' % ((time.time() - start) / 60))
foo.array2raster(segment, files.X, files.segment, dataType=gdal.GDT_UInt32)
foo.sieveRaster(files.segment, files.sieve, threshold=12, connectedness=4, dataType=gdal.GDT_UInt32)
foo.raster2vector(files.sieve, files.vector)
"""

"""
def computeIdentifiers(tile, args):
    files = makeFiles(tile)
    foo.convexHulls(files.idsv, files.hull)
    

def computeIdentifiers(tile, args):
    print(tile)
    files = makeFiles(tile)
    # Data
    segment = foo.raster2array(files.seg)
    mask    = foo.raster2array(files.mask).astype(bool)
    # Masks buildings and excluded pixels
    if os.path.exists(files.yh2):
        yh2  = foo.raster2array(files.yh2).astype(bool)
        mask = np.logical_or(np.invert(mask), yh2)
    ids = np.where(mask, 0, segment)
    if np.count_nonzero(ids) > 0:
    # Normalises identifiers
        rank = scipy.stats.rankdata(ids.flatten(), method='dense') - 1
        ids  = rank.reshape(ids.shape)
        # Saves identifier raster and shapefile
        tmpFold, tmpFile = makeTempFile(tile, "ids", "shp")
        foo.resetFolder(tmpFold)
        foo.array2raster(ids, files.seg, files.ids, noDataValue=0, dataType=gdal.GDT_UInt16)
        foo.raster2vector(files.ids, tmpFile, fieldName='ids', dataType=ogr.OFTInteger, fieldIndex=0)
        # # Validates geometries
        # if args.makeValid:
        #     tmpFold, tmpFileValid = makeTempFile(tile, "idsValid", "shp")
        #     foo.buffer(tmpFile, tmpFileValid, 0)
        #     tmpFile = tmpFileValid
        # # Converts polygons to multipolygons
        # command = "ogr2ogr {} {} -dialect sqlite -sql \"SELECT ST_Union(geometry), ids FROM {} GROUP BY ids\""
        # command = command.format(files.idsv, tmpFile, foo.fileName(tmpFile)) # Output, Input, Layer
        # os.system(command)
        # # Computes convex hulls
        # foo.convexHulls(files.idsv, files.hull)
        # foo.resetFolder(tmpFold, remove=True)
"""