#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Predicts buildings from Etat-Major maps (step 1)
@author: Clement Gorin
@contact: gorinclem@gmail.com
@date: August 2021
'''

#%% MODULES AND PATHS

# Modules
import argparse
import concurrent.futures
import cv2
import joblib
import histo1860_functions as foo
import itertools
import numpy as np
import os
import pandas as pd
import shutil
import skimage
import skimage.exposure
import skimage.feature
import sklearn
import sklearn.ensemble
import sklearn.model_selection
import time

from osgeo import gdal
from osgeo import ogr

# Paths and utilities
paths  = foo.makePaths('buildings')
tile   = '0880_6260' # Testing only

#%% FUNCTIONS

# Builds file paths
def makeFiles(tile):
    files = argparse.Namespace(
        scem   = os.path.join(paths.scem, 'scem_{}.tif'),
        mask   = os.path.join(paths.mask, 'mask_{}.tif'),
        X      = os.path.join(paths.X,    'X_{}.tif'),
        X1     = os.path.join(paths.X1,   'X1_{}.npy'),
        Xy1    = os.path.join(paths.Xy1,  'Xy1_{}.tif'),
        Xy1_df = os.path.join(paths.Xy1,  'Xy1_{}.feather'),
        y      = os.path.join(paths.y,    'y_{}.tif'),
        yh1p   = os.path.join(paths.yh1,  'yh1p_{}.tif'),
        yh1    = os.path.join(paths.yh1,  'yh1_{}.tif'),
        yh1fix = os.path.join(paths.tmp,  'yh1fix_{}.tif'),
        tmp    = os.path.join(paths.tmp,  'tmp_{}.tif')
    )
    files = dict((key, value.format(tile)) for key, value in vars(files).items())
    files = argparse.Namespace(**files)
    return(files)

# Computes inputs variables
def computeX1(tile):
    files   = makeFiles(tile)
    image   = foo.raster2array(files.X)
    # Colour
    colour  = skimage.color.rgb2lab(image)
    # Texture
    texture = list()
    gray    = skimage.color.rgb2gray(image)
    for argument in [(8, 1), (16, 2), (24, 3), (32, 4)]:
        texture.append(skimage.feature.local_binary_pattern(gray, *argument, 'ror'))
    texture = np.dstack(texture)
    vars    = np.dstack((colour, texture))
    # Normalisation (not good makes variable tile specific)
    # means = np.mean(vars, axis=(0, 1), keepdims=True)
    # stds  = np.std(vars,  axis=(0, 1), keepdims=True)
    # vars  = (vars - means) / stds
    return(vars)

# Computes borders
def computeBorder(array, class_value=1, kernel_size=3):
    subset = np.isin(array, class_value).astype(int)
    inner  = subset - skimage.morphology.erosion(subset, skimage.morphology.square(kernel_size))
    outer  = skimage.morphology.dilation(subset, skimage.morphology.square(kernel_size)) - subset
    inside = (subset - inner).astype(bool)
    border = (inner + outer).astype(bool)
    return(inside, border, inner, outer)

# Computes training samples
def computeSamples(tile, args):
    print(tile)
    files = makeFiles(tile)
    # Data
    y    = foo.raster2array(files.y)
    mask = foo.raster2array(files.mask)
    mask = mask & np.isin(y, 0, invert = True)
    # Computes borders
    inside1, border1 = computeBorder(y, 1)
    inside2, border2 = computeBorder(y, 2)
    border1[border2 == 1] = 0 # Gives priority to border2
    # Computes candidate pixels
    y1cand = (mask & inside1).astype(bool)
    y2cand = (mask & inside2).astype(bool)
    b1cand = (mask & border1).astype(bool)
    b2cand = (mask & border2).astype(bool)
    y0cand = (mask & np.isin(y, [3, 4, 5, 6, 7, 8]) & np.invert(border1) & np.invert(border2)).astype(bool)
    # Draws samples
    np.random.seed(1)
    index  = np.arange(5000 * 5000).reshape(5000, 5000)
    y1draw = np.random.choice(index[y1cand], int(args.y1share * np.sum(y1cand)), replace=False)
    y2draw = np.random.choice(index[y2cand], int(args.y2share * np.sum(y2cand)), replace=False)
    b1draw = np.random.choice(index[b1cand], int(args.b1share * np.sum(b1cand)), replace=False)
    b2draw = np.random.choice(index[b2cand], int(args.b2share * np.sum(b2cand)), replace=False)
    y0draw = np.random.choice(index[y0cand], int(args.y0share * np.sum(y0cand)), replace=False)
    # Creates sampled raster
    sample = np.zeros((5000 * 5000)).astype(int)
    np.put(sample, y1draw, 1)
    np.put(sample, y2draw, 2)
    np.put(sample, b1draw, 3)
    np.put(sample, b2draw, 4)
    np.put(sample, y0draw, 5)
    sample = sample.reshape(5000, 5000)
    foo.array2raster(sample, files.y, files.Xy1, noDataValue=0, dataType=gdal.GDT_Byte)

# Computes training data
def computeXy1(tile):
    print(tile)
    files  = makeFiles(tile)
    sample = foo.raster2array(files.Xy1)
    y      = foo.raster2array(files.y)
    X      = computeX1(tile)
    index  = sample > 0
    sample = np.column_stack((sample[index], y[index], X[index]))
    sample = pd.DataFrame(sample, columns = ['type', 'y', 'L', 'A', 'B', 'tex1', 'tex2', 'tex3', 'tex4'])
    sample = sample.astype({'type': 'int', 'y': 'int'})
    sample.to_feather(files.Xy1_df)
    return(files.Xy1_df)
    
# Computes estimated response
def computeYh1(tile):
    print(tile)
    files = makeFiles(tile)
    mask  = foo.raster2array(files.mask).flatten().astype(bool)
    if np.count_nonzero(mask) > 0:
        X   = computeX1(tile)
        X   = X.reshape((5000 * 5000, X.shape[2]))
        yhp = np.zeros((len(mask), 2))
        yhp[mask, ...] = fh1.predict_proba(X[mask, ...])
        yhp = yhp[:,1].reshape(5000, 5000)
        foo.array2raster(yhp, files.X, files.yh1p, noDataValue=-1, dataType=gdal.GDT_Float32)

# Filters pixels by size
def filterBySize(array, arange=np.arange(1, 6), connectivity=2, background=0):
    # Label areas
    labels = skimage.morphology.label(array, connectivity=connectivity, background=background)
    # Computes counts
    values = np.arange(np.amin(labels), np.amax(labels) + 1)
    counts = np.bincount(labels.flatten())
    index  = np.isin(values, background, invert=True)
    values = values[index]
    counts = counts[index]
    # Maps values
    output = np.zeros(values.max() + 1, dtype=counts.dtype)
    output[values] = counts
    output = output[labels]
    # Filters values
    output = np.isin(output, arange)
    return(output)

# Computes post-processing
def postProcess(tile, args):
    # args  = argparse.Namespace(threshold=0.5, augment=True, morphology=True, remove_walls=True)
    print(tile)
    files = makeFiles(tile)
    # Data
    yh1p = foo.raster2array(files.yh1p)
    yh1  = yh1p > args.threshold
    # Fixes predictions
    if args.augment == True and (os.path.exists(files.y) or os.path.exists(files.yh1fix)):
        # Aggregate fixes
        fix = np.zeros((5000, 5000), dtype=bool)
        if os.path.exists(files.yh1fix):
            tmp = foo.raster2array(files.yh1fix)
            fix = np.logical_or(fix, tmp) 
        if os.path.exists(files.y):
            tmp = foo.raster2array(files.y)
            tmp = np.logical_and(np.isin(tmp, [1, 2]), yh1p > 0.1) # IGN mistakes
            fix = np.logical_or(fix, tmp)
        del(tmp)
        # Computes categories
        inside, _, inner, _ = computeBorder(fix, True)
        # Fixes inside
        inside = np.where(inside) # Low probabilities are not fixed
        index  = np.arange(len(inside[0]))
        sample = np.random.choice(index, size=int(0.95 * len(index)), replace=False)
        yh1[inside[0][sample],inside[1][sample]] = 1
        # Fixes inner border
        inner  = np.where(inner) # Low probabilities are not fixed
        index  = np.arange(len(inner[0]))
        sample = np.random.choice(index, size=int(0.95 * len(index)), replace=False)
        yh1[inner[0][sample],inner[1][sample]] = 1
    # Fills holes and removes small aggregates
    if args.morphology:
        yh1 = yh1.astype(bool)
        yh1 = skimage.morphology.remove_small_holes(yh1, 25)
        yh1 = skimage.morphology.remove_small_objects(yh1, 5)
    # Morphological opening
    if args.remove_walls:
        labels   = skimage.morphology.label(yh1, connectivity=2, background=0)
        nowalls  = skimage.morphology.opening(yh1, skimage.morphology.square(3)).astype(bool)
        # Adds borders that were smoothed out with the convolution
        isolated = np.where(nowalls, 0, labels) # Labels present in nowalls?
        isolated = filterBySize(isolated, np.arange(1, 12))
        nowalls  = np.where(isolated, 1, nowalls)
        # Removes small aggregates that were left by the convolution (e.g. parts of walls)
        nowalls  = nowalls.astype(bool)
        nowalls  = skimage.morphology.remove_small_objects(nowalls, 5, connectivity=2)
        # Adds aggregates that disappeared during the convolution
        noids    = list(set(np.unique(labels[labels > 0])) - set(np.unique(labels[nowalls])))
        nowalls  = np.where(np.isin(labels, noids), 1, nowalls)
    foo.array2raster(nowalls, files.yh1p, files.yh1, noDataValue=0, dataType=gdal.GDT_Byte)
   
# Training tiles identifiers
def getOHStiles():
    ohsDS  = ogr.GetDriverByName('GPKG').Open(os.path.join(paths.tile, 'ohs_tiles.gpkg'), 0)
    ohsLay = ohsDS.GetLayer()
    tiles  = [feature.GetField('tile') for feature in ohsLay]
    return(tiles)

# Checks results
def checkTile(tile, args):
    print(tile)
    files = argparse.Namespace(
        scem = os.path.join(paths.scem, 'scem_{}.tif'),
        ohs  = os.path.join(paths.ohs,  'ohs_{}.tif'),
        yh1  = os.path.join(paths.yh1,  'yh1_{}.tif'),
        yh1p = os.path.join(paths.yh1,  'yh1p_{}.tif'),
        tmp  = os.path.join(paths.tmp,  'tmp_{}'),
        yh1_style = os.path.join(paths.data,  'styles', 'buildings_yh1.qml'),
    )
    files = argparse.Namespace(**dict((k, v.format(tile)) for k, v in vars(files).items()))
    if args.scem:
        os.system('open {}'.format(files.scem))
    if args.yh1 and os.path.exists(files.yh1):
        outFile = files.tmp + "_yh1_view.gpkg"
        foo.raster2vector(files.yh1, outFile, driver="GPKG", layerName='')
        shutil.copyfile(files.yh1_style, outFile.replace("gpkg", "qml"))
        os.system('open {}'.format(outFile))
    if args.yh1p and os.path.exists(files.yh1p):
        os.system('open {}'.format(files.yh1p))

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

#%% COMPUTES TRAINING SAMPLES
    
# Computes training samples
tiles = foo.getTiles(paths.y)
args  = argparse.Namespace(y1share=0.1, b1share=0.1, y2share=0.2, b2share=0.2, y0share=0.005)
with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    executor.map(computeSamples, tiles, itertools.repeat(args))
del(tiles, args)

# Computes training data
tiles = foo.getTiles(paths.y)
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    samples = list(executor.map(computeXy1, tiles))
del(tiles)

#%% FITS RANDOM FOREST

# Loads training samples
samples = foo.getTiles(paths.Xy1, pattern1='.feather')
samples = [makeFiles(sample).Xy1_df for sample in samples]
samples = pd.concat([pd.read_feather(sample) for sample in samples])

# Diagnostics
samples.shape
samples.y.value_counts()
samples.type.value_counts()

# Subsampling and recoding
samples   = samples.sample(int(10e6), axis=0, replace=False, random_state=1)
samples.y = samples.y.replace(2, 1)
samples.y = samples.y.replace([3, 4, 5, 6, 7, 8], 0)

# Training samples
Xtrain = samples.drop(['type', 'y'], axis=1)
ytrain = samples.y
del(samples)

# Estimation
start = time.time() # 12 minutes on WKSPSE
args  = argparse.Namespace(n_estimators=None, criterion='gini', max_features='sqrt', bootstrap=True, oob_score=True, n_jobs=-1, random_state=1, verbose=1, warm_start=True)
fh1   = sklearn.ensemble.RandomForestClassifier(**vars(args))
oobs  = list()

for n_estimators in np.arange(10, 110, 10, dtype=int):
    fh1.set_params(n_estimators = n_estimators)
    fh1.fit(Xtrain, ytrain)
    print('Trees: {} - OOB: {}'.format(n_estimators, fh1.oob_score_))
    oobs.append(fh1.oob_score_)

joblib.dump(fh1, os.path.join(paths.fh1, 'fh1.sav'))
print('%.2f' % ((time.time() - start) / 60))
del(start, args, Xtrain, ytrain)

# Diagnostics
# fh1 = joblib.load(os.path.join(paths.fh1, 'fh1.sav'))
fh1.oob_score_
importance = pd.Series(fh1.feature_importances_, index=['l', 'a', 'b', 'tex1', 'tex2', 'tex3', 'tex4'])
importance = importance.sort_values(ascending=True)
importance.plot(kind='barh', figsize=(10, 10))
del(importance)

#%% PREDICTS TILES

# Loads model
fh1 = joblib.load(os.path.join(paths.fh1, 'fh1.sav'))
fh1.set_params(n_jobs=2, verbose=0)

# Predicts tiles
# 127 tiles with no buildings
tiles = foo.getTiles(paths.X, paths.yh1, pattern2='yh1p_', operator='difference')
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(computeYh1, tiles)
del(tiles)

#%% POST PROCESSES TILES

# 3 tiles empty after post-processing
tiles  = foo.getTiles(paths.yh1, paths.yh1, pattern1='yh1p_', pattern2='yh1_', operator='difference')
cities = pd.read_csv(os.path.join(paths.data, 'buildings', 'fixes', 'cities.csv'))
tiles  = list(cities.tile)
args   = argparse.Namespace(threshold=0.5, augment=True, morphology=True, remove_walls=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(postProcess, tiles, itertools.repeat(args))
del(tiles, args)

#%% CHECKS RESULTS

args   = argparse.Namespace(scem=True, yh1=True, yh1p=False)
cities = pd.read_csv(os.path.join(paths.data, 'buildings', 'fixes', 'cities.csv'))
tiles  = list(cities.tile)
for tile in tiles:
    checkTile(tile, args)

tile = '0740_6760'
postProcess(tile, argparse.Namespace(threshold=0.5, augment=True, morphology=True, remove_walls=True))
checkTile(tile, argparse.Namespace(scem=True, yh1=True, yh1p=False))