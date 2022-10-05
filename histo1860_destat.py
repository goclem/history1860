#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Descriptive statistics for Arthisto1860
@author:      Clement Gorin
@contact:     gorinclem@gmail.com
@date:        August 2021
'''

#%% MODULES

import argparse
import concurrent.futures
import cv2
import functools
import histo1860_functions as foo
import numpy as np
import importlib
import itertools
import joblib
import os
import osgeo.gdal
import osgeo.ogr
import pandas as pd
import pickle
import scipy
import shutil
import sklearn
import sklearn.ensemble
import skimage.morphology
# foo = importlib.reload(foo)

from osgeo import gdal

# Paths
pathsBuild = foo.makePaths('buildings')
pathsLand  = foo.makePaths('landuse')
pathsLand.clc  = os.path.join(pathsLand.base, 'data_2018')
pathsLand.alti = os.path.join(pathsLand.base, 'data_project', 'elevation')
tile   = '0880_6260'
# paths.shared = os.path.join(paths.base, 'shared_data', '1860', 'qgis_destat1860', 'layers')

#%% Functions

# Creates file paths
def makeFiles(tile):
    files = argparse.Namespace(
        scem = os.path.join(pathsLand.scem, 'scem_{}.tif'),
        mask = os.path.join(pathsLand.mask, 'mask_{}.tif'),
        y    = os.path.join(pathsLand.y,    'y_{}.tif'),
        y2b  = os.path.join(pathsBuild.y2,  'y2_{}.npy'),
        ylu  = os.path.join(pathsLand.ydat, 'y_{}.pkl'),
        Xlu  = os.path.join(pathsLand.Xdat, 'X_{}.pkl'),
        yh   = os.path.join(pathsLand.yh,   'yh_{}.tif'),
        yh1  = os.path.join(pathsBuild.yh1, 'yh1_{}.tif'),
        yh2  = os.path.join(pathsBuild.yh2, 'yh2_{}.tif'),
        X2   = os.path.join(pathsBuild.X2,  'X2_{}.npy'),
        clc  = os.path.join(pathsLand.clc,  'clc_{}.tif'),
        alti = os.path.join(pathsLand.alti, 'elevation_{}.tif')
    )
    files = dict((k, v.format(tile)) for k, v in vars(files).items())
    files = argparse.Namespace(**files)
    return(files)

# Computes borders
def computeBorder(array, class_value=1, kernel_size=3):
    subset = np.isin(array, class_value).astype(int)
    inner  = subset - skimage.morphology.erosion(subset, skimage.morphology.square(kernel_size))
    outer  = skimage.morphology.dilation(subset, skimage.morphology.square(kernel_size)) - subset
    inside = (subset - inner).astype(bool)
    border = (inner + outer).astype(bool)
    return(inside, border, inner, outer)

# Computes multi-class confusion matrix
def confusionMatrix(tile:str, args:dict):
    """
    args = argparse.Namespace(
        mapValues    ={0:0, 1:1, 2:1, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0},
        labels       ={0:'Undefined', 1:'Buildings', 3:'Crops', 4:'Meadows', 5:'Pastures', 6:'Specialised', 7:'Forests', 8:'Water'}, 
        dropValues   =[0],
        removeBorders=None)
    """
    print(tile)
    files = makeFiles(tile)
    # Data
    y    = foo.raster2array(files.y)
    yh   = foo.raster2array(files.yh)
    mask = foo.raster2array(files.mask).astype(bool)
    # Maps values
    for key, value in args.mapValues.items():
        if key != value:
            y  = np.where(y  == key, value, y)
            yh = np.where(yh == key, value, yh)
    # Removes y borders
    if args.removeBorders in ['y', 'both']:
        inside = np.zeros(mask.shape, dtype=bool)
        for key, value in args.mapValues.items():
            temp, _, _, _ = computeBorder(y, value)
            inside = np.logical_or(inside, temp)
        mask = np.logical_and(mask, inside)
        del(inside, temp, _)
    # Removes yh borders
    if args.removeBorders in ['yh', 'both']:
        inside = np.zeros(mask.shape, dtype=bool)
        for key, value in args.mapValues.items():
            temp, _, _, _ = computeBorder(yh, value)
            inside = np.logical_or(inside, temp)
        mask = np.logical_and(mask, inside)
        del(inside, temp, _)
    # Confusion matrix
    confmat = pd.DataFrame({'y' : y[mask], 'yh' : yh[mask]})
    confmat = pd.crosstab(confmat.y, confmat.yh, margins=False)
    confmat = confmat.reindex(index=args.labels, columns=args.labels, fill_value=0)
    # Drop values
    if args.dropValues is not None:
        confmat = confmat.drop(args.dropValues, axis=0)
        confmat = confmat.drop(args.dropValues, axis=1)
    # Formats axes
    confmat = confmat.rename_axis('True', axis=0)
    confmat = confmat.rename_axis('Predicted', axis=1)
    confmat = confmat.rename(args.labels, axis=0)
    confmat = confmat.rename(args.labels, axis=1)
    return(confmat)

def rasterStats(tile:str, 
                mapValues:dict={0:0, 1:1, 2:1, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}, 
                stats:list=['tp', 'tn', 'fp', 'fn', 'mask'],
                removeBorders:str=None):
    """Displays binary predictions statistics

    Args:
        tile (str): Tile identifier
        mapValues (dict, optional): Mapping to binary raster
    """
    print(tile)
    files = makeFiles(tile)
    # Data
    y    = foo.raster2array(files.y)
    yh   = foo.raster2array(files.yh)
    mask = np.where(y == 0, 0, 1)
    # Maps values
    for key, value in mapValues.items():
        if key != value:
            y  = np.where(y  == key, value, y)
            yh = np.where(yh == key, value, yh)
    # Removes y borders
    if removeBorders in ['y', 'both']:
        _, border, _, _ = computeBorder(y, 1)
        mask = np.logical_and(mask, np.invert(border))
    # Removes yh borders
    if removeBorders in ['yh', 'both']:
        _, border, _, _ = computeBorder(yh, 1)
        mask = np.logical_and(mask, np.invert(border))
    # Statistics
    tp = np.logical_and(np.logical_and(y == 1, yh == 1), mask)
    tn = np.logical_and(np.logical_and(y == 0, yh == 0), mask)
    fp = np.logical_and(np.logical_and(y == 0, yh == 1), mask)
    fn = np.logical_and(np.logical_and(y == 1, yh == 0), mask)
    destats = {'tp': np.sum(tp), 'tn': np.sum(tn), 'fp': np.sum(fp), 'fn': np.sum(fn)}
    # Save files
    os.system('open {}'.format(files.scem))
    for stat in stats:
        outFile = os.path.join(pathsLand.tmp, '{}_{}.tif'.format(stat, tile))
        foo.array2raster(locals()[stat], files.scem, outFile)
        shutil.copy(os.path.join(pathsLand.styles, 'destat.qml'), outFile.replace('tif', 'qml'))
        os.system('open {}'.format(outFile))
    return(destats)

# Computes binary confusion matrix
def binaryConfusionMatrix(tile, args):
    print(tile)
    files = makeFiles(tile)
    # Data
    y    = foo.raster2array(files.y)
    yh   = foo.raster2array(files.yh)
    mask = foo.raster2array(files.mask).astype(bool)
    # Maps values
    for key, value in args.mapValues.items():
        if key != value:
            y  = np.where(y  == key, value, y)
            yh = np.where(yh == key, value, yh)
    # Corrects borders
    if args.removeBorders in ['y', 'both']:
        _, border, _, _ = computeBorder(y, 1)
        mask = np.logical_and(mask, np.invert(border))
        del(_, border)
    if args.removeBorders in ['yh', 'both']:
        _, border, _, _ = computeBorder(yh, 1)
        mask = np.logical_and(mask, np.invert(border))
        del(_, border)
    # Rasters
    tp = np.logical_and(np.logical_and(y == 1, yh == 1), mask)
    tn = np.logical_and(np.logical_and(y == 0, yh == 0), mask)
    fp = np.logical_and(np.logical_and(y == 0, yh == 1), mask)
    fn = np.logical_and(np.logical_and(y == 1, yh == 0), mask)
    stats = {'tp': np.sum(tp), 'tn': np.sum(tn), 'fp': np.sum(fp), 'fn': np.sum(fn)}
    # Rasters
    if args.rasters:
        # Raster tile
        outFile = os.path.join(paths.destat, os.path.basename(files.scem))
        shutil.copy(files.scem, outFile)
        os.system('open {}'.format(outFile))
        # Predictions
        outFile = os.path.join(paths.destat, 'yh2_{}.shp'.format(tile))
        foo.raster2vector(files.yh2, outFile)
        shutil.copy(os.path.join(paths.styles, 'buildings_yh2.qml'), outFile.replace('shp', 'qml'))
        os.system('open {}'.format(outFile))
        # Statistics
        for stat in ['tp', 'tn', 'fp', 'fn', 'mask']:
            outFile = os.path.join(paths.destat, '{}_{}_correct_{}.tif'.format(stat, tile, args.removeBorders))
            foo.array2raster(locals()[stat], files.y, outFile, noDataValue=0, dataType=gdal.GDT_Byte)
            shutil.copy(os.path.join(paths.styles, 'destat.qml'), outFile.replace('tif', 'qml'))
            os.system('open {}'.format(outFile))
    return(stats)

# Computes statistics multiclass binary confusion matrix
def percentages(confmat, axis, total=True):
    if axis not in ['True', 'Predicted']:
        raise ValueError('Axis must be True or Predicted')
    if axis=='True':
        print('Read as rows:\nAmong the pixels of [true], [value]% are predicted as [predicted]')
        confmat = confmat.divide(confmat.sum(axis=1), axis=0)
    if axis=='Predicted':
        print('Read as columns:\nAmong the pixels predicted as [predicted], [value]% are in fact [true]')
        confmat = confmat.divide(confmat.sum(axis=0), axis=1)
    # Adding totals
    if total:
        confmat.loc['Total', :] = confmat.sum(axis=0)
        confmat.loc[:, 'Total'] = confmat.sum(axis=1)
    confmat = confmat.applymap(lambda value: '{0:.2f} %'.format(value * 100))
    return(confmat)

def makeBinary(confmat, target='Buildings'):
    tp = confmat.loc[target, target]
    tn = confmat.drop(index=target, columns=target).values.sum()
    fp = confmat.loc[:, target].drop(target).values.sum()
    fn = confmat.loc[target, :].drop(target).values.sum()
    stats = {'tp': tp, 'tn': tn, 'fp':fp, 'fn':fn}
    return(stats)

# Computes statistics from binary confusion matrix
def computeStatistics(binaryConfmat):
    tp, tn, fp, fn = binaryConfmat
    nobs       = fp + fn + tp
    recall     = tp / (tp + fn) # Number of correctly predicted positive observations / total of correctly predicted observations
    precision  = tp / (tp + fp) # Number of correctly predicted positive observations / total of positive observations
    statistics = pd.DataFrame({'Observations': [nobs], 'Recall': [recall], 'Precision': [precision]})
    statistics.Observations = statistics.Observations.apply(lambda value : '{:,}'.format(value))
    statistics.Recall       = statistics.Recall.apply(lambda value: '{0:.1f} %'.format(value * 100))
    statistics.Precision    = statistics.Precision.apply(lambda value: '{0:.1f} %'.format(value * 100))
    return(statistics)

def tabulate(tile):
    files = makeFiles(tile)
    yh  = foo.raster2array(files.yh)
    tab = pd.value_counts(yh[yh > 0])
    tab = tab.reindex(index = [1, 3, 4, 5, 6, 7, 8], fill_value=0)
    tab = tab.rename({1:'Buildings', 3:'Crops', 4:'Meadows', 5:'Pastures', 6:'Specialised', 7:'Forests', 8:'Water'})
    return(tab)

# Training tiles identifiers
def getOHStiles(file='ohs_tiles.gpkg'):
    ohsDS  = osgeo.ogr.GetDriverByName('GPKG').Open(os.path.join(paths.tile, file), 0)
    ohsLay = ohsDS.GetLayer()
    tiles  = [feature.GetField('tile') for feature in ohsLay]
    return(tiles)
#%% MULTI-CLASS CONFUSION MATRICES

# ALL CLASSES WITHOUT BORDER CORRECTION

# Computations
args = argparse.Namespace(
    mapValues    ={0:0, 1:1, 2:1, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8},
    labels       ={0:'Undefined', 1:'Buildings', 3:'Crops', 4:'Meadows', 5:'Pastures', 6:'Specialised', 7:'Forests', 8:'Water'}, 
    dropValues   =[0],
    removeBorders=None)

tiles = foo.getTiles(pathsLand.y, pathsLand.yh, operator='intersection')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    yhcms = list(executor.map(confusionMatrix, tiles, itertools.repeat(args)))

# Serialise
yhcms = dict(zip(tiles, yhcms))
with open(os.path.join(pathsLand.destat, 'yhcms.pkl'), 'wb') as file:
    pickle.dump(yhcms, file)

# Aggregation
with open(os.path.join(pathsLand.destat, 'yhcms.pkl'), 'rb') as file:
    yhcms = pickle.load(file)
yhcm = functools.reduce(lambda cm1, cm2: cm1.add(cm2, fill_value=0), yhcms.values())
yhcm.to_pickle(os.path.join(pathsLand.destat, 'yhcm.pkl'))

# Descriptive statistics
yhcm = pd.read_pickle(os.path.join(pathsLand.destat, 'yhcm.pkl'))
yhcm.values.sum()
percentages(yhcm, axis='True')
percentages(yhcm, axis='Predicted')

print(percentages(yhcm, axis='True').to_latex(float_format='%.2f'))
print(percentages(yhcm, axis='Predicted').to_latex(float_format='%.2f'))

# ALL CLASSES WITH BORDER CORRECTION

# Computations
args = argparse.Namespace(
    mapValues    ={0:0, 1:1, 2:1, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8},
    labels       ={0:'Undefined', 1:'Buildings', 3:'Crops', 4:'Meadows', 5:'Pastures', 6:'Specialised', 7:'Forests', 8:'Water'}, 
    dropValues   =[0],
    removeBorders='both')

tiles = foo.getTiles(pathsLand.y, pathsLand.yh, operator='intersection')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    yhcmnbs = list(executor.map(confusionMatrix, tiles, itertools.repeat(args)))

# Serialise
yhcmnbs = dict(zip(tiles, yhcmnbs))
with open(os.path.join(pathsLand.destat, 'yhcmnbs.pkl'), 'wb') as file:
    pickle.dump(yhcmnbs, file)

# Aggregation
with open(os.path.join(pathsLand.destat, 'yhcmnbs.pkl'), 'rb') as file:
    yhcmnbs = pickle.load(file)
yhcmnb = functools.reduce(lambda cm1, cm2: cm1.add(cm2, fill_value=0), yhcmnbs.values())
yhcmnb.to_pickle(os.path.join(pathsLand.destat, 'yhcmnb.pkl'))

# Descriptive statistics
yhcmnb = pd.read_pickle(os.path.join(pathsLand.destat, 'yhcmnb.pkl'))
yhcmnb.values.sum() # Removes 13%
percentages(yhcmnb, axis='True')
percentages(yhcmnb, axis='Predicted')

print(percentages(yhcmnb, axis='True').to_latex(float_format='%.2f'))
print(percentages(yhcmnb, axis='Predicted').to_latex(float_format='%.2f'))

# SUMMARY TABLE (predall.tex)

summary = pd.concat([
    pd.Series(np.diag(percentages(yhcm,   axis='True',      total=False))),
    pd.Series(np.diag(percentages(yhcm,   axis='Predicted', total=False))),
    pd.Series(np.diag(percentages(yhcmnb, axis='True',      total=False))),
    pd.Series(np.diag(percentages(yhcmnb, axis='Predicted', total=False)))
], axis = 1)
summary = summary.transpose()
summary.columns =['Buildings', 'Crops', 'Meadows', 'Pastures', 'Specialised', 'Forests', 'Water']
summary.index   =['Recall', 'Precision', 'Recall', 'Precision']

print(summary.to_latex())

# Other statistics
np.diag(yhcm).sum() / (yhcm.values.sum()) * 100
np.diag(yhcmnb).sum() / (yhcmnb.values.sum()) * 100

#%% INVESTIGATING ERRORS

yhcmnybs = pd.read_pickle(os.path.join(pathsLand.destat, 'yhcmnybs.pkl'))
prestats = [makeBinary(cm) for cm in yhcmnybs.values()]
prestats = pd.DataFrame(prestats, index=yhcmnybs.keys())
prestats.sort_values('fn', ascending=False)

# False positive
rasterStats('0700_7060', stats=['fp'])
rasterStats('0720_7040', stats=['fp'])
rasterStats('0700_6540', stats=['fp'])

# False negative
rasterStats('0700_7060', stats=['fn'])
rasterStats('0720_7040', stats=['fn'])
rasterStats('0880_6260', stats=['fn'])



#%% CONFUSION MATRIX BUILDINGS

yhbcm   = makeBinary(yhcm, 'Buildings')
yhbcmnb = makeBinary(yhcmnb, 'Buildings')
computeStatistics(yhbcm)
computeStatistics(yhbcmnb)

#%% TABULATE ALL CLASSES

tiles = foo.getTiles(pathsLand.yh, pattern1='tif$')

tabs = list()
for tile in tiles:
    print(tile)
    tabs.append(tabulate(tile))
    
# Serialise
tabs = dict(zip(tiles, tabs))
with open(os.path.join(pathsLand.destat, 'tabs.pkl'), 'wb') as file:
    pickle.dump(tabs, file)
    
# Aggregation
with open(os.path.join(pathsLand.destat, 'tabs.pkl'), 'rb') as file:
    tabs = pickle.load(file)

tab = functools.reduce(lambda x, y: x.add(y, fill_value=0), list(tabs.values()))
obs = tab.values.sum()
tab = tab.apply(lambda value: '{0:.2f} %'.format(value / obs * 100))
print(tab.to_latex())

#%% TABULATES ALL CLASSES 2018

def tabulateLanduse(tile, altitude_threshold=0):
    files   = makeFiles(tile)
    # Data
    land2018 = foo.raster2array(files.clc)
    land1860 = foo.raster2array(files.yh)
    # Selects observations
    keep = np.logical_and(land1860 > 0, land2018 > 0) # Intersection
    if altitude_threshold > 0:
        altitude = foo.raster2array(files.alti)
        altitude = cv2.resize(altitude, keep.shape)
        keep     = np.logical_and(keep, altitude <= altitude_threshold)
    # Utilities
    index = [1, 3, 4, 5, 6, 7, 8, 10]
    names = {1:'Buildings', 3:'Crops', 4:'Meadows', 5:'Pastures', 6:'Specialised', 7:'Forests', 8:'Water', 10:'Urban'}
    # 1860 data
    tab1860 = pd.value_counts(land1860[keep])
    tab1860 = tab1860.reindex(index=index, fill_value=0)
    tab1860 = tab1860.rename(names)
    # 2018 data
    tab2018 = pd.value_counts(land2018[keep])
    tab2018 = tab2018.reindex(index=index, fill_value=0)
    tab2018 = tab2018.rename(names)
    return(tab1860, tab2018)

tiles = foo.getTiles(pathsLand.alti)
for tile in tiles:
    files = makeFiles(tile)
    alti  = foo.raster2array(files.alti)
    np.sum(alti <= 1500)

# x tiles are not defined in 2018
tiles    = foo.getTiles(pathsLand.yh, pathsLand.clc, operator='intersection')
tabs1860 = list()
tabs2018 = list()

for tile in tiles:
    print(tile)
    tab1860, tab2018 = tabulateLanduse(tile, altitude_threshold=1500)
    tabs1860.append(tab1860)
    tabs2018.append(tab2018)
    
# Serialise
tabs1860 = dict(zip(tiles, tabs1860))
tabs2018 = dict(zip(tiles, tabs2018))

with open(os.path.join(pathsLand.destat, 'tabs1860alt1500.pkl'), 'wb') as file:
    pickle.dump(tabs1860, file)
with open(os.path.join(pathsLand.destat, 'tabs2018alt1500.pkl'), 'wb') as file:
    pickle.dump(tabs2018, file)
    
# Aggregation
files    = ['tabs1860.pkl', 'tabs1860alt1500.pkl', 'tabs2018.pkl', 'tabs2018alt1500.pkl']
tabsComp = list()

for file in files:
    with open(os.path.join(pathsLand.destat, file), 'rb') as file:
        tabs = pickle.load(file)
    tabs = functools.reduce(lambda x, y: x.add(y, fill_value=0), list(tabs.values()))
    tabsComp.append(tabs)

tabsComp = pd.concat(tabsComp, axis=1)
tabsComp.columns = ['landuse1860', 'landuse1860_below1500', 'landuse2018', 'landuse2018_below1500']
tabsComp['landuse1860_above1500'] = tabsComp.landuse1860 - tabsComp.landuse1860_below1500
tabsComp['landuse2018_above1500'] = tabsComp.landuse2018 - tabsComp.landuse2018_below1500
tabsComp = tabsComp.div(tabsComp.sum(axis=0), axis=1) * 100
tabsComp = tabsComp.round(2)
tabsComp = tabsComp[sorted(tabsComp.columns)]
tabsComp = tabsComp.applymap(lambda value: '{0:.2f}%'.format(value))
print(tabsComp.to_latex())

#%% MULTI-CLASS STATISTICS
    
# Aggregate statistics
cm  = functools.reduce(lambda x, y: x.add(y, fill_value=0), tabs)
idx = ('Buildings', 'Land', 'Pastures', 'Crops', 'Forests', 'Water')
cm  = cm.reindex(index=idx, columns=idx, fill_value=0).astype(int)

# Pixels
cm_npx = cm.copy()
cm_npx['Total'] = cm_npx.sum(axis=1)
cm_npx = cm_npx.append(cm_npx.sum(axis=0).rename('Total'))
print(cm_npx.to_latex())

# Col percentages
cm_colshr = (cm.apply(lambda c: c/c.sum(), axis=0) * 100)
cm_colshr['Total'] = cm_colshr.sum(axis=1)
cm_colshr = cm_colshr.append(cm_colshr.sum(axis=0).rename('Total'))
print(cm_colshr.to_latex(float_format='%.2f'))

# Row percentages
cm_rowshr = (cm.apply(lambda r: r/r.sum(), axis=1) * 100)
cm_rowshr['Total'] = cm_rowshr.sum(axis=1)
cm_rowshr = cm_rowshr.append(cm_rowshr.sum(axis=0).rename('Total'))
print(cm_rowshr.to_latex(float_format='%.2f'))

#%% COUNT TILES AND PIXELS

# All tiles
# Dropped a IGN tile that was completely empty
tiles   = foo.getTiles(pathsLand.mask)
npixels = list()
for tile in tiles:
    files = makeFiles(tile)
    mask  = foo.raster2array(files.mask)
    npixels.append(np.sum(mask))

# Statistics
len(npixels)                  # Total number of tiles
len(npixels) * 5000 * 5000    # Total of pixels (billion)
np.sum(np.array(npixels) > 0) # Number of predicted tiles
np.sum(npixels) / 1e9         # Number of predicted pixels (billion) 33863287961

# Training tiles
tiles   = foo.getTiles(pathsLand.y)
npixels = list()
for tile in tiles:
    files = makeFiles(tile)
    y     = foo.raster2array(files.y)
    npixels.append(np.sum(np.isin(y, np.arange(1, 9))))

# Statistics
len(tiles)            # Number of available training tiles
np.sum(npixels) / 1e9 # Number of available training pixels (billion) 6161964810

#%% RANDOM FOREST STATISTICS

fh1 = joblib.load(os.path.join(paths.fh1, 'fh1.sav'))
fh1.oob_score_

fh2 = joblib.load(os.path.join(paths.fh2, 'fh2.sav'))
fh2.oob_score_

#%% COUNTS OBSERVATIONS

# First building prediction
def countObservations(tile):
    files = makeFiles(tile)
    mask  = foo.raster2array(files.mask)
    nobs  = np.sum(mask)
    return(nobs)

tiles = foo.getTiles(pathsBuild.X)
nobs  = list()
for tile in tiles: nobs.append(countObservations(tile))
np.sum(nobs)

# Second building prediction
def countObservationsX2(tile):
    files = makeFiles(tile)
    nobs  = np.load(files.X2).shape[0]
    return(nobs)

def countObservationsY2b(tile, undefined=-1, type='all'):
    files = makeFiles(tile)
    nobs  = np.load(files.y2b)
    if type == 'all':   # Number of superpixels
        nobs  = len(nobs)
    if type == 'train': # Number of superpixels for training
        nobs = np.sum(nobs != undefined)
    return(nobs)

tiles  = foo.getTiles(pathsBuild.X2, pattern1='npy')
nobs   = list()
for tile in tiles: nobs.append(countObservationsX2(tile))
np.sum(nobs)

tiles = foo.getTiles(pathsBuild.y2, pattern1='npy')
nobs  = list()
for tile in tiles: nobs.append(countObservationsY2b(tile))
np.sum(nobs)

# Landuse prediction
def countObservationsXlu(tile):
    files = makeFiles(tile)
    nobs  = pd.read_pickle(files.Xlu)
    nobs  = nobs.shape[0]
    return(nobs)

def countObservationsYlu(tile, undefined=1, type='all'):
    files = makeFiles(tile)
    nobs  = pd.read_pickle(files.ylu).y.values
    if type == 'all':   # Number of superpixels
        nobs = len(nobs)
    if type == 'train': # Number of superpixels for training
        nobs = np.sum(nobs != undefined)
    return(nobs)

tiles = foo.getTiles(pathsLand.Xdat, pattern1='pkl')
nobs  = list()
for tile in tiles: nobs.append(countObservationsXlu(tile))

tiles = foo.getTiles(pathsLand.ydat, pattern1='pkl')
nobs  = list()
for tile in tiles: nobs.append(countObservationsYlu(tile, type='train'))
np.sum(nobs)

#%% TABULATES RESPONSE

# Tabulates training classes
def tableY(tile):
    print(tile)
    files = makeFiles(tile)
    y     = foo.raster2array(files.y)
    mask  = foo.raster2array(files.mask).astype(bool)
    table = pd.value_counts(y[mask])
    table = table.reindex(index=[0, 1, 2, 3, 4, 5, 6, 7, 8], fill_value=0)
    return(table)

# Computations
tiles = foo.getTiles(paths.y)
table = pd.Series(0, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
for tile in tiles:
    table = table.add(tableY(tile))

table.pop(0)
share = table / table.sum() * 100
del(tiles)

#%% CONFUSION MATRICES FIRST PREDICTION

# Old version for Gilles presentation
# paths.yh1 = paths.yh1 + '_old'
tiles = getOHStiles(file='ohs_tiles_old.gpkg')
# Remove tiles whose files are unavailable
filt  = foo.getTiles(paths.y, paths.yh1, operator='intersection')
tiles = list(set(tiles) & set(filt))
# Remove tiles that were not in previous table
filt  = pd.read_csv(os.path.join(paths.destat, '{}.csv'.format('confmats_yh3')), sep=';', index_col=0).index
tiles = list(set(tiles) & set(filt))
len(tiles)

# Confusion matrices for yh1
args = argparse.Namespace(target='yh1', removeBorders=None, rasters=False, yValues=[1, 2], yhValues=[1])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    confmats_yh1 = list(executor.map(binaryConfusionMatrix, tiles, itertools.repeat(args)))  
confmats_yh1 = pd.DataFrame(confmats_yh1, index=tiles)
confmats_yh1.sort_index()
confmats_yh1.to_csv(os.path.join(paths.destat, 'confmats_yh1.csv'), sep=';')

# Confusion matrices for yh1 without y and yh borders
args = argparse.Namespace(target='yh1', removeBorders='both', rasters=False, yValues=[1, 2], yhValues=[1])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    confmats_yh1_noborder = list(executor.map(binaryConfusionMatrix, tiles, itertools.repeat(args)))
confmats_yh1_noborder = pd.DataFrame(confmats_yh1_noborder, index=tiles)
confmats_yh1_noborder.sort_index()
confmats_yh1_noborder.to_csv(os.path.join(paths.destat, 'confmats_yh1_noborder.csv'), sep=';')

# Confusion matrices for yh2
args = argparse.Namespace(target='yh2', removeBorders=None, rasters=False, yValues=[1, 2], yhValues=[1])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    confmats_yh2 = list(executor.map(binaryConfusionMatrix, tiles, itertools.repeat(args)))  
confmats_yh2 = pd.DataFrame(confmats_yh2, index=tiles)
confmats_yh2.sort_index()
confmats_yh2.to_csv(os.path.join(paths.destat, 'confmats_yh2.csv'), sep=';')

# Confusion matrices for yh2 without y and yh borders
args = argparse.Namespace(target='yh2', removeBorders='both', rasters=False, yValues=[1, 2], yhValues=[1])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    confmats_yh2_noborder = list(executor.map(binaryConfusionMatrix, tiles, itertools.repeat(args)))
confmats_yh2_noborder = pd.DataFrame(confmats_yh2_noborder, index=tiles)
confmats_yh2_noborder.sort_index()
confmats_yh2_noborder.to_csv(os.path.join(paths.destat, 'confmats_yh2_noborder.csv'), sep=';')

# Aggregate matrix for yh1
confmat_yh1 = pd.read_csv(os.path.join(paths.destat, '{}.csv'.format('confmats_yh1')), sep=';', index_col=0)
confmat_yh1 = confmat_yh1.loc[set(confmat_yh1.index) & set(tiles)]
confmat_yh1 = confmat_yh1.sum(axis=0)
statistics  = computeStatistics(confmat_yh1)
print(statistics.iloc[:3,:2].to_latex(index=False, float_format='%.4f'))

# Aggregate matrix for yh1 without y and yh borders
confmat_yh1_noborder = pd.read_csv(os.path.join(paths.destat, '{}.csv'.format('confmats_yh1_noborder')), sep=';', index_col=0)
confmat_yh1_noborder = confmat_yh1_noborder.loc[set(confmat_yh1_noborder.index) & set(tiles)]
confmat_yh1_noborder = confmat_yh1_noborder.sum(axis=0)
statistics = computeStatistics(confmat_yh1_noborder)
print(statistics.iloc[:3,:2].to_latex(index=False, float_format='%.4f'))

# Aggregate matrix for yh2
confmat_yh2 = pd.read_csv(os.path.join(paths.destat, '{}.csv'.format('confmats_yh2')), sep=';', index_col=0)
confmat_yh2 = confmat_yh2.loc[set(confmat_yh2.index) & set(tiles)]
confmat_yh2 = confmat_yh2.sum(axis=0)
statistics  = computeStatistics(confmat_yh2)
print(statistics.iloc[:3,:2].to_latex(index=False, float_format='%.4f'))

# Aggregate matrix for yh2 without y and yh borders
confmat_yh2_noborder = pd.read_csv(os.path.join(paths.destat, '{}.csv'.format('confmats_yh2_noborder')), sep=';', index_col=0)
confmat_yh2_noborder = confmat_yh2_noborder.loc[set(confmat_yh2_noborder.index) & set(tiles)]
confmat_yh2_noborder = confmat_yh2_noborder.sum(axis=0)
statistics  = computeStatistics(confmat_yh2_noborder)
print(statistics.iloc[:3,:2].to_latex(index=False, float_format='%.4f'))

# Aggregate matrix for yh3
confmat_yh3 = pd.read_csv(os.path.join(paths.destat, '{}.csv'.format('confmats_yh3')), sep=';', index_col=0)
confmat_yh3 = confmat_yh3.loc[set(confmat_yh3.index) & set(tiles)]
confmat_yh3 = confmat_yh3.sum(axis=0)
statistics  = computeStatistics(confmat_yh3)
print(statistics.iloc[:3,:2].to_latex(index=False, float_format='%.4f'))

# Aggregate matrix for yh3 without y and yh borders
confmat_yh3_noborder = pd.read_csv(os.path.join(paths.destat, '{}.csv'.format('confmats_yh3_noborder')), sep=';', index_col=0)
confmat_yh3_noborder = confmat_yh3_noborder.loc[set(confmat_yh3_noborder.index) & set(tiles)]
confmat_yh3_noborder = confmat_yh3_noborder.sum(axis=0)
statistics  = computeStatistics(confmat_yh3_noborder)
print(statistics.iloc[:3,:2].to_latex(index=False, float_format='%.4f'))

#%% CONFUSION MATRICES BUILDINGS

# General options
args = argparse.Namespace(
    target       =None,
    mapValues    ={1:1, 2:1, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0},
    removeBorders=None,
    rasters      =False)

tiles = foo.getTiles(paths.y, os.path.join(paths.data, 'buildings', 'yh1'), operator='intersection')

# Statistics yh1 without correction (after wall post-processing)
args.target = "yh1"
args.removeBorders = None

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    yh1cms = list(executor.map(binaryConfusionMatrix, tiles, itertools.repeat(args)))

yh1cm   = pd.DataFrame(yh1cms, index=tiles)
yh1cm.to_csv(os.path.join(paths.destat, 'yh1cm.csv'), sep=';')
yh1cm   = yh1cm.sum()
yh1stat = computeStatistics(yh1cm)

# Statistics yh2 without correction
args.target = "yh2"
args.removeBorders = None

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    yh2cms = list(executor.map(binaryConfusionMatrix, tiles, itertools.repeat(args)))

yh2cm   = pd.DataFrame(yh2cms, index=tiles)
yh2cm.to_csv(os.path.join(paths.destat, 'yh2cm.csv'), sep=';')
yh2cm   = yh2cm.sum()
yh2stat = computeStatistics(yh2cm)

# Statistics yh2 with border correction
args.target = 'yh2'
args.removeBorders = 'both'

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    yh2cmnbs = list(executor.map(binaryConfusionMatrix, tiles, itertools.repeat(args)))

yh2cmnb   = pd.DataFrame(yh2cmnbs, index=tiles)
yh2cmnb.to_csv(os.path.join(paths.destat, 'yh2cmnb.csv'), sep=';')
yh2cmnb   = yh2cmnb.sum()
yh2statnb = computeStatistics(yh2cmnb)