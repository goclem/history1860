#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Extract streets from Etat-Major maps
@author: Clement Gorin
@contact: gorinclem@gmail.com
@version: 2022.01.31
'''

#%% MODULES

import argparse
import concurrent.futures
import itertools
import numpy as np
import os
import scipy.ndimage
import shutil
import skimage
import skimage.feature
import skimage.filters
import socket

from osgeo import gdal

import sys
sys.path.append('/Users/clementgorin/Dropbox/research/arthisto/arthisto1860')
import histo1860_functions as foo

#%% FUNCTIONS

# Sets paths
def makePaths():
    if socket.gethostname() == 'PSE-CALC-MAPHIS':
        os.environ['USERPROFILE'] = 'F:\cgorin'
    source = os.path.join(os.path.expanduser('~'), 'Dropbox', 'data')
    base   = os.path.join(os.path.expanduser('~'), 'Dropbox', 'research', 'arthisto')
    paths  = argparse.Namespace(
        source      = source,
        base        = base,
        street      = os.path.join(base,   'shared_data', 'landuse18602020', 'qgis_water18602020', 'layers'),
        landuse1860 = os.path.join('G:\\', 'final', 'landuse1860'),
        landuse2018 = os.path.join('G:\\', 'final', 'landuse2018'),
        street1860  = os.path.join('G:\\', 'final', 'streets1860'),
        street2018  = os.path.join('G:\\', 'final', 'streets2018'),
        tmp         = os.path.join(os.path.expanduser('~'), 'Temporary'),
        desktop     = os.path.join(os.path.expanduser('~'), 'Desktop'),
        styles      = os.path.join(base, 'data_1860', 'styles')
    )
    return(paths)

# Builds file paths
def makeFiles(tile:str, outlabel:str='', tmplabel:str='') -> argparse.Namespace:
    paths = makePaths()
    files = argparse.Namespace(
        scem          = os.path.join(paths.source,      'ign_scem', 'scem_{}.tif'),
        landuse1860   = os.path.join(paths.landuse1860, 'yh_{}.tif'),
        landuse2018   = os.path.join(paths.landuse2018, 'clc_{}.tif'),
        street1860    = os.path.join(paths.street1860,  'street1860_{}' + outlabel + '.tif'),
        street2018    = os.path.join(paths.street2018,  'street2018_{}' + outlabel + '.tif'),
        scemCopy      = os.path.join(paths.street,      'scem_{}.tif'),
        build1860vec  = os.path.join(paths.street,      'build1860_{}.gpkg'),
        build2018vec  = os.path.join(paths.street,      'build2018_{}.gpkg'),
        street1860vec = os.path.join(paths.street,      'street1860_{}' + outlabel + '.gpkg'),
        street2018vec = os.path.join(paths.street,      'street2018_{}' + outlabel + '.gpkg'),
        urban2018vec  = os.path.join(paths.street,      'urban2018_{}.gpkg'),
        tmp           = os.path.join(paths.tmp, tmplabel + '_{}.tif')
    )
    files = dict((k, v.format(tile)) for k, v in vars(files).items())
    files = argparse.Namespace(**files)
    return(files)

def computeStreets(tile:str, args:argparse.Namespace) -> np.array:
    files = makeFiles(tile, outlabel=args.outlabel)
    # Loads data
    if args.year == '1860':
        builds = foo.raster2array(files.landuse1860)
    if args.year == '2018':
        builds = foo.raster2array(files.landuse2018)
    builds = np.isin(builds, 1)
    # Labels buildings with their size (larger buildings are given more importance)
    if np.sum(builds > 0):
        labels = skimage.measure.label(builds)
        values = np.arange(1, np.amax(labels) + 1)
        counts = np.bincount(labels.flatten())[1:]
        labels = foo.mapValues(labels, values, counts)
        # Performs convolution
        kernel  = skimage.morphology.disk(args.kernelDilatation)
        streets = scipy.ndimage.convolve(labels, kernel)
        # Filters pixels by quantile
        streets = (streets > np.quantile(streets, args.quantile)).astype(bool)
        # Mathematical morphology
        kernel  = skimage.morphology.disk(args.kernelErosion)
        streets = skimage.morphology.erosion(streets, kernel)
        # Post-processing
        streets = skimage.morphology.remove_small_holes(streets, args.maxHoleSize, connectivity=2)
        streets = skimage.morphology.remove_small_objects(streets, args.minObjectSize, connectivity=2)
        # Binary
        streets = np.where(labels > 0, 0, streets)
    else:
        streets = np.zeros(builds.shape, dtype=bool)
    if args.year == '1860':
        foo.array2raster(streets, files.landuse1860, files.street1860, noDataValue=0, dataType=gdal.GDT_Byte)
    if args.year == '2018':
        foo.array2raster(streets, files.landuse2018, files.street2018, noDataValue=0, dataType=gdal.GDT_Byte)
    
def seeStreets(tile:str, outlabel:str='', scem:bool=False, build1860:bool=False, build2018:bool=False, street1860:bool=False, street2018:bool=False, display:bool=False):
    paths = makePaths()
    files = makeFiles(tile, outlabel=outlabel, tmplabel='tmp')
    if scem:
        if not os.path.exists(files.scemCopy):
            shutil.copy(files.scem, files.scemCopy)
        if display:
            os.system('open {}'.format(files.scemCopy))
    if build1860:
        if not os.path.exists(files.build1860vec):
            landuse = foo.raster2array(files.landuse1860)
            landuse = np.isin(landuse, 1)
            foo.array2raster(landuse, files.landuse1860, files.tmp)
            foo.raster2vector(files.tmp, files.build1860vec, driver='GPKG', layerName='')
            shutil.copy(os.path.join(paths.styles, 'buildings_yh2.qml'), files.build1860vec.replace('gpkg', 'qml'))
        if display:
            os.system('open {}'.format(files.build1860vec))
    if build2018:
        if not os.path.exists(files.build2018vec):
            landuse = foo.raster2array(files.landuse2018)
            landuse = np.isin(landuse, 1)
            foo.array2raster(landuse, files.landuse2018, files.tmp)
            foo.raster2vector(files.tmp, files.build2018vec, driver='GPKG', layerName='')
            shutil.copy(os.path.join(paths.styles, 'buildings_yh2.qml'), files.build2018vec.replace('gpkg', 'qml'))
        if display:
            os.system('open {}'.format(files.build2018vec))
    if street1860:
        foo.raster2vector(files.street1860, files.street1860vec, driver='GPKG', layerName='')
        shutil.copy(os.path.join(paths.styles, 'street.qml'), files.street1860vec.replace('gpkg', 'qml'))
        if display:
            os.system('open {}'.format(files.street1860vec))
    if street2018:
        foo.raster2vector(files.street2018, files.street2018vec, driver='GPKG', layerName='')
        shutil.copy(os.path.join(paths.styles, 'street.qml'), files.street2018vec.replace('gpkg', 'qml'))
        if display:
            os.system('open {}'.format(files.street2018vec))

cities = {
  'Paris1':      '0640_6880',
  'Paris2':      '0640_6860',
  'Marseille':   '0880_6260',
  'Lyon1':       '0840_6540',
  'Lyon2':       '0840_6520',
  'Toulouse1':   '0560_6280',
  'Toulouse2':   '0560_6300',
  'Nice':        '1040_6300',
  'Nantes':      '0340_6700',
  'Montpellier': '0760_6280',
  'Strasbourg':  '1040_6860',
  'Bordeaux':    '0400_6440',
  'Lille1':      '0700_7060',
  'Lille2':      '0700_7080',
  'Rennes':      '0340_6800',
  'Reims':       '0760_6920',
  'St Etienne':  '0800_6500',
  'Toulon':      '0920_6240',
  'Le Havre':    '0480_6940',
  'Grenoble':    '0900_6460',
  'Angers':      '0420_6720',
  'Nimes':       '0800_6320',
  'Clermont1':   '0700_6520',
  'Clermont2':   '0700_6540',
  'Le Mans':     '0480_6780',
  'Aix-en-Prov': '0880_6280',
  'Brest':       '0140_6840',
  'Tours':       '0520_6720',
  'Amiens':      '0640_6980',
  'Poitiers':    '0480_6620',
  'Metz':        '0920_6900',
  'Rouen':       '0560_6940',
  'Orleans':     '0600_6760',
  'Valenciennes':'0720_7040',
  'Nancy':       '0920_6860',
  'Mulhouse':    '1020_6760',
  'Avignon':     '0840_6320'
  }
# 'Dijon':       '0840_6700'

#%% COMPUTE STREETS

paths = makePaths()

tiles = foo.getTiles(paths.landuse1860, paths.street1860, suffix2='_')
# Compute streets (kernel 5 baseline)
args = argparse.Namespace(year='1860', outlabel='_k7', kernelDilatation=9, kernelErosion=9, quantile=.0, maxHoleSize=10000, minObjectSize=1000)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(computeStreets, tiles, itertools.repeat(args))

tiles = foo.getTiles(paths.landuse2018, paths.street2018, suffix2='_')
args = argparse.Namespace(year='2018', outlabel='_k7', kernelDilatation=9, kernelErosion=9, quantile=.0, maxHoleSize=10000, minObjectSize=1000)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(computeStreets, tiles, itertools.repeat(args))

# Display streets
for tile in tiles:
    seeStreets(tile, scem=True, display=True)
for tile in tiles:
    seeStreets(tile, build1860=True, display=True)
for tile in tiles:
    seeStreets(tile, build2018=True, display=True)
for tile in tiles:
    seeStreets(tile, outlabel='_k9', street1860=True, display=False)
for tile in tiles:
    seeStreets(tile, outlabel='_k9', street2018=True, display=False)
# %%
