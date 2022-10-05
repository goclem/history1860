#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Augments training set with false negative from step 1
@author: Clement Gorin
@contact: gorinclem@gmail.com
@date: April 2021
'''

#%% MODULES

import argparse
import pandas as pd
import histo1860_functions as foo
import os
import shutil

# Paths
tile  = '0880_6260'
paths = foo.makePaths('landuse')
#%% FUNCTIONS

# Creates temporary file
def makeTempFile(tile, tmpLab="tmp", tmpExt="tif", reset=True):
    folder = os.path.join('/Users/clementgorin/Temporary', tile)
    file   = os.path.join(folder, "{}_{}.{}".format(tmpLab, tile, tmpExt))
    foo.resetFolder(folder)
    return(folder, file)

# Creates file paths for buildings
def makeFilesBuildings(tile):
    paths = foo.makePaths('buildings')
    files = argparse.Namespace(
        scem = os.path.join(paths.scem, 'scem_{}.tif'),
        y    = os.path.join(paths.y,    'y_{}.tif'),
        yh1  = os.path.join(paths.yh1,  'yh1_{}.tif'),
        yh2  = os.path.join(paths.yh2,  'yh2_{}.tif'),
        tmp  = os.path.join(paths.tmp,  'tmp_{}'),
        yh1_style = os.path.join(paths.styles, 'buildings_yh1.qml'),
        yh2_style = os.path.join(paths.styles, 'buildings_yh2.qml')
    )
    files = dict((k, v.format(tile, tile)) for k, v in vars(files).items())
    files = argparse.Namespace(**files)        
    return(files)

# Creates file paths for landuse
def makeFilesLanduse(tile):
    paths = foo.makePaths('landuse')
    files = argparse.Namespace(
        scem = os.path.join(paths.scem, 'scem_{}.tif'),
        y    = os.path.join(paths.y,    'y_{}.tif'),
        yh   = os.path.join(paths.data, 'yh', 'yh_{}.gpkg'),
        ids  = os.path.join(paths.ids,  'ids_{}.tif'),
        yStyle   = os.path.join(paths.styles, 'all_classes_raster.qml'),
        idStyle = os.path.join(paths.styles, 'superpixels.qml')
    )
    files = dict((k, v.format(tile, tile)) for k, v in vars(files).items())
    files = argparse.Namespace(**files)        
    return(files)

# Checks tile for landuse
def checkLanduse(tile, args):
    print(tile)
    files = makeFilesLanduse(tile)
    if args.scem:
        os.system('open {}'.format(files.scem))
    if args.yh and os.path.exists(files.yh):
        os.system('open {}'.format(files.yh))
    if args.y and os.path.exists(files.y):
        tmpFold, tmpFile = makeTempFile(tile, 'yView', 'tif')
        shutil.copyfile(files.y, tmpFile)
        shutil.copyfile(files.yStyle, tmpFile.replace("tif", "qml"))
        os.system('open {}'.format(tmpFile))
    if args.ids and os.path.exists(files.ids):
        tmpFold, tmpFile = makeTempFile(tile, 'idsView', 'gpkg')
        foo.raster2vector(files.ids, tmpFile, driver="GPKG", layerName='')
        shutil.copyfile(files.idStyle, tmpFile.replace("gpkg", "qml"))
        os.system('open {}'.format(tmpFile))

# Checks tile for buildings
def checkBuildings(tile, args):
    print(tile)
    files = makeFilesBuildings(tile)
    if args.scem:
        os.system('open {}'.format(files.scem))
    if args.yh1 and os.path.exists(files.yh1):
        outFile = files.tmp + "_yh1_view.gpkg"
        foo.raster2vector(files.yh1, outFile, driver="GPKG", layerName='')
        shutil.copyfile(files.yh1_style, outFile.replace("gpkg", "qml"))
        os.system('open {}'.format(outFile))
    if args.yh2 and os.path.exists(files.yh2):
        outFile = files.tmp + "_yh2_view.gpkg"
        foo.raster2vector(files.yh2, outFile, driver="GPKG", layerName='')
        shutil.copyfile(files.yh2_style, outFile.replace("gpkg", "qml"))
        os.system('open {}'.format(outFile))
    if args.yh and os.path.exists(files.yh):
        outFile = files.tmp + "_yh_view.tif"
        shutil.copyfile(files.yh, outFile)
        shutil.copyfile(files.yh_style, outFile.replace("tif", "qml"))
        os.system('open {}'.format(outFile))

cities = {
  'Paris1':      '0640_6880', # Red inside - Major fix
  'Paris2':      '0640_6860', # Red inside - Major fix
  'Marseille':   '0880_6260', # Red inside - OK
  'Lyon1':       '0840_6540', # Red inside - OK
  'Lyon2':       '0840_6520', # Red inside - OK
  'Toulouse1':   '0560_6280', # Red inside - Minor fix
  'Toulouse2':   '0560_6300', # Red inside - OK
  'Nice':        '1040_6300', # OK
  'Nantes':      '0340_6700', # Red inside - Major fix (to do)
  'Montpellier': '0760_6280', # Red inside - Really good
  'Strasbourg':  '1040_6860', # Red inside - Really good
  'Bordeaux':    '0400_6440', # Red inside - Major fix
  'Lille1':      '0700_7060', # Red inside - OK
  'Lille2':      '0700_7080', # Red inside - OK
  'Rennes':      '0340_6800', # Red inside - Really good
  'Reims':       '0760_6920', # Red inside - Major fix
  'St Etienne':  '0800_6500', # Red inside - OK
  'Toulon':      '0920_6240', # OK
  'Le Havre':    '0480_6940', # Red inside - Minor fix
  'Grenoble':    '0900_6460', # Red inside - Minor fix
  'Dijon':       '0840_6700', # Red inside - Minor fix
  'Angers':      '0420_6720', # OK
  'Nimes':       '0800_6320', # Red inside - Minor fix
  'Clermont1':   '0700_6520', # Red inside - OK
  'Clermont2':   '0700_6540', # Red inside - OK
  'Le Mans':     '0480_6780', # Red inside - OK
  'Aix-en-Prov': '0880_6280', # Red inside - Really good
  'Brest':       '0140_6840', # Red inside - Major fix
  'Tours':       '0520_6720', # Red inside - Minor fix
  'Amiens':      '0640_6980', # Red inside - Minor fix
  'Poitiers':    '0480_6620', # Red inside - Major fix
  'Metz':        '0920_6900', # Red inside - OK
  'Rouen':       '0560_6940', # Red inside - Major fix
  'Orleans':     '0600_6760', # Red inside - Major fix
  'Valenciennes':'0720_7040', # Red inside - Really good
  'Nancy':       '0920_6860', # Red inside - Minor fix
  'Mulhouse':    '1020_6760', # Red inside - Really good
  'Avignon':     '0840_6320', # Red inside - Really good
  'Quimper':     '0160_6800',
  'Versailles':  '0620_6860',
  'St-Germain':  '0620_6880',
  'Lagny':       '0620_6880',
  'Caen':        '0440_6920',
  'Bayeux':      '0420_6920',
  'Cherbourg':   '0360_6960',
  'Coutances':   '0360_6900',
  'La Rochelle1':'0360_6580',
  'La Rochelle2':'0380_6580',
  'Rochefort':   '0380_6560',
  'Saintes':     '0400_6540',
  'Cognac':      '0440_6520',
  'Bezriers':    '0700_6260',
  'Tarbes':      '0460_6260',
  'Chartres':    '0580_6820',
  'Etampes':     '0620_6820',
  'Beauvais':    '0620_6940',
  'Dunkerque':   '0640_7120',
  'Bergues':     '0660_7100',
  'Calais':      '0600_7100',
  'Montauban':   '0560_6340',
  'Angers':      '0420_6720',
  'Auxerre':     '0740_6760',
  'Lisieux':     '0480_6900',
  'Colmar1':     '1020_6800',
  'Colmar2':     '1000_6800',
  'Niort':       '0420_6660',
  'Troye1':      '0760_6800',
  'Troye2':      '0760_6820',
  'Troye3':      '0780_6820',
  'B-en-bresse': '0860_6580',
  'Besancon':    '0920_6700',
  'M-de-Marsan': '0400_6320',
  'Limoges':     '0560_6540',
  'Chateauroux': '0600_6640',
  'Laval':       '0400_6800'
  }

# cities = pd.DataFrame.from_dict(cities, orient='index', columns=['tile'])
# cities = cities.reset_index()
# cities.columns = ['city', 'tile']
# cities.to_csv(os.path.join(paths.data, 'buildings', 'fixes', 'cities.csv'), index=False)

#%% CITIES TO CHECK

tiles = list(cities.tile)
# filt  = foo.getTiles(paths.yh2)
# tiles = list(set(tiles) & set(filt))
len(tiles)
    
args = argparse.Namespace(scem=True, y=True, yh=True, ids=True)
checkLanduse(cities["Bordeaux"], args)
checkLanduse(cities["Reims"], args)
checkLanduse(cities["Lille1"], args)
checkLanduse(cities["Lille2"], args)
checkLanduse(cities["Lyon1"], args)
checkLanduse(cities["Lyon2"], args)
checkLanduse(cities["Paris2"], args)
checkLanduse(cities["Paris1"], args)

# Exports grayscale images
for tile in cities.values():
    scem = foo.raster2array(os.path.join(paths.scem, 'scem_{}.tif').format(tile))
    scem = skimage.color.rgb2gray(scem)
    scem = skimage.exposure.rescale_intensity(scem, out_range=(0, 255))
    scem = scem.astype(np.ubyte)
    foo.array2raster(scem, 
                     os.path.join(paths.scem, 'scem_{}.tif').format(tile),
                     os.path.join('/Users/clementgorin/Desktop/buildings1860/layers', 'scem_{}.tif').format(tile),
                     dataType=gdal.GDT_Byte)

# Copies SCEM images
for tile in cities.values():
    shutil.copy(
        os.path.join(paths.scem, 'scem_{}.tif').format(tile),
        os.path.join('/Users/clementgorin/Dropbox/research/arthisto/shared_data/1860/qgis_buildings1860/layers', 'scem_{}.tif').format(tile))




# Problèmes Marais salants
# 0320_6660
# 0300_6660