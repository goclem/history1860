# ARTHISTO DATA PREPARATION

pacman::p_load(data.table, dplyr, sf, stringr)
setwd("~/Dropbox/research/arthisto/data_1850")
options(scipen = 999)

# Merges BDHISTO shapefiles ----------------------------------------------

# After downlaoding merges the shapefiles of all departements into a single one
merge_shapefiles <- function(srcpattern, outlayer) {
  srcfiles <- list.files("~/Downloads", pattern = srcpattern, full.names = T, recursive = T)
  srcfiles <- str_subset(srcfiles, "GEOREF_NIVEAU3")
  outfile1 <- sprintf("/Users/clementgorin/Dropbox/data/ign_cartoem/%s.shp",  outlayer)
  outfile2 <- sprintf("/Users/clementgorin/Dropbox/data/ign_cartoem/%s.gpkg", outlayer)
  commands <- sprintf("ogr2ogr -f 'ESRI Shapefile' -s_srs EPSG:2154 -t_srs EPSG:2154 %s %s", outfile1, srcfiles[1])
  commands <- c(commands, sprintf("ogr2ogr -f 'ESRI Shapefile' -s_srs EPSG:2154 -t_srs EPSG:2154 -update -append %s %s", outfile1, srcfiles[-1]))
  commands <- c(commands, sprintf("ogr2ogr -f GPKG -s_srs EPSG:2154 -t_srs EPSG:2154 %s %s", outfile2, outfile1))
  for(command in commands) {
    print(command)
    system(command)
  }
}

# Merges all shapefiles
merge_shapefiles("A_1_LIMITE_ADMINISTRATIVE.shp$", "limite_administrative")
merge_shapefiles("A_2_SURFACE_ADMINISTRATIVE.shp$", "surface_administrative")
merge_shapefiles("B_1_TRONCON_DE_ROUTE.shp$", "troncon_de_route")
merge_shapefiles("B_2_TRONCON_DE_VOIE_FERREE.shp$", "troncon_de_voie_ferree")
merge_shapefiles("B_3_TRONCON_DE_COURS_D_EAU.shp$", "troncon_de_cours_d_eau")
merge_shapefiles("C_1_1_OCS_ANCIEN.shp$", "ocs_ancien")
merge_shapefiles("C_1_2_OCS_ANCIEN_SANS_BATI.shp$", "ocs_ancien_sans_bati")
merge_shapefiles("C_2_1_BATIMENT.shp$", "batiment")
merge_shapefiles("C_2_2_BATIMENT_HORS_ZONE_URBAINE.shp$", "batiment_hors_zone_urbaine")
merge_shapefiles("C_3_ZONE_URBAINE.shp$", "zone_urbaine")

tmpfiles <- list.files("~/Dropbox/data/ign_bdcartoEM", "\\.(shp|dbf|prj|shx)$", full.names = T)
file.remove(tmpfiles, recursive = T)

# Computes building blocs -------------------------------------------------

pacman::p_load(dplyr, sf, stringr)
setwd("~/Dropbox/data/ign_bdcartoEM")

build <- st_read("batiment.gpkg")
bloc  <- filter(build, !(CLEABS %in% st_read("batiment_hors_zone_urbaine.gpkg")$CLEABS))
bloc  <- mutate(bloc, SURFACE = round(st_area(bloc)))
bloc  <- mutate(bloc, SURFACE_QUANTILE = as.numeric(cut(SURFACE, quantile(SURFACE, probs = seq(0, 1, .005)), include.lowest = T)))
bloc  <- mutate(bloc, SURFACE_QUANTILE = seq(0, 995, 5)[match(bloc$SURFACE_QUANTILE, 1:200)])
bloc  <- filter(bloc, SURFACE_QUANTILE >= 990)
st_write(bloc, "batiment_bloc.gpkg", delete_dsn = T)

# Computes buffers --------------------------------------------------------

pacman::p_load(dplyr, sf, stringr)
setwd("~/Dropbox/data/ign_bdcartoEM")

layers  <- c("limite_administrative.gpkg", "troncon_de_route.gpkg", "troncon_de_voie_ferree.gpkg", "troncon_de_cours_d_eau.gpkg")
lapply(layers, function(layer) {
  vector <- st_read(layer, quiet = T)
  vector <- st_buffer(vector, 4)
  st_write(vector, layer, delete_dsn = T, quiet = T)
})

# Creates tiles files -----------------------------------------------------

pacman::p_load(data.table, dplyr, future.apply, raster, sf)

tiles <- st_read("~/Dropbox/research/arthisto/data_1860/tiles/scem_tiles.gpkg")
tiles <- tiles %>% mutate(tile = str_remove_all(NOM, "SCEM40K_|_L93")) %>% select(tile)
st_write(tiles, "~/Dropbox/research/arthisto/data_1860/tiles/scem_tiles.gpkg", delete_dsn = T)

# BDCARTOEM
dep <- st_read("~/Dropbox/data/osm_departement/departement_2021/departement_2021.shp")
dep <- dep %>%
  dplyr::select(INSEE_DEP) %>%
  rename(code_insee = INSEE_DEP) %>%
  filter(code_insee %in% c("02", "03", "07", 13, 15, 18, 22, 29, 42, 43, 48, 56, 59, 62, 63, 74)) %>%
  st_transform(2154)
ohs <- st_intersection(tiles, dep)
st_write(ohs, "~/Dropbox/research/arthisto/data_1860/tiles/ohs_tiles.gpkg", delete_dsn = T)

# FIXES
fixes <- c('0640_6880', '0640_6860', '0880_6260', '0840_6540', '0840_6520', '0560_6280', '0560_6300', '1040_6300', '0340_6700', '0760_6280', '1040_6860', '0400_6440', '0700_7060', '0700_7080', '0340_6800', '0760_6920', '0800_6500', '0920_6240', '0480_6940', '0900_6460', '0840_6700', '0420_6720', '0800_6320', '0700_6520', '0700_6540', '0480_6780', '0880_6280', '0140_6840', '0520_6720', '0640_6980', '0480_6620', '0920_6900', '0560_6940', '0600_6760', '0720_7040', '0920_6860', '1020_6760', '0840_6320')
fixes <- tiles %>%
  subset(tile %in% fixes)
st_write(fixes, "~/Dropbox/research/arthisto/data_1860/tiles/fixes.gpkg", delete_dsn = T)

# ohs2cls -----------------------------------------------------------------

pacman::p_load(data.table, sf, dplyr, stringr)
setwd("~/Dropbox/research/arthisto/data_1860")

computeAreas <- function(vec) {
  areas <- vec %>%
    select(THEME) %>%
    rename(ohs_id = THEME) %>%
    mutate(ohs_area = st_area(.)) %>%
    st_drop_geometry() %>% 
    group_by(ohs_id) %>% 
    summarise(ohs_area = sum(ohs_area)) %>%
    data.table()
  return(areas)
}

lab <- fread("~/Dropbox/research/arthisto/data_1860/ohs_lab.csv")
lab <- mutate(lab, ohs_label = iconv(ohs_label, to = "ASCII//TRANSLIT"))
lab <- mutate(lab, ohs_label = str_remove_all(ohs_label, "\\^"))
lab <- mutate(lab, ohs_label = str_replace_all(ohs_label, "'e|`e", "e"))
lab <- mutate(lab, ohs_label = str_replace_all(ohs_label, "'E", "E"))
ohs <- st_read("~/Dropbox/data/ign_ohs/merged_layers.gpkg", quiet = T)
ohs <- computeAreas(ohs)
ohs <- mutate(ohs, ohs_area = round(as.numeric(ohs_area / sum(ohs_area) * 100), 4))
ohs <- merge(ohs, lab, by = "ohs_id", all.x = T)
ohs <- ohs[order(-ohs_area), c("ohs_id", "ohs_label", "ohs_area"), with = F]
ohs <- cbind(data.table(cls_id = 0, cls_lab = ""), ohs)

fwrite(ohs, "~/Dropbox/research/arthisto/data_1860/ohs2cls.csv", sep = ";")
info <- fread("~/Dropbox/research/arthisto/data_1860/ohs2cls.csv")[, c("ohs_id", "cls_id", "cls_lab", "details")]
ohs  <- merge(ohs[, -c("cls_id", "cls_lab")], info, by = "ohs_id", all.x = T)

# Export to LaTeX
ohs2cls <- fread("~/Dropbox/research/arthisto/data_1860/ohs2cls.csv")[, -c("details")]
ohs2cls <- ohs2cls[order(cls_id, -ohs_area)]
ohs2cls <- rbind(ohs2cls[cls_id != 0], ohs2cls[cls_id == 0])
ohs2cls <- setNames(ohs2cls, c("BDCarto id", "BDCarto label", "BDCarto area", "Class id", "Class label"))
ohs2cls <- ohs2cls[, c("Class id", "Class label", "BDCarto id", "BDCarto label", "BDCarto area")]

print(xtable::xtable(data.frame(ohs2cls), digits=4), include.rownames=FALSE)

# Simplified commune shapefile --------------------------------------------

pacman::p_load(dplyr, sf)

com19 <- st_read("~/Dropbox/data/communes_2019/com19.shp") %>%
  select("insee") %>% 
  st_transform(2154) %>% 
  st_simplify(dTolerance = 25) %>%
  st_cast("MULTILINESTRING") %>%
  st_write("~/Dropbox/research/arthisto/data/com19/com19_simple.shp")

# Match OHS classes with SCEM classes -----------------------------------------

pacman::p_load(raster, rgeos)

ohs      <- shapefile("~/Dropbox/research/arthisto/data_1850/ign/BDCARTO_EM_1-0__SHP_LAMB93_D013_2018-10-01/BDCARTO_EM/1_DONNEES_LIVRAISON_2019/BDCEM_1-0_SHP_LAMB93_D013/GEOREF_NIVEAU3/C_ENTITES_SURFACIQUES_OCS/C_1_2_OCS_ANCIEN_SANS_BATI.shp")

# Computes area
ohs$area <- gArea(ohs, byid = T)
dat      <- ohs@data[, c("THEME", "area")]
dat      <- aggregate(area ~ THEME, dat, sum, na.rm = T)
dat$area <- dat$area / sum(dat$area) * 100

# Creates database
theme   <- lapply(strsplit(dat$THEME, "EM"), function(.) as.data.table(t(.))[, -1])
theme   <- setnames(rbindlist(theme, fill = T), c("theme_id1", "theme_id2"))
theme[] <- lapply(theme, function(.) ifelse(nchar(.) != 0, paste0("EM", .), .))
dat     <- cbind(theme, "share_area" = round(dat$area, 4))

# Merge with all classes
theme <- fread("~/Desktop/themes.csv")
theme2 <- merge(theme, dat, by = c("theme_id1", "theme_id2"), all.x = T, sort = F)
theme2$share_area[is.na(theme2$share_area)] <- 0
theme2$share_area <- round(theme2$share_area, 4)
fwrite(theme2, "~/Desktop/themes2.csv", sep = ";")

# Reference raster --------------------------------------------------------

pacman::p_load(data.table, raster)

ca200 <- fread("/Users/clementgorin/Dropbox/research/artmetro/data/raw/ca200.csv", select = c("idinseeca200", "xcar", "ycar", "principal_99"))
ca200 <- ca200[complete.cases(ca200)][order(ycar, xcar)]
ca200[, `:=`(xcar = xcar + 100, ycar = ycar + 100)]
ca200[, caid := seq(nrow(ca200))]

ca200_tot <- rasterFromXYZ(ca200[, c("xcar", "ycar", "caid")], crs = CRS("+init=epsg:3035"))
writeRaster(ca200_tot, "~/Dropbox/research/arthisto/data/ca200_tot.tif", overwrite = T)
saveRDS(ca200, "~/Dropbox/research/arthisto/data/ca200_ids.rds")

# Rasters communes --------------------------------------------------------

pacman::p_load(data.table, raster)

ca200 <- raster("~/Dropbox/research/arthisto/data/ca200_tot.tif")
com11 <- setNames(shapefile("~/Dropbox/data/communes_2011/com11.shp")[, "INSEE_COM"], "comid")
com11 <- spTransform(com11, crs(ca200))

caxy    <- rasterToPoints(ca200, spatial = T)
idx     <- over(caxy, com11) # (!) time
comca11 <- setValues(ca200, as.numeric(idx$comid[match(getValues(ca200), caxy[[1]])]))

writeRaster(comca11, "~/Dropbox/research/arthisto/data/comca_2011.tif", overwrite = T)


# Average year ----------------------------------------------------------
# Utilisation du tableau d'assemblage

(66 * mean(c(1818, 1835)) + 86 * mean(c(1835, 1845)) + 81 * mean(c(1845, 1855)) + 38 * mean(c(1855, 1866))) / (66 + 86 + 81 + 38)

# Seprarate class files ---------------------------------------------------

pacman::p_load(sf, dplyr)

ocs <- st_read("/Users/clementgorin/Dropbox/data/ign_cartoEM/ohs_ocs_ancien_sans_bati.gpkg")
ocs <- ocs %>% select(THEME) %>% rename(ohs_id = THEME)
ocs <- merge(ocs, lab, all.x = T)
ocs <- split(ocs, ocs[["ohs_label"]])

lapply(names(ocs), function(nam) st_write(ocs[[nam]], file.path("~/Desktop/ohs", str_c(nam, ".gpkg"))))

# Copies style file -------------------------------------------------------

pacman::p_load(stringr)
setwd("~/Dropbox/research/arthisto/data_1860")

rsts <- dir("y", pattern = "\\.tif$", full.names = T)
rsts <- str_replace(rsts, "\\.tif$", ".qml")
file.copy("styles/all.qml", rsts)

