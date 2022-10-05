

pacman::p_load(sf)
setwd("~/Dropbox/research/arthisto/data_1860/yh")

# Water vectors
files <- dir(pattern="gpkg$")
file  <- "yh_0400_6440.gpkg"
for(file in files) {
  print(file)
  vec <- st_read(file, quiet = T)
  vec <- filter(vec, class == 8)
  st_write(vec, file.path("~/tmp", str_replace(file, "yh", "water")), quiet = T, delete_dsn = T)
}
"cd ~/tmp"
"ogrmerge.py -single -f GPKG -o rivers1860.gpkg *.gpkg"

# Extract buildings
pacman::p_load(sf, stringr, dplyr)

srcFiles <- dir("~/Dropbox/research/arthisto/data_1860/yh", pattern=".gpkg$", full.names = T)
outFiles <- sprintf("~/tmp/%s", str_replace(basename(srcFiles), "yh", "build"))

for(i in seq(srcFiles)) {
  print(basename(srcFiles[i]))
  vecs <- st_read(srcFiles[i], quiet = T)
  if(any(vecs$class == 1)) {
    vecs <- vecs[vecs$class == 1, ]
    st_write(vecs, outFiles[i], quiet = T, delete_dsn = T)
  }
}

vecs <- dir("~/tmp", pattern = "gpkg$", full.names = T)
vecs <- lapply(vecs, function(vec) st_read(vec, quiet = T))
vecs <- bind_rows(vecs)
st_write(vecs, "~/Desktop/buildings1860.gpkg", quiet = T, delete_dsn = T)