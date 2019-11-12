# Gang Salvador Project: Image Analysis
# Fabiola Alba Vivar and Furkan Top
# If running from another compute, please instal packages

#Library
#devtools::install_github("dkahle/ggmap")
library(sf)
library(digest)
library(raster)
library(dplyr)
library(tmap)    # for static and interactive maps
library(leaflet) # for interactive maps
library(mapview) # for interactive maps
library(ggplot2) # tidyverse data visualization package
library(shiny)   # for web applications
library(rgdal)
library(ggmap)
library(maptools)
library(foreign)
library(readxl)
library(tidyverse)

##### First: Setting Up the program

## Setting Working Directory
setwd("/")

## Google API Key
# Save Api Key (Fabiola's)
register_google(key = "")
## Importing ShapeFiles and Data 

# San Salvador City Shapefile
shp <- readOGR(dsn ="san_salvador_segments.shp", stringsAsFactors = F)
# Gang Bounds from Micaela's Project
bound <- readOGR(dsn ="gang_territory_greater_san_salvador.shp", stringsAsFactors = F)
# Fieldwork Sample
fieldwork <-read_excel("survey_observations.xlsx")

# Check Data
summary(shp@data)
summary(shp@polygons)
summary(fieldwork)

## Drawing  El Salvador
map_Salvador <- ggplot() + geom_polygon(data = shp, aes(x = long, y = lat, group = group), 
                               colour = "black", fill = NA)

map_Gang <- ggplot() + geom_polygon(data = bound, aes(x = long, y = lat, group = group),
                               colour = "blue", fill = NA)

# Combining
map.Gang.Bound <- ggplot() +  geom_polygon(data = shp, aes(x = long, y = lat, group = group), 
                                           colour = "grey", fill = NA,  position = "identity")+
  geom_polygon(data = bound, aes(x = long, y = lat, group = group),
                 colour = "blue", fill = NA, position = "identity")
  

map.field <- ggplot(fieldwork, aes(x=long, y=lat)) +  geom_point(size = 0.3) + 
  xlab('Longitude') + 
  ylab('Latitude') +
  coord_cartesian(ylim = c(13.64, 13.73), 
                xlim = c(-89.255, -89.17))

map.field
map_Salvador+ theme_void()
map_Gang + theme_void()

## We can see that 
map.Gang.Bound + theme_void()

## Now let's compile information form google earth
# San Salvador Satellite View

ggmap(
  ggmap = get_map(
    "San Salvador",
    zoom = 13, scale = "auto",
    maptype = "satellite",
    source = "google"),
  extent = "device",
  legend = "topright"
)

## Now with the Gang Bounds
# Note: One point is off. 

# Gang Satellite
Gang.Boud<-qmap('San Salvador', maptype = 'satellite', zoom = 16) +
  geom_polygon(data = bound, aes(x = long, y = lat, group = group),
               colour = "blue", fill = NA, position = "identity")
Gang.Boud + theme_void()

# Gang Hybrid
Gang.Boud.Hybrid<-qmap('San Salvador', maptype = 'hybrid', zoom = 16) +
  geom_polygon(data = bound, aes(x = long, y = lat, group = group),
               colour = "blue", fill = NA, position = "identity")
Gang.Boud.Hybrid + theme_void()

## Now can we try to sample some areas around does boundaries:(using Mica's sample)
## generate id
fieldwork<-mutate(fieldwork,id=rownames(fieldwork)) 
id <- fieldwork$id
df<-fieldwork %>%  unite(center,lat,long,sep = ",")
center<-df$center

#for(i in 1:length(id))

## Extracting Google Earth Images from fieldwork 
for(i in 1501:length(id)){
png(file = paste('GE_',id[i],'.png', sep ='')) 
print(ggmap(
  ggmap = get_map(location = center[i],
                  zoom = 20, scale = "auto",
                  maptype = "satellite",
                  source = "google"),
  extent = "device",
  legend = "topright"
)
)
dev.off()
}

#for(i in 1:length(id))
for(i in 1:2)
{
  ggmap(
    ggmap = get_map(location = cbind(lat[i],long[i]),
                    zoom = 20, scale = "auto",
                    destfile = paste(i,".png",sep=""),
                    maptype = "satellite",
                    source = "google"),
    extent = "device",
    legend = "topright"
  )
}

pdf("Test.pdf")
  ggmap(
  ggmap = get_map(location = c(-89.19627,13.67125),
                  zoom = 20, scale = "auto",
                  maptype = "satellite",
                  source = "google"),
  extent = "device"
)
dev.off()

#####################################################################
# Now let's test this data frame for Image Analysis in Python

#First, let's create a excel file 

fieldwork<-mutate(fieldwork, file=paste("GE_",id,".png",sep=""))
write.csv(fieldwork,"sample.csv")
