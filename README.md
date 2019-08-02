# wnv_SES

This is the dev branch where we can combine code before committing it to master

## Instruction to run R linear regression (ncsa.RMD):
 ### library needed: 
install.packages('faraway')

install.packages('MASS')

install.packages('leaps')

install.packages('caret') 

Install R and R studio. Open the R markdown files via R studio. Run these codes before excuating. It should work if system has depencies installed, 
or otherwise google it.

The files I used to build the model is 'data.csv', which is Under the path Data/R_model_data/. 

Noted that the code should get an error if you run it without changing the path of 'data.csv' when reading it. Change the path to where it is in your PC.

## Instrcution to run R weather data(weather_data.RMD)

### library needed: 

library(reshape2) ##melting dataframes

library(dplyr) #data wrangling

library(raster) ##working with raster data

library(sp) ##manipulationg spatial data

library(prism) ##prism data access

The code should be able to run without any modification once you install or libraies successfully. However, you need to wait until they install all data needed. The path I put those installed data is "~/prismtmp", you can manually change them if you want.



