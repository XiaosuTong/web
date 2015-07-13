
## install packages required to build the docs
#install.packages("devtools")
#library(devtools)
#install_github("staticdocs", "hadley")
#install_github("buildDocs", "hafen")


## load the pacakge
library(buildDocs)

files <- list.files("./docs")
## function to build the docs
#   assuming your working directory is 
#   doc.RHIPE/docs/
buildDocs(
   docsLoc       = "./docs",
   outLoc        = "./",
   pageList      = files,
   copyrightText = "Xiaosu Tong"
)
