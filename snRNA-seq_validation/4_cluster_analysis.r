library(Seurat)
library(DoubletFinder)
library(tidyverse)
library(Matrix)
library(matrixStats)
library(harmony)
library(cowplot)
library(clustree)
library(ggrepel)
library(tidydr)
library(paletteer)
set.seed(123)



# ============ Task definition ============
GEO_ID <- "GSE178265"

# ============ Initial setup ============
file_path <- "./1_snRNA-seq_validation"  
PATH_O <- file.path(file_path, "Output230412",GEO_ID) 
PATH_O_data <- file.path(PATH_O,"Data") 
PATH_O_data_seq <- file.path(PATH_O_data,"data_seq") 
PATH_O_fig <- file.path(PATH_O,"Figure")


so<-readRDS(file = file.path(PATH_O_data_seq,"3_GSE178265_harmony_240312.rds"))


so$KNN_cluster=so$harmony_Indivi_ID_knn_res.0.1
Idents(so)=so$KNN_cluster

## ============ Cell type identification ============
#try different markers

MARKERS <- unique(c("SLC18A2","SLC6A3","TH","RBFOX3","AQP4","CX3CR1","VCAN","OLIG1","CLDN5"))
MARKERS_SN_nn <- unique(c("SLC18A2","SLC6A3","TH","RBFOX3","AQP4","CX3CR1","VCAN","OLIG1","CLDN5"))
MARKERS_SN_nc <- unique(c("GFAP","OLR1","GINS3","TH","SLC6A3","RGS5","GAD1","GAD2","CSF1R","MOG","MOBP","PALM2","LGALS1","PPM1G",'VCAN'))
MARKERS_SN_bioRxiv <- unique(c("SLC17A6","GAD2","SLC6A3","TH","SLC18A2","DCC","GALNTL6","RIT2","RBFOX3","AQP4","C3","MOG","VCAN","FLT1","PDGFRB","COL1A2","SKAP1"))
MARKERS_SN_bioRxiv2 <- unique(c("RBFOX3","GAD1","NRGN","AQP4","GFAP","MOG","C3","CSF1R","CD74","TYROBP","VCAN","FLT1","PDGFRB"))
MARKERS_SN_cell <- unique(c("AQP4","MGP","PLXDC1","LGAM2","P2RY12","TH","SYT1","GAD2","MOBP","OLIG1"))
MARKERS_SN_all<- unique(append(MARKERS_SN_nn,MARKERS_SN_nc))
MARKERS_SN_all<- unique(append(MARKERS_SN_all,MARKERS_SN_bioRxiv))
MARKERS_SN_all<- unique(append(MARKERS_SN_all,MARKERS_SN_bioRxiv2))
MARKERS_SN_all<- unique(append(MARKERS_SN_all,MARKERS_SN_cell))

MARKERS_SN_GSE17_har0.1=unique(c("SLC18A2","SLC6A3","TH","RBFOX3","AQP4","C3","P2RY12","CX3CR1","VCAN","OLIG1","MOBP","MOG","CLDN5","FLT1","COL1A2","MGP"))

p0<-DotPlot(so, features =MARKERS_SN_GSE17_har0.1)+RotatedAxis()
svg(file.path(PATH_O_fig, "4_GSE178265_DotPlot0_markergenes.svg"),height=7,width=10)
p0
dev.off()




so <- RenameIdents(
  so,
`0` = "ODC",
`1` = "non-DA",
`2` = "Astro",
`3` = "MG",
`4` = "OPC",
`5` = "Endo_Peri_Fib",
`6` = "DA",
`7` = "non-DA",
`8` = "non-DA",
`9` = "non-DA",
`10` = "non-DA",
`11` = "non-DA",
`12` = "MG",
`13` = "non-DA"
)

so$Celltype_har_0.1=Idents(so)

p0<-DotPlot(so, features =MARKERS_SN_GSE17_har0.1)+RotatedAxis()
svg(file.path(PATH_O_fig, "4_GSE178265_DotPlot_markergenes.svg"),height=7,width=10)
p0
dev.off()

saveRDS(so, file = file.path(PATH_O_data_seq,"4_GSE178265_celltype_240312.rds"),compress=F)






































































































































































































































































































































