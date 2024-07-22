library(Seurat)
set.seed(123)
library(DoubletFinder)
library(tidyverse)
library(Matrix)
library(matrixStats)
library(harmony)
library(cowplot)
library(ggplot2)
library(clustree)
library(paletteer)
library(ggrepel)
library(tidydr)



# ============ Task definition ============
GEO_ID <- "GSE178265" 

# ============ Initial setup ============
file_path <- "./1_snRNA-seq_validation"  

PATH_O <- file.path(file_path, "Output230412",GEO_ID) 
PATH_O_data <- file.path(PATH_O,"Data") 
PATH_O_data_seq <- file.path(PATH_O_data,"data_seq") 
PATH_O_DEGs<- file.path(PATH_O_data,"DEGs") 
PATH_O_fig <- file.path(PATH_O,"Figure") 


so<-readRDS(file = file.path(PATH_O_data_seq,"2_GSE178265_QC_doublets_240312.rds"))


p1=DimPlot(so,group.by="doubFind_res",reduction="UMAP_doublets",raster= FALSE)
p2=DimPlot(so,group.by="doubFindadj_res",reduction="UMAP_doublets",raster= FALSE)
p3=DimPlot(so,label=T,reduction="UMAP_doublets",raster= FALSE)
p4=DimPlot(so,label=FALSE,group.by="Indivi_ID",reduction="UMAP_doublets",raster= FALSE)
ggsave(p1+p2+p3+p4,filename = file.path(PATH_O_fig,"3_GSE178265_Umap_doublets240312.png"),width =25, height = 16,dpi=100)


so <- subset(so, subset =
doubFindadj_res =="Singlet")   


#####harmony process based on Indivi_ID
dim.usage=30

so@meta.data$Indivi_ID<-as.factor(so@meta.data$Indivi_ID)  
so <- RunPCA(so,reduction.name = 'PCA', reduction.key = 'PCA_')
so <- RunHarmony(so, reduction="PCA",c("Indivi_ID"),reduction.save = "harmony_Indivi_ID")
so <- RunUMAP(so,  dims = 1:dim.usage,reduction = "harmony_Indivi_ID",reduction.name = 'UMAP_harmony_Indivi_ID')



p2=DimPlot(so,reduction="UMAP_doublets",group.by="Indivi_ID",raster=FALSE)
p2_3=DimPlot(so,reduction="UMAP_harmony_Indivi_ID",group.by="Indivi_ID",label=F,raster=FALSE)
p=plot_grid(p2+p2_3, ncol = 1)
ggsave(p,filename = file.path(PATH_O_fig,"3_GSE178265_Umap_cluster_Group_diff_batch_harmony240312.png"),width =10,height = 5,dpi=100)


so <- FindNeighbors(so,reduction="harmony_Indivi_ID",graph.name="harmony_Indivi_ID_knn",k.param = 30,  dims = 1:dim.usage)
so <- FindClusters(so, graph.name="harmony_Indivi_ID_knn",resolution = c(0.05,0.1,0.2))

p3_1=DimPlot(so,reduction="UMAP_harmony_Indivi_ID",group.by="harmony_Indivi_ID_knn_res.0.05",label=TRUE,raster=FALSE)
p3_2=DimPlot(so,reduction="UMAP_harmony_Indivi_ID",group.by="harmony_Indivi_ID_knn_res.0.1",label=TRUE,raster=FALSE)
p3_3=DimPlot(so,reduction="UMAP_harmony_Indivi_ID",group.by="harmony_Indivi_ID_knn_res.0.2",label=TRUE,raster=FALSE)

p=plot_grid(p3_1+p3_2+p3_3, ncol = 1)
ggsave(p,filename = file.path(PATH_O_fig,"3_GSE178265_Umap_cluster_Group_diff_batch_harmony_diff_reso240312.png"),width =10,height = 6,dpi=100)
saveRDS(so, file = file.path(PATH_O_data_seq,"3_GSE178265_harmony_240312.rds"),compress=FALSE)


