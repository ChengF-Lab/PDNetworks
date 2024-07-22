library(Seurat)
library(DoubletFinder)
library(tidyverse)
library(Matrix)
library(matrixStats)
set.seed(123)


# ============ Task definition ============
GEO_ID <- "GSE178265" 

# ============ Initial setup ============
file_path <- "./1_snRNA-seq_validation"  

PATH_O <- file.path(file_path, "Output230412",GEO_ID) 
PATH_O_data <- file.path(PATH_O,"Data") 
PATH_O_data_seq <- file.path(PATH_O_data,"data_seq") 
PATH_O_fig <- file.path(PATH_O,"Figure") 


so<-readRDS(file.path(PATH_O_data_seq,"1_so_all_SN_data_3_200_230119.rds"))

####PART1: QC process

#vlnplot to check data quality
Idents(so)=so$Indivi_ID
feas = c("nFeature_RNA", "nCount_RNA", "percent.mt","percent.ribo","percent.hb")
vlnplot=VlnPlot(so, pt.size=0,features = feas, ncol = 5)+NoLegend()
ggsave(vlnplot,filename = file.path(PATH_O_fig,"2_GSE178265_BeforeQC_Vln240312.png"),width =10,height = 4,dpi=100)
#scatter plot
plot1 <- FeatureScatter(so, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(so, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
ggsave(plot1+plot2,filename = file.path(PATH_O_fig,"2_GSE178265_beforeQC_FeaScatter240312.png"),width =10,height = 4,dpi=100)

# #boxplot
# C <- so@assays$RNA@counts
# C <- Matrix::t(Matrix::t(C)/Matrix::colSums(C)) * 100
#
# #most_expressed <- order(apply(C, 1, median), decreasing = T)[20:1] #it is time-consuming and use matrix to do it
# row_medians <- rowMedians(as.matrix(C))
# most_expressed <- order(row_medians, decreasing = TRUE)[20:1]
# #like this one
#
# png(file.path(PATH_O_fig,"2_GSE178265_beforeQC_mostexpressed240312.png"))
# boxplot(as.matrix(t(as.matrix(C[most_expressed, ]))), cex = 0.1, las = 1, xlab = "% total count per cell",
#     col = (scales::hue_pal())(20)[20:1], horizontal = TRUE)
# dev.off()


so <- subset(so, subset =
nFeature_RNA > 200 &
nFeature_RNA < 10000 &
nCount_RNA > 650 &  #REPORTED IN PAPER 
nCount_RNA<30000 &
percent.mt < 10 &
percent.hb<5 &
percent.ribo<5)



# Filter MALAT1
so <- so[!grepl("Malat1", rownames(so)), ]
Idents(so)=so$Indivi_ID

feas = c("nFeature_RNA", "nCount_RNA", "percent.mt","percent.ribo","percent.hb")
vlnplot=VlnPlot(so, pt.size=0,features = feas, ncol = 3)+NoLegend()
ggsave(vlnplot,filename = file.path(PATH_O_fig,"2_GSE178265_AfterQC_Vln240312.png"),width =10,height = 8,dpi=100)
#scatter plot
plot1 <- FeatureScatter(so, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(so, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
ggsave(plot1+plot2,filename = file.path(PATH_O_fig,"2_GSE178265_AfterQC_FeaScatter240312.png"),width =10,height =8,dpi=100)



#do doublets first and then do QC, it is more reasonable
############################Delete the doublets using DoubletFinder
#Step1: cluster  (normalize,pca,...)
#Step2: delete (define doubletrate for different samples and then identify doublets)

#1 define DR rate
DR_chose<-function(data,DRs){
cell_num=dim(data@meta.data)[1]
DR=cell_num*8*1e-6
print(paste("Cell number:",cell_num,"; DoubletRate:",DR,sep=""))
return(DR)
}

#2  find existed doublets
dim.usage=30
Find_doublet <- function(data){
#optimize pk value
sweep.res.list <- paramSweep_v3(data, PCs = 1:dim.usage, sct = FALSE)
sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
p<-as.numeric(as.vector(bcmvn[bcmvn$MeanBC==max(bcmvn$MeanBC),]$pK))
#homotypic doublet proportion estimate
annotations<-data@meta.data$seurat_clusters
homotypic.prop <- modelHomotypic(annotations)  #0.3149338
DoubletRate=DR_chose(data,DRs) ##give rate according to number of cells
nExp_poi <- round(DoubletRate*nrow(data@meta.data))
nExp_poi.adj <- round(nExp_poi*(1-homotypic.prop))
#nExp_poi
data <- doubletFinder_v3(data, PCs = 1:dim.usage, pN = 0.25, pK = p, nExp = nExp_poi, reuse.pANN = FALSE, sct = FALSE)
colnames(data@meta.data)[ncol(data@meta.data)-1] = "doubFind_score"   #change col of "pANNxxxx" to "doubFind_score"
colnames(data@meta.data)[ncol(data@meta.data)] = "doubFind_res"   #change col of "DF.classificaionsxxxx" to "doubFind_res"
#nExp_poi_adj
data <- doubletFinder_v3(data, PCs = 1:dim.usage, pN = 0.25, pK = p, nExp = nExp_poi.adj, reuse.pANN = FALSE, sct = FALSE)
colnames(data@meta.data)[ncol(data@meta.data)-1] = "doubFindadj_score"
colnames(data@meta.data)[ncol(data@meta.data)] = "doubFindadj_res"
return(data)
}


subsetList <- function(myList, elementNames) {   #for list merge
  sapply(elementNames, FUN=function(x) myList[[x]])
}

so.list<-SplitObject(so,split.by="Indivi_ID")
so.list<- lapply(X = so.list, FUN = function(x) {
#before delete doublet, cluster first
x <- NormalizeData(x)
x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
x <- ScaleData(x)
x <- RunPCA(x)
x <- RunUMAP(x, dims = 1:dim.usage)
x <- FindNeighbors(x,reduction="pca",k.param = 30,  dims = 1:dim.usage)
x <- FindClusters(x, resolution = 0.2) 
})






dim.usage=30
so <- merge(x = so.list[[1]], y = c(subsetList(so.list, seq(2, length(so.list)))))
so <- NormalizeData(so)
so <- FindVariableFeatures(so, selection.method = "vst", nfeatures = 2000)
so <- ScaleData(so)
so <- RunPCA(so,reduction.name = 'PCA_doublets', reduction.key = 'PCA_doublets_')
so <- RunUMAP(so, reduction ='PCA_doublets',  reduction.name = 'UMAP_doublets',dims = 1:dim.usage)
so <- FindNeighbors(so,reduction="PCA_doublets",k.param = 30,  dims = 1:dim.usage)
so <- FindClusters(so, resolution = 0.2)


p1=DimPlot(so,group.by="doubFind_res",reduction="UMAP_doublets",raster= FALSE)
p2=DimPlot(so,group.by="doubFindadj_res",reduction="UMAP_doublets",raster= FALSE)
p3=DimPlot(so,label=T,reduction="UMAP_doublets",raster= FALSE)
p4=DimPlot(so,label=FALSE,group.by="Indivi_ID",reduction="UMAP_doublets",raster= FALSE)
ggsave(p1+p2+p3+p4,filename = file.path(PATH_O_fig,"2_GSE178265_Umap_doublets240312.png"),width =25, height = 16,dpi=100)
saveRDS(so, file = file.path(PATH_O_data_seq,"2_GSE178265_QC_doublets_240312.rds"),compress=F)






