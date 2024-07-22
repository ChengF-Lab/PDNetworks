library(Seurat)
library(DoubletFinder)
library(tidyverse)
library(Matrix)
library(matrixStats)
library(SingleCellExperiment)
library(dplyr)
library(data.table)
library(ggplot2)
library(ggrepel)

set.seed(123)



# ============ Task definition ============
GEO_ID <- "GSE178265" 

# ============ Initial setup ============
file_path <- "./1_snRNA-seq_validation"  
PATH_O <- file.path(file_path, "Output230412",GEO_ID) 
PATH_O_data <- file.path(PATH_O,"Data") 
PATH_O_data_seq <- file.path(PATH_O_data,"data_seq") 
PATH_O_DEGs<- file.path(PATH_O_data,"DEGs") 
PATH_O_DEGs_seurat<- file.path(PATH_O_DEGs,"DEGs_seurat") 
PATH_O_fig <- file.path(PATH_O,"Figure") 
PATH_O_fig_DEGs=file.path(PATH_O_fig,"DEGs") 




celltypes=c("DA","non-DA","Astro","MG","OPC","ODC","Endo_Peri_Fib")
num_celltypes=length(celltypes)
compare_group1 =c("Ctrl")
compare_group2=c("Disease")
compare_group=paste0(compare_group2," vs. ",compare_group1)
num_compare_group=length(compare_group)



so<-readRDS(file = file.path(PATH_O_data_seq,"4_GSE178265_celltype_240312.rds"))

cut_off=0.25

so$Group_Celltype=paste(so$Celltype_har_0.1,so$Group,sep="_")
Idents(so)=so$Group_Celltype
DEGs_data_all=data.frame()

for (i in 1:num_celltypes){
    for (j in 1:num_compare_group){
celltype=celltypes[i]
group1=compare_group1[j]
group2=compare_group2[j]
Celltype_Group1=paste(celltype,group1,sep="_")
Celltype_Group2=paste(celltype,group2,sep="_")
DEGs <- FindMarkers(so,ident.1 =Celltype_Group2, ident.2 = Celltype_Group1,logfc.threshold=0.1)  #usually ident. vs. ident.2 , so we need to give right ident.1 and ident.2
DEGs$celltype=celltype
DEGs$group1=group2
DEGs$group2=group1
DEGs$compare_group =compare_group[j]
DEGS_file=file.path(PATH_O_DEGs_seurat,paste0("GSE178265_DEGs_celltypes_",celltype,"_",group2,"_vs_",group1,"240412_seurat.tsv"))
write.table(DEGs, file = DEGS_file, quote = FALSE, sep = "\t", col.names = NA)
if (dim(DEGs_data_all)[1]=0){DEGs_data_all=DEGs} else{DEGs_data_all=rbind(DEGs_data_all,DEGs)}
cat(celltype,"is done")
}
}

DEGS_all_file=file.path(PATH_O_DEGs_seurat,paste0("GSE178265_DEGs_celltypes_all_240412_seurat.tsv"))
write.table(DEGs_data_all, file = DEGS_all_file, quote = FALSE, sep = "\t", col.names = NA)




###: DA vs others in PD samples
groups_need=c("Disease")
so_sub <- subset(so, subset = Group %in% groups_need)
so_sub$Celltypes_DA_mark=ifelse(so_sub$Celltype_har_0.1=="DA","DA","Others")
celltypes=c("DA","Others")
Idents(so_sub)=so_sub$Celltypes_DA_mark
DEGs <- FindMarkers(so_sub,ident.1 ="DA", ident.2 = "Others",logfc.threshold=0.1)

DEGs$group1="DA"
DEGs$group2="Others"
DEGs$compare_group ="DA vs.Others"
DEGS_file=file.path(PATH_O_DEGs_seurat,paste0("GSE178265_DEGs_celltypes_DA_vs_others_240412_seurat.tsv"))
write.table(DEGs, file = DEGS_file, quote = FALSE, sep = "\t", col.names = NA)
