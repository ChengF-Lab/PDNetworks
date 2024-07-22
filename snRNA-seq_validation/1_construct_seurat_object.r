###construct seurat object
#Load package
library("Seurat")
library(DoubletFinder)
library(future)
# plan(strategy = 'multiprocess', workers = 20)
# options(future.globals.maxSize = 500 * 1024^3)
set.seed(123)


#Load data and give sample information
# ============ Task definition ============
GEO_ID <- "GSE178265" # change this to the GEO id, i.e., folder name

# ============ Initial setup ============
file_path <- "./1_snRNA-seq_validation"  #give general dir of input file, which further includes GSE file of different sample (related details of dir, age, sex ,etc. are recorded in "sample_info.tsv"  )

PATH_O <- file.path(file_path, "Output230412",GEO_ID) # output file dictonary
PATH_O_data <- file.path(PATH_O,"Data") # output file dictonary
PATH_O_data_seq <- file.path(PATH_O_data,"data_seq") # output file dictonary
PATH_O_fig <- file.path(PATH_O,"Figure") # outpu



so.data <- Read10X(data.dir = "./1_snRNA-seq_validation/Input_snRAN_seq_data/GSE178265_RAW/Human")  #three files of barcodes, genes, and matrixs
so <- CreateSeuratObject(counts = so.data, min.cells=3, min.features=200,project = "GSE178265")





#add metedata, such as sex, age, ...

so_ls <- list()  # subset divided by orig.ident
so_orig.ident <-  unique(so@meta.data$orig.ident)
for (i in 1:length(so_orig.ident)){
curr_so <- subset(so,orig.ident == so_orig.ident[i])
so_ls[[i]] <-curr_so
}



sample_info=read.csv(file.path(PATH_O_data_seq,"Sample_info_1_220719.csv"),header=TRUE)
label=0
for (i in 1:length(so_ls)){
for (j in 1:dim(sample_info)[1]){
if (grepl(sample_info$Sample_title[j],so_ls[[i]]@meta.data$orig.ident[i]))  {  
so_ls[[i]]@meta.data$GSM_ID<- sample_info$GSM_ID[j]
so_ls[[i]]@meta.data$Indivi_ID<- sample_info$Indivi_ID[j]
so_ls[[i]]@meta.data$Tissue<- sample_info$Tissue[j]
so_ls[[i]]@meta.data$Sex<- sample_info$Sex[j]
so_ls[[i]]@meta.data$Age<- sample_info$Age[j]
so_ls[[i]]@meta.data$PMI<- sample_info$PMI[j]
so_ls[[i]]@meta.data$Group<- sample_info$Group[j]
so_ls[[i]]@meta.data$Status<- sample_info$Status[j]
so_ls[[i]]@meta.data$Death<- sample_info$Death[j]
so_ls[[i]]@meta.data$Facs<- sample_info$Facs[j]
so_ls[[i]][["percent.mt"]] <- PercentageFeatureSet(so_ls[[i]], pattern = "^MT-")
so_ls[[i]][["percent.ribo"]] <- PercentageFeatureSet(so_ls[[i]], pattern = "^RP[SL]")
so_ls[[i]][["percent.hb"]] <- PercentageFeatureSet(so_ls[[i]], pattern = "^HB[^(P)]")
label<-1
if (label ==1){
break}
}
}
}



subsetList <- function(myList, elementNames) {
  sapply(elementNames, FUN=function(x) myList[[x]])
}
#give the default sample ID by index
generateID <- function(ele) {
  return(paste0("Sample_", ele))
}


so_all <- merge(x = so_ls[[1]], y = c(subsetList(so_ls, seq(2, length(so_ls)))), add.cell.ids = sapply(seq(1, length(so_ls)),generateID))

SN <- subset(so_all,subset= Tissue== 'SN')
saveRDS(SN, file =file.path(PATH_O_data_seq,"1_so_all_SN_data_3_200_230119.rds"),compress=F)  
