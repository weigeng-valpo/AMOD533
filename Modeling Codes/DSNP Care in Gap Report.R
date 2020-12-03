# ---
#   title: "R Notebook"
# output: html_notebook
# ---
#   
#   GIC Report September:
#   #August Run : The change in the script is that we parse the file by market and then get top40, care managers etc. 
#   
#   ```{r}
library(rmarkdown)
setwd("C:/Users/A427813/Desktop/Python/DSNP GIC Report/Input_File")
library(readxl)
library(readr)
DSNP_Comprehensive_Report_20201014 <- read_csv("//mbip/medicarepBI/Projects/COE/DSNP/Projects/DSNP CM Comprehensive Report/DSNP_Comprehensive_Report_20201014.csv", 
                                               col_types = cols(Reporting_Date = col_character()))


Original_data = DSNP_Comprehensive_Report_20201014
rm(DSNP_Comprehensive_Report_20201014)

str(Original_data)
library(dplyr)
library(lubridate)
Original_data =  Original_data %>% mutate_if(is.POSIXct,as.Date, na.rm = TRUE)
library(dplyr)
library(lubridate)
Original_data = Original_data %>% mutate_if(is.Date,as.character, na.rm = TRUE)
Original_data[Original_data == "1970-01-01"] <- ""
Original_data = filter(Original_data,!duplicated(MEDICARE_NUMBER))
table(is.na(Original_data$Manager))
table(Original_data$Contract_Number,is.na(Original_data$Assigned_Primary_Contact_Name))
#Since there are members with UnAssigned contact name, 
Original_data$Manager[is.na(Original_data$Manager)] <- "UnAssigned"
#Since there are members with UnAssigned contact name, 
Original_data$RNPM_Name[is.na(Original_data$RNPM_Name)] <- "UnAssigned"
#UnAssigned Primary contact name
Original_data$Assigned_Primary_Contact_Name[is.na(Original_data$Assigned_Primary_Contact_Name)] <- "UnAssigned"
# ```
# 
# The following code parses the data file by Markets. Use the column "Contract_Number"
# ```{r}
table(Original_data$Contract_Number)
Split_by_Contract_Number = split(Original_data,Original_data$Contract_Number)
names(Split_by_Contract_Number)

#Need not run this code as the test is to see if all code can be done to manipute using lists and only the final ouput file is split to multiple dataframes at the end
# #names(Split_by_Manager)[[1]] = "No_Manager"
# list(Split_by_Manager,
#          envir = globalenv())
# ```
# 
# The following code calculates the top 40 high priority members per RNPM
# ```{r}
#In a function
library(dplyr)
library(tidyr)
Split_by_Contract_Number = lapply(Split_by_Contract_Number, function(x) x[with(x, rev(order(x$High_Prioritize_Member,x$Prioritize_Indx))),])
top40 = lapply(Split_by_Contract_Number, function(x) x  %>% slice(1:40))  
# ```
# #Changes within lists:
# 1. Remove the High_Prioritize_Member, Prioritize_Index columns from all three lists. 
# 
# ```{r}
Split_by_Contract_Number = lapply(Split_by_Contract_Number, function(x)
  x[!(names(x) %in% c("High_Prioritize_Member","Prioritize_Indx"))])
top40 = lapply(top40, function(x)
  x[!(names(x) %in% c("High_Prioritize_Member","Prioritize_Indx"))])
#split_primary_assigned_contact = lapply(split_primary_assigned_contact, function(x)
# x[!(names(x) %in% c("High_Prioritize_Member","Prioritize_Indx"))])
# ```
# .Instead add Priority Index column with 1:n
# ```{r}
Split_by_Contract_Number = lapply(Split_by_Contract_Number, function(x)
  x %>% mutate(Priority_ID = row_number()))
top40 = lapply(top40, function(x)
  x %>% mutate(Priority_ID = row_number()))


#Make sure the priority index is the first column
library(dplyr)
Split_by_Contract_Number = lapply(Split_by_Contract_Number, function(x)
  x <- x[,c(which(colnames(x)=="Priority_ID"),1:ncol(x)-1)])
top40 = lapply(top40, function(x)
  x <- x[,c(which(colnames(x)=="Priority_ID"),1:ncol(x)-1)])


#top40_test = lapply(top40, function(x)
# x %>% mutate(Priority_ID = row_number()))

```

The following code splits the Supervisor list into further data frames by the RNPM name
```{r}
split_primary_assigned_contact = lapply(Split_by_Contract_Number, function(x) split(x,x$RNPM_Name))

split_primary_assigned_contact =  sapply(1:length(split_primary_assigned_contact),
                                         function(i) purrr::keep(split_primary_assigned_contact[[i]], ~nrow(.) > 0))
```

#Change the Priority Index column to Top40_Flag and add the priority index for the specified primary care manager
```{r}
split_primary_assigned_contact = lapply(split_primary_assigned_contact, lapply,
                                        function(x)  x %>% mutate(Top_40_Flag = if_else(Priority_ID <= 40,'Y','N')))
split_primary_assigned_contact = lapply(split_primary_assigned_contact, lapply,
                                        function(x)  x %>% mutate(Priority_ID = row_number()))
split_primary_assigned_contact = lapply(split_primary_assigned_contact, lapply,
                                        function(x) x <- x[,c(which(colnames(x)=="Top_40_Flag"),1:ncol(x)-1)])
```

Merge Lists
```{r}
Split_by_Supervisor_test = list()
name_vec = names(Split_by_Contract_Number)
i = 1
for(i in 1:11){
  Split_by_Supervisor_test[[i]] = list(Split_by_Contract_Number[[i]],top40[[i]])
  names(Split_by_Supervisor_test[[i]]) = list(name_vec[i],"Top40")}

View(Split_by_Supervisor_test)
i = 1
j = 1
for(i in 1:11){
  for(j in 1:length(split_primary_assigned_contact[[i]])){
    Split_by_Supervisor_test[[i]][[2+j]]= split_primary_assigned_contact[[i]][[j]]}}
#   attr(Split_by_Supervisor_test[[i]],"names") = c(name_vec[i],"Top40",split_primary_assigned_contact[[i]])
#   }
# }
attr(Split_by_Supervisor_test[[1]],"names")
```

#Rename the tabs 
```{r}
names(split_primary_assigned_contact[[1]])
attr(Split_by_Supervisor_test[[1]],"names") = c("H1609", "Top40","Priscilla Weightman","UnAssigned")

```


#Second item
```{r}
names(split_primary_assigned_contact[[2]])
attr(Split_by_Supervisor_test[[2]],"names") = c("H1692","Top40","Jim McPhilomy","Mary Kay Brown","Maureen Duncan","Temi Osabiya","UnAssigned")
```

#Third item
```{r}
names(split_primary_assigned_contact[[3]])
attr(Split_by_Supervisor_test[[3]],"names") = c("H3146", "Top40","Andrea Johnson","UnAssigned")
```

#Fourth item
```{r}
names(split_primary_assigned_contact[[4]])
attr(Split_by_Supervisor_test[[4]],"names") = c("H3239", "Top40","Mary Kay Brown","Priscilla Weightman","UnAssigned")
```



#Fifth item
```{r}
names(split_primary_assigned_contact[[5]])
attr(Split_by_Supervisor_test[[5]],"names") = c("H3959" ,"Top40","Ann Charbonneau", "Jim McPhilomy", "Lisa Viggiano", "Lori Wilson",     
                                                "Mary Kay Brown", "Mary Lee Trafican", "Maureen Duncan", "Monica Lynch", "UnAssigned")
```


#Sixth item
```{r}
names(split_primary_assigned_contact[[6]])
attr(Split_by_Supervisor_test[[6]],"names") = c("H5302", "Top40","Andrea Johnson","Deb Prosser", "Priscilla Weightman",
                                                "UnAssigned"  )
```


#Seventh item
```{r}
names(split_primary_assigned_contact[[7]])
attr(Split_by_Supervisor_test[[7]],"names") = c("H5325","Top40","Angela Walby", "Cyndi Archer", "Lori Wilson", "Tina Kinney", "UnAssigned") 
```


#Eight item
```{r}
names(split_primary_assigned_contact[[8]])
attr(Split_by_Supervisor_test[[8]],"names") = c("H5337", "Top40","Jim McPhilomy", "Kasie Blindauer", "Mary Kay Brown",  "Sarah Gunder",   
                                                "Sue Schieberl", "Temi Osabiya", "UnAssigned")
```

#Ninth item
```{r}
names(split_primary_assigned_contact[[9]])
attr(Split_by_Supervisor_test[[9]],"names") = c( "H5593", "Top40","Cyndi Archer","June Wilwert","UnAssigned")
```

#Tenth item
```{r}
names(split_primary_assigned_contact[[10]])
attr(Split_by_Supervisor_test[[10]],"names") = c( "H7149", "Top40", "Cyndi Archer","June Wilwert","UnAssigned")
```


#Eleventh item
```{r}
names(split_primary_assigned_contact[[11]])
attr(Split_by_Supervisor_test[[11]],"names") = c("H8597" ,"Top40","Angela Walby","Marie Thomas","UnAssigned")
```

#Set folder for outfiles and export all to excel format
```{r}
setwd("C:/Users/A427813/Desktop/Python/DSNP GIC Report/Gaps_In_Care_Report")

library(writexl)

sapply(1:length(Split_by_Supervisor_test),function(i) write_xlsx(Split_by_Supervisor_test[[i]],path = paste0(name_vec[i],'.xlsx'),col_names = T, format_headers = TRUE))
```
