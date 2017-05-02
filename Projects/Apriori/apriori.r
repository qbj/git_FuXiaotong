library(arules)

setwd('C:\\Users\\fu\\Documents\\R\\Demo\\Apriori2')

#transaction database
tnx_db<-read.transactions('demo_basket',sep=',')
head(as(tnx_db,'matrix'))
colnames(as(tnx_db,'matrix'))

#find rules
rules<-apriori(
  tnx_db,
  parameter=list(support=0.05)
)

#support, confidence, lift
inspect(rules)
