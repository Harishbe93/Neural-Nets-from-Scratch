data=iris
library(caret)

index=createDataPartition(data$Species,p=0.70,list=F)
train=data[index,]
test=data[-index,]

distance <- function(ex1,ex2){
	sqrt(sum((ex1-ex2)^2))
}

Neighbours <- function(train,test,K,ind){
	predictions=vector()
	for(i in 1:nrow(test)){
		dist=data.frame(class=character(),distance=numeric())
		t1=as.numeric(test[i,-ind])
		for(j in 1:nrow(train)){
			t2=as.numeric(train[j,-ind])
			dis=distance(t1,t2)
			dist=rbind(dist,data.frame(class=train[j,ind],distance=dis))
		}
		dist=dist[order(dist$distance),]
		topK=dist[1:K,]
		predictions[i]=names(sort(-table(topK$class)))[1]	
	}
	return(predictions)
}

pred=Neighbours(train,test,3,5)
confusionMatrix(test$Species,pred)
