library(VIM) ##Viewing and imputing omitted values (KNN imputation )
library(ggplot2) #Data visualization
library(purrr) #Use vectors
library(tidyr) #Data structures
library(psych) #Graphics
library(e1071)# Calculate skewness
library(neuralnet)#Use neural networks 
library(MASS)# For multiple linear regression
library(reshape2) #transform data 
library(car) # Use Regression
library(caret)#Use cross-validation
library(Metrics)#Use decision trees
library(partykit)#Use decision trees 
library(rpart)#Use decision trees 
library(rpart.plot) #Use decision trees  
library(caret)# For modeling 
library(dplyr)# For data manipulation
library(plyr) # Mapvalues() function 
library(corrplot)#Correlation chart
library(cowplot)#Grid plot
library(gbm) #Importance of variables  chart


df<-read.csv('df_Pequim.csv',header = TRUE, sep=',',dec = '.')

data<-df


#Dataset dimension 
dim(data)

#Variables Names
names(data)

#Preview the first 6 lines 
head(data)

#Remove variable Hotel Name
data<-data[,-1]

#View type of variables
str(data)

#View variables descriptive values
summary(data)

#Convert logical variables to 0 and 1 since machine learning algorithms deal better with numerical values.
data$cancel<-data$cancel*1
data$rooms<-data$rooms*1
data$breakfast<-data$breakfast*1

#Histogram of all numeric variables
data[-c(3,4,5)] %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()
#In first analysis of the histograms, we can verify that there is skewness in the distribution of data in the following variables: distance, number of comments and price. 
#The most adequate solution to deal with this problem will be to use the logarithm of the values observed in these variables.
# # The distribution of the stars variable also indicates that it has to be transformed into an ordinal categorical variable.


#Treat missing values of the score review variable using the nearest neighbors imputation method
set.seed(123)
summary(data)
imputdata1 <- data
imputdata1 <- kNN(data, variable = "score_review", k = 17)
summary(imputdata1)

#K is chosen through the square root of the number of total observations in the dataset
summary(imputdata1)
ncol(imputdata1) 
head(imputdata1)


#A new column of logical values has been added at the end of the dataset, it must be removed
imputdata1 <- subset(imputdata1,select = price:stars)
head(imputdata1)
summary(imputdata1)
data <- imputdata1

#Confirm that there is no NA's
apply(data,2,function(x) sum(is.na(x)))

#View again the Histogram of all numeric variables after the treatment of missing data   
data[-c(3,4,5)] %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()
#variables price, distance and number of comments do not have a normal distribution and seem to indicate the existence of outliers,
#later we will proceed with the treatment of this problem.


## Visualization of the variable stars with ggplot -geometric density
ggplot(data, aes(stars)) + geom_density(adjust = 1/5)
#The density analysis of the variable suggests the transformation of the variable into an ordinal categorical
#Some accommodations do not have a star rating, in order to solve this problem we proceed with the creation
#of containers [0, 1-3, 4-5] in order to convert the variable stars into categorical ordinal.
#Create categories:
data$stars[data$stars==0]<-'0'
data$stars[data$stars>=1 & data$stars<=3]<-'1-3'
data$stars[data$stars>=4 & data$stars<=5]<-'4-5'

#De forma a melhorar a performance dos algoritmos de aprendizagem automática de seguida procede-se a realização de ordinal encoding 
#para a variável estrelas.  
#Função para realizar ordinal encoding
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}
#Aplicar ordinal encoding à variável estrelas 
data$stars<- encode_ordinal(data$stars, order = c('0', '1-3', '4-5'))



#Visualizar correlações entre variáveis numéricas através do valor de correlações e scatter plot
pairs.panels(data[c(1,2,6,7)], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = FALSE,  # show density plots
             ellipses = FALSE # show correlation ellipses
)

## Visualização da variavel score_review com ggplot -geometric density
ggplot(data, aes(score_review)) + geom_density(adjust = 1/5)
#A analise de densidade da variavel score_review sugere a não transformação da variável em categórica ordinal 
#A transformação da variável em continua categórica irá fazer perder o detalhe de informação apresentado entre 
#o secore 6 e 9.5.Deste modo irá ser mantida como numérica continua.     


#Verificar os quartis das variáveis numéricas
quantile(data$price)
quantile(data$distance)
quantile(data$ncomments)

#Pela análise dos histogramas parece existir alguns outliers, vamos usar o boxplot para cada variável numérica para confirmar 
#which devolve a linha do hotel que e outlier
#Tem que se usar a variável df com os dados originais para identificar os hotéis pertencentes aos outlieres 
#Verificar quais os outliers da variavel price
ggplot(data, aes(x="", y=price))+geom_boxplot()
df[which(data$price>quantile(data$price,prob=0.75)+1.5*(quantile(data$price,prob=0.75)-quantile(data$price,prob=0.25))),c(1,2)]
df[which(data$price<quantile(data$price,prob=0.25)-1.5*(quantile(data$price,prob=0.75)-quantile(data$price,prob=0.25))),c(1,2)]

#Verificar quais os outliers da variável distance
ggplot(data, aes(x="", y=distance))+geom_boxplot()
df[which(data$distance>quantile(data$distance,prob=0.75)+1.5*(quantile(data$distance,prob=0.75)-quantile(data$distance,prob=0.25))),c(1,7)]
df[which(data$distance<quantile(data$distance,prob=0.25)-1.5*(quantile(data$distance,prob=0.75)-quantile(data$distance,prob=0.25))),c(1,7)]

#Verificar quais os outliers da variável number of comments
ggplot(data, aes(x="", y=ncomments))+geom_boxplot()
df[which(data$ncomments>quantile(data$ncomments,prob=0.75)+1.5*(quantile(data$ncomments,prob=0.75)-quantile(data$ncomments,prob=0.25))),c(1,3)]
df[which(data$ncomments<quantile(data$ncomments,prob=0.25)-1.5*(quantile(data$ncomments,prob=0.75)-quantile(data$ncomments,prob=0.25))),c(1,3)]
# Através da análise manual de cada outlier foi possível identificar que todas as observações 
# Resultam de dados corretamente extraídos do website booking.com 

#De seguida procede-se a remoção de 1 outlier situado em posição muito extrema de forma a melhorar a distribuição da Variabilidade dos dados
#Remoção da observação com pereço igual a 1130 
data<-data[-c(which(data$price==1130)), ]


#Através da análise de distribuição e outlieres foi possível verificar que que algo tem que ser feito para corrigir a distribuição dos dados, 
#uma vez que temos uma assimetria dos dados bastante elevada com muitos outlieres. Algoritmos de machine learning não lidam bem com distribuições assimétricas e outliers, Desta forma vai de seguida proceder-se ao logaritmo das observações 
#de das variaveis distancia, numero de comentários e preço. 
#Verificar novamente as distribuições das varáveis antes de proceder ao logaritmo das variáveis  
data[-c(3,4,5)] %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#De seguida separa-se um Dataframe sem transformação de logaritmo para caso necessário mais tarde se poder usar 
data_no_log<-data

#Aplicar os logaritmos às variaveis.  
data$distance<-log(data$distance)
data$ncomments<-log(data$ncomments+1)
data$price<-log(data$price)
#Verificar a assimetria da distribuição 
skewness(data$distance)
skewness(data$ncomments)
skewness(data$price)

#Verificar a distribuição depois de serem aplicados os logaritmos
data[-c(3,4,5)] %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#Visualizar novamente correlações entre variáveis numéricas através do valor de correlações e scatter plot após aplicação do Logaritmo  
pairs.panels(data[c(1,2,6,7)], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = FALSE,  # show density plots
             ellipses = FALSE # show correlation ellipses
)

#Visualizar correlações entre todas as variáveis 
pairs.panels(data, 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = FALSE,  # show density plots
             ellipses = FALSE # show correlation ellipses
)

#Verificar novamente outliers após aplicação do Logaritmo  
ggplot(data, aes(x="", y=distance))+geom_boxplot()
ggplot(data, aes(x="", y=ncomments))+geom_boxplot()
ggplot(data, aes(x="", y=price))+geom_boxplot()
#Após a aplicação do logaritmos nas variáveis distancia, numero de comentários e preço o problema de normalidade e outliers melhorou significativamente, 
#obtendo uma distribuição quase normal para as variáveis e reduzindo imenso o numero e distancias dos outliers.  


#De seguida procede-se à separação entre treino e teste 
## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))
## Definir a semente para obter sempre a mesma separação 
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]

#Normalização dos dados de treino.(Media 0 Desvio Padrão 1) 
train_scale<-data_no_log[train_ind, ]
for (i in 1:ncol(train)) train_scale[,i]<-(train[,i]-mean(train[,i]))/sd(train[,i]) 
#Verificar 
round(colMeans(train_scale[,c(1:ncol(train_scale))]),2)
apply(train_scale,2,sd)
#Histograma de dados de treino Com estandardização MinMax
train_scale[-c(3,4,5)] %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()


#Normalização Min-Max
train_MinMax <- data_no_log[train_ind, ]
for (i in 1:ncol(train)) train_MinMax[,i]<-(train[,i]-min(train[,i]))/(max(train[,i])-min(train[,i])) 
#Verificar 
apply(train_MinMax,2,max)
apply(train_MinMax,2,min)
# Podemos verificar que todas as variaveis foram estandardizada. 
#Histograma de dados de treino Com estandardização MinMax
train_MinMax[-c(3,4,5)] %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

#Dados de treino sem logaritmo
train_no_log <- data_no_log[train_ind, ]

############################################################################################################################
############################################################################################################################
###############                                   Notas importantes                                      ###################
############################################################################################################################
############################################################################################################################
###############                     O logaritmo foi aplicado ao treino e ao teste                        ################### 
############################################################################################################################
############################################################################################################################
############################################################################################################################
###############          Dependendo dos algoritmos de aprendizagem automática poderá ser utilizado       ###################
###############                               os diferentes datsets de treino:                           ###################
############################################################################################################################
############################################################################################################################
############################################################################################################################

#'train_no_log' -Dados de treino sem aplicação de logaritmo às variaveis Preço, distancia e numero de comentário 
#‘train’- Dados de treino com aplicação de logaritmo às variaveis Preço, distancia e numero de comentário. 
#‘train_scale’ - Dados de treino com normalização media 0, desvio padrão 1.
#‘train_MinMax’ - Dados de treino com estandardização Mínimo 0, máximo 1.


###########################################################################################################################
#                                            Regressao Linear Multipla                                                    #
###########################################################################################################################
# Regressao Linear multipla 
lm1 <- lm(train$price ~ ncomments+distance+score_review, data=train)
summary(lm1)


#Testar o modelo criado (lm1) com os dados não vistos
estimativas<-exp(predict(lm1,test[,-1]))
#Criação da tabela com os valores reais e estimativas da variável target preço
tabela<-data.frame(VReais=exp(test$price),VPrevistos=estimativas)

#Adição de uma coluna à tabela que indica os erros associados às estimativas de preço para os dados não vistos
tabela$error<-with(tabela,exp(test$price)-estimativas)
tabela
# Teste do Modelo criado - Cálculo do Erro Quadrático Médio do model lm1 para os dados não vistos (conjunto teste)
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Teste do Modelo criado - Cálculo da Raiz Quadrada do Erro Quadrático Médio do model lm1 para os dados não vistos (conjunto teste)
RMSE_teste<-sqrt(MSE_teste)
RMSE_teste
# Teste do Modelo criado - Cálculo da Erro Absoluto Médio do model lm1 para os dados não vistos (conjunto teste)
MAE_teste<-with(tabela,mean(abs(error)))
MAE_teste

#########################################################################################################################
#                                  Regressao Linear Multipla com método para trás e para a frente                       #
#########################################################################################################################

# Regressao Linear multipla 
lm <- lm(train$price ~., data=train)
# Stepwise regression model
lm <- stepAIC(lm, direction = "both", trace = FALSE)
summary(lm)

#Testar o modelo criado (lm) com os dados não vistos
estimativas<-exp(predict(lm,test[,-1]))
#Criação da tabela com os valores reais e estimativas da variável target preço
tabela<-data.frame(VReais=exp(test$price),VPrevistos=estimativas)

#Adição de uma coluna à tabela que indica os erros associados às estimativas de preço para os dados não vistos
tabela$error<-with(tabela,exp(test$price)-estimativas)
tabela

# Teste do Modelo criado - Cálculo do Erro Quadrático Médio do model lm para os dados não vistos (conjunto teste)
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Teste do Modelo criado - Cálculo da Raiz Quadrada do Erro Quadrático Médio do model lm para os dados não vistos (conjunto teste)
RMSE_teste<-sqrt(MSE_teste)
RMSE_teste
# Teste do Modelo criado - Cálculo da Erro Absoluto Médio do model lm para os dados não vistos (conjunto teste)
MAE_teste<-with(tabela,mean(abs(error)))
MAE_teste



#########################################################################################################################
#                                  Regressao Linear Multipla com método de reamostragem                                 #
#########################################################################################################################

# Método de reamostragem: Validação Cruzada (com k=10)
MR.control<-trainControl(method="cv",number=10)
# Validação Cruzada (com k=10) e apresentação do melhor modelo criado com todos os dados do conjunto Treino
model_fit<-train(price ~ ., data=train,method="lm",trControl=MR.control )

# Resumo do melhor Modelo de Regressão Linear criado durante as k=10 tentativas
model_fit
# Teste do Modelo criado (usando os dados do conjunto teste)
# Teste do Modelo criado - cálculo das estimativas da variável Preço para os dados não vistos do conjunto teste
estimativas<-exp(predict(model_fit,test))
# Teste do Modelo criado - Criação da tabela com os valores reais e estimativas da variável preço para os dados não vistos do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=estimativas)
# Teste do Modelo criado - Adição de uma coluna à tabela que indica aos erros associados às estimativas do preço para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-estimativas)
# Visualização da tabela
tabela
# Teste do Modelo criado - Cálculo do Erro Quadrático Médio do para os dados não vistos (conjunto teste)
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Teste do Modelo criado - Cálculo da Raiz Quadrada do Erro Quadrático Médio para os dados não vistos (conjunto teste)
RMSE_teste<-sqrt(MSE_teste)
RMSE_teste
# Teste do Modelo criado - Cálculo da Erro Absoluto Médio para os dados não vistos (conjunto teste)
MAE_teste<-with(tabela,mean(abs(error)))
MAE_teste


#############################################################################################################################
#                                            Rede Neuronal                                                                 #
#############################################################################################################################

#Rede Neuronal
set.seed(222)
# Treinar modelo net 
net<- neuralnet(train$price~., train, linear.output = TRUE,hidden=c(2,1),rep=1) 
#Fazer o Plot da neuralnet
plot(net)

#Testar o modelo criado (net) com os dados não vistos
estimativas<-exp(predict(net,test[,-1]))
#Criação da tabela com os valores reais e estimativas da variável target preço
tabela<-data.frame(VReais=exp(test$price),VPrevistos=estimativas)

#Adição de uma coluna à tabela que indica os erros associados às estimativas de preço para os dados não vistos
tabela$error<-with(tabela,exp(test$price)-estimativas)
tabela

# Teste do Modelo criado - Cálculo do Erro Quadrático Médio do modelo net para os dados não vistos (conjunto teste)
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
MSE.nn <- sum((exp(test$price) - estimativas)^2)/nrow(test)
MSE.nn
# Teste do Modelo criado - Cálculo da Raiz Quadrada do Erro Quadrático Médio do modelo net para os dados não vistos (conjunto teste)
RMSE_teste<-sqrt(MSE_teste)
RMSE_teste
# Teste do Modelo criado - Cálculo da Erro Absoluto Médio do modelo net para os dados não vistos (conjunto teste)
MAE_teste<-with(tabela,mean(abs(error)))
MAE_teste


#############################################################################################################################
#                                          Arvores de decisao                                                                #
#############################################################################################################################

set.seed(222)
# Criação do Modelo de Árvore Decisão, a partir do conjunto treino (train), recorrendo ao Algoritmo CART
model_tree<-rpart(formula = train$price~.,data=train,method="anova",control=rpart.control(xval = 10))
# Resumo do Modelo de Árvore de Decisão criado (model_tree), o qual permite determinar o nº total de nós da árvore
model_tree_party<-as.party(model_tree)
model_tree_party
rpart.plot(model_tree,yesno=TRUE)
printcp(model_tree)
# Teste ao Modelo criado (model_tree) com base nos dados não vistos do conjunto teste (test)
# Cálculo das previsões recorrendo ao modelo criado (model_tree)
model_tree_previsao<-exp(predict(model_tree,test,type="vector"))
plot(exp(test$price),model_tree_previsao,main="Árvore model_tree: Previstos vs Reais",xlab="Reais",ylab="Previstos",col=c('black','blue'))
abline(0,1, col='red')
# Teste do Modelo criado (model_tree): Criação da tabela com os valores reais e estimativas da variável preço para os dados não vistos 
#do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=model_tree_previsao)
# Teste do Modelo criado (model_tree): Adição de uma coluna à tabela que indica aos erros associados às estimativas do preço (obtidas com o model_tree) para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-model_tree_previsao)
# Visualização da tabela
tabela
#Calculo MSE
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Cálculo do RMSE
model_tree_rmse<-RMSE(pred=model_tree_previsao,obs=exp(test$price))
model_tree_rmse
# Cálculo do MAE
model_tree_mae<-MAE(pred=model_tree_previsao,obs=exp(test$price))
model_tree_mae

set.seed(222)
# Criação de uma árvore de decisão a partir do model_tree e da operação poda
# Criar a Tabela Complexidade e representação gráfica de cp vs xvalerror
a<-model_tree$cptable
plotcp(model_tree)
cp<-which.min(model_tree$cptable[,"xerror"])
cp
# Criação e Teste de um novo modelo de árvore
model_tree_prune<-prune(model_tree,cp=0.01503426)
# Resumo do Modelo de Árvore de Decisão criado (model_tree_prune), o qual permite determinar o nº total de nós da árvore
model_tree_party<-as.party(model_tree_prune)
model_tree_party
rpart.plot(model_tree_prune,yesno=TRUE)
# Teste do novo modelo de árvore (model_tree_prune)
# Cálculo das previsões recorrendo ao novo modelo criado (model_tree_prune)
model_tree_prune_previsao<-exp(predict(model_tree_prune,test,type="vector"))
plot(exp(test$price),model_tree_prune_previsao,main="Árvore model_tree_prune: Previstos vs Reais",xlab="Reais",ylab="Previstos",col=c('black','blue'))
abline(0,1, col='red')
# Teste do Modelo criado (model_tree_prune): Criação da tabela com os valores reais e estimativas da variável preço para os dados não vistos do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=model_tree_prune_previsao)
# Teste do novo modelo criado (model_tree_prune): Adição de uma coluna à tabela que indica aos erros associados às estimativas de Preço (obtidas com o model_tree_new) para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-model_tree_prune_previsao)
# Visualização da tabela
tabela
#Calculo MSE
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Cálculo do RMSE
model_tree_prune_rmse<-RMSE(pred=model_tree_prune_previsao,obs=exp(test$price))
model_tree_prune_rmse
# Cálculo do MAE
model_tree_prune_mae<-MAE(pred=model_tree_prune_previsao,obs=exp(test$price))
model_tree_prune_mae



##########################################################################################################################
###                               Arvores de decisão com metodo de validação cruzada.                                   #
##########################################################################################################################
set.seed(222)
cv.control<-trainControl(method="cv",number=10)
model_cv<-train(price ~.,data=train,method="rpart",metric="RMSE",tuneLength=5,trControl=cv.control)
#O RMSE e MAE apresentados no output seguinte encontram-se transformados em logaritmo, os valores na escala original encontram-se nos outputs seguintes 
model_cv
plot(model_cv)
# Teste ao Modelo criado (model_cv) com base nos dados não vistos do conjunto teste (test)
# Cálculo das previsões recorrendo ao modelo criado (model_cv)
model_cv_previsao<-exp(predict(model_cv,test))
plot(exp(test$price),model_cv_previsao,main="Árvore model_cv (k=10): Previstos vs Reais",xlab="Reais",ylab="Previstos",col=c('black','blue'))
abline(0,1, col='red')
# Teste do Modelo criado (model_cv): Criação da tabela com os valores reais e estimativas da variável Preço para os dados não vistos do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=model_cv_previsao)
# Teste do novo modelo criado (model_cv): Adição de uma coluna à tabela que indica aos erros associados às estimativas Preço (obtidas com o model_cv) para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-model_cv_previsao)
# Visualização da tabela
tabela
#Calculo MSE
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Cálculo do RMSE
model_cv_rmse<-RMSE(pred=model_cv_previsao,obs=exp(test$price))
model_cv_rmse
# Cálculo do MAE
model_cv_mae<-MAE(pred=model_cv_previsao,obs=exp(test$price))
model_cv_mae
# Representação Gráfica da melhor árvore obtida
rpart.plot(model_cv$finalModel)
plot(varImp(model_cv),main="Importância das Variáveis com a árvore model_cv (k=10)")


###########################################################################################################################
#                                                    Bagging                                                              #
###########################################################################################################################

set.seed(222)
# Criação de um modelo de árvore recorrendo ao método Bagging
cv.control<-trainControl(method="cv",number=10,savePredictions="final")
model_bag<-train(price ~.,data=train,method="treebag",nbagg=80,metric="MAE",tuneLength=5,trControl=cv.control)
#O RMSE e MAE apresentados no output seguinte encontram-se transformados em logaritmo, os valores na escala original encontram-se nos outputs seguintes 
model_bag
# Teste ao Modelo criado (model_bag) com base nos dados não vistos do conjunto teste (test)
#
# Cálculo das previsões recorrendo ao modelo criado (model_cv)
model_bag_previsao<-exp(predict(model_bag,test))
plot(exp(test$price),model_bag_previsao,main="Árvore obtida com método Bagging: Previstos vs Reais",xlab="Reais",ylab="Previstos",col=c('black','blue'))
abline(0,1,col='red')

# Teste do Modelo criado (model_bag): Criação da tabela com os valores reais e estimativas da variável Preço para os dados não vistos do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=model_bag_previsao)
# Teste do novo modelo criado (model_bag): Adição de uma coluna à tabela que indica aos erros associados às estimativas de Preço (obtidas com o model_bag) 
#para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-model_bag_previsao)
# Visualização da tabela
tabela
#Calculo MSE
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Cálculo do RMSE
model_bag_rmse<-RMSE(pred=model_bag_previsao,obs=exp(test$price))
model_bag_rmse
# Cálculo do MAE
model_bag_mae<-MAE(pred=model_bag_previsao,obs=exp(test$price))
model_bag_mae
#Importância das variáveis
plot(varImp(model_bag),main="Importância das Variáveis com o modelo de árvore obtido com o método Bagging")


###########################################################################################################################
#                                  Arvore de decisão com metodo Florestas Aleatórias.                                                #
###########################################################################################################################


# Criação de um modelo de árvore recorrendo ao método Florestas Aleatórias
#
set.seed(222)
cv.control<-trainControl(method="cv",number=10,savePredictions="final")
model_forest<-train(price ~.,data=train,method="ranger",metric="RMSE", num.trees=300, importance = "impurity", tuneLength=5,trControl=cv.control)
model_forest

# Teste ao Modelo criado (model_forest) com base nos dados não vistos do conjunto teste (test)
#
# Cálculo das previsões recorrendo ao modelo criado (model_forest)
model_forest_previsao<-exp(predict(model_forest,test))
plot(exp(test$price),model_forest_previsao,main="Árvore obtida com método Florestas Aleatórias: Previstos vs Reais",xlab="Reais",ylab="Previstos",col=c('black','blue'))
abline(0,1,col='red')
# Teste do Modelo criado (model_forest): Criação da tabela com os valores reais e estimativas da variável Preço para os dados não vistos do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=model_forest_previsao)
# Teste do novo modelo criado (model_forest): Adição de uma coluna à tabela que indica os erros associados às estimativas de Preço (obtidas com o model_cv) para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-model_forest_previsao)
# Visualização da tabela
tabela
#Calculo MSE
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Cálculo do RMSE
model_forest_rmse<-RMSE(pred=model_forest_previsao,obs=exp(test$price))
model_forest_rmse
# Cálculo do MAE
model_forest_mae<-MAE(pred=model_forest_previsao,obs=exp(test$price))
model_forest_mae
#Importância das variáveis
plot(varImp(model_forest),main="Importância das Variáveis com o modelo de árvore obtido com o método Florestas Aleatórias")



###########################################################################################################################
#                    Arvore de decisão com metodo Florestas Aleatórias. (Com ajustamento de dos                           #
#                             hiperparamstreos de forma a melhora o desempenho do modelo)                                 #
###########################################################################################################################


# Criação de um modelo de árvore recorrendo ao método Florestas Aleatórias
#
set.seed(222)
cv.control<-trainControl(method="cv",number=10,savePredictions="final")
tune_forest<-expand.grid(mtry=c(1,2,3,4,5,6,7), splitrule='variance', min.node.size=c(1,3,5,8))
model_forest<-train(price ~.,data=train,method="ranger",metric="RMSE", num.trees=300, importance = "impurity", tuneLength=5,trControl=cv.control, tuneGrid=tune_forest)
model_forest

# Teste ao Modelo criado (model_forest) com base nos dados não vistos do conjunto teste (test)
#
# Cálculo das previsões recorrendo ao modelo criado (model_forest)
model_forest_previsao<-exp(predict(model_forest,test))
plot(exp(test$price),model_forest_previsao,main="Árvore obtida com método Florestas Aleatórias: Previstos vs Reais",xlab="Reais",ylab="Previstos",col=c('black','blue'))
abline(0,1,col='red')
# Teste do Modelo criado (model_forest): Criação da tabela com os valores reais e estimativas da variável Preço para os dados não vistos do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=model_forest_previsao)
# Teste do novo modelo criado (model_forest): Adição de uma coluna à tabela que indica os erros associados às estimativas de Preço (obtidas com o model_cv) para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-model_forest_previsao)
# Visualização da tabela
tabela
#Calculo MSE
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Cálculo do RMSE
model_forest_rmse<-RMSE(pred=model_forest_previsao,obs=exp(test$price))
model_forest_rmse
# Cálculo do MAE
model_forest_mae<-MAE(pred=model_forest_previsao,obs=exp(test$price))
model_forest_mae
#Importância das variáveis
plot(varImp(model_forest),main="Importância das Variáveis com o modelo de árvore obtido com o método Florestas Aleatórias")




###########################################################################################################################
#                       Boosting (Com ajustamento de dos Hiper parâmetros de forma a melhora o desempenho do modelo)      #                                       #
###########################################################################################################################


# Criação de um modelo de árvore recorrendo ao método Boosting
#
set.seed(222)
cv.control<-trainControl(method="cv",number=10,savePredictions="final")
tune_gbm<-expand.grid(interaction.depth=c(1,2,3,4,5), n.trees= c(100,200,500,1000),shrinkage=c(0.005,0.01,0.05,0.1,0.2,0.3), n.minobsinnode=c(5,7,10,12))
model_boosting<-train(price ~.,data=train,method="gbm",metric="RMSE", tuneLength=5,trControl=cv.control, tuneGrid=tune_gbm)
model_boosting

# Teste ao Modelo criado (model_boosting) com base nos dados não vistos do conjunto teste (test)
#
# Cálculo das previsões recorrendo ao modelo criado (model_boosting)
model_boosting_previsao<-exp(predict(model_boosting,test))
plot(exp(test$price),model_boosting_previsao,main="Árvore obtida com método Florestas Aleatórias: Previstos vs Reais",xlab="Reais",ylab="Previstos",col=c('black','blue'))
abline(0,1,col='red')
# Teste do Modelo criado (model_boosting): Criação da tabela com os valores reais e estimativas da variável Preço para os dados não vistos do conjunto teste
tabela<-data.frame(VReais=exp(test$price),VPrevistos=model_boosting_previsao)
# Teste do novo modelo criado (model_boosting): Adição de uma coluna à tabela que indica os erros associados às estimativas de Preço (obtidas com o model_boosting) para os dados não vistos do conjunto teste
tabela$error<-with(tabela,exp(test$price)-model_boosting_previsao)
# Visualização da tabela
tabela
#Calculo MSE
MSE_teste<-with(tabela,mean(error^2))
MSE_teste
# Cálculo do RMSE
model_boosting_rmse<-RMSE(pred=model_boosting_previsao,obs=exp(test$price))
model_boosting_rmse
# Cálculo do MAE
model_boosting_mae<-MAE(pred=model_boosting_previsao,obs=exp(test$price))
model_boosting_mae
#Importância das variáveis
plot(varImp(model_boosting),main="Importância das Variáveis com o modelo de árvore obtido com o método model_boosting")



