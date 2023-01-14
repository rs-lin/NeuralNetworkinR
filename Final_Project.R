# Final Project
library(tidyverse)
library(dplyr)
library(pROC)
data_2021<-read.csv("data_2021.csv")
spring_indicators<- c(58,72,77,121,201,205,210,256,412,428,494,500,501)
data <- filter(data_2021,phenophase_id ==spring_indicators)
data1 <- subset(data, select = c(4,5,6,17,18))
data1 = data1[!(data1$phenophase==-1),]
# produce weights
casl_nn_make_weights <-function(sizes)
  {
    L <- length(sizes) - 1L
    weights <- vector("list", L)
    for (j in seq_len(L))
    {
      w <- matrix(rnorm(sizes[j] * sizes[j + 1L]),
                  ncol = sizes[j],
                  nrow = sizes[j + 1L])
      weights[[j]] <- list(w=w,
                           b=rnorm(sizes[j + 1L]))
    }
    weights
}

casl_nn_make_weights_mu <-function(sizes)
  {
    L <- length(sizes) - 1L
    weights <- vector("list", L)
    for (j in seq_len(L))
    {
      w <- matrix(rnorm(sizes[j] * sizes[j + 1L],
                        sd = 1/sqrt(sizes[j])),
                  ncol = sizes[j],
                  nrow = sizes[j + 1L])
      v <- matrix(0,
                  ncol = sizes[j],
                  nrow = sizes[j + 1L])
      weights[[j]] <- list(w=w,
                           v=v,
                           b=rnorm(sizes[j + 1L]))
    }
    weights
}

# RELU
casl_util_ReLU <-function(v){
    v[v < 0] <- 0
    v
}

# Derivative of RELU
casl_util_ReLU_p <-function(v)
  {
    p <- v * 0
    p[v > 0] <- 1
    p
}

#softmax for classification
casl_util_softmax <-function(z)
  {
    exp(z) / sum(exp(z))
}

casl_util_sigmoid<- function(z)
{
  exp(z)/(1+exp(z))
}
# derivative of MSE
casl_util_mse_p <-function(y, a)
  {
    (a - y)
}
#mse
casl_util_mse <-function(y, a)
{
  (a - y)^2
}
# cross entropy
casl_util_cross_entropy<-function(y,a)
{
  -sum(y*log(a))
}
# cross entropy
casl_util_cross_entropy<-function(y,a)
{
  -(sum(y[,1]*log(a[,1]))+sum(y[,2]*log(a[,2])))
}

casl_nn_forward_prop <-function(x, weights, sigma)
  {
    L <- length(weights)
    z <- vector("list", L)
    a <- vector("list", L)
    for (j in seq_len(L))
    {
      a_j1 <- if(j == 1) x else a[[j - 1L]]
      z[[j]] <- weights[[j]]$w %*% a_j1 + weights[[j]]$b
      a[[j]] <- if (j != L) sigma(z[[j]]) else z[[j]]
    }
    list(z=z, a=a)
}

casl_nn_backward_prop <- function(x, y, weights, f_obj, sigma_p, f_p)
{
  z <- f_obj$z; a <- f_obj$a
  L <- length(weights)
  grad_z <- vector("list", L)
  grad_w <- vector("list", L)
  for (j in rev(seq_len(L)))
  {
    if (j == L)
    {
      grad_z[[j]] <- f_p(y, a[[j]])
    } else {
      grad_z[[j]] <- (t(weights[[j + 1]]$w) %*%
                        grad_z[[j + 1]]) * sigma_p(z[[j]])
    }
    a_j1 <- if(j == 1) x else a[[j - 1L]]
    grad_w[[j]] <- grad_z[[j]] %*% t(a_j1)
  }
  list(grad_z=grad_z, grad_w=grad_w)
}

casl_nn_sgd <- function(X, y, sizes, epochs, eta, weights=NULL)
  {
    if (is.null(weights))
    {
      weights <- casl_nn_make_weights(sizes)
    }
    for (epoch in seq_len(epochs))
    {
      for (i in seq_len(nrow(X)))
      {
        f_obj <- casl_nn_forward_prop(X[i,], weights,
                                      casl_util_ReLU)
        b_obj <- casl_nn_backward_prop(X[i,], y[i,], weights,
                                       f_obj, casl_util_ReLU_p,
                                       casl_util_mse_p)
        for (j in seq_along(b_obj))
        {
          weights[[j]]$b <- weights[[j]]$b -
            eta * b_obj$grad_z[[j]]
          weights[[j]]$w <- weights[[j]]$w -
            eta * b_obj$grad_w[[j]]
        }
      }
    }
    weights
  }

casl_nnmulti_forward_prop <-function(x, weights, sigma)
  {
    L <- length(weights)
    z <- vector("list", L)
    a <- vector("list", L)
    for (j in seq_len(L))
    {
      a_j1 <- if(j == 1) x else a[[j - 1L]]
      z[[j]] <- weights[[j]]$w %*% a_j1 + weights[[j]]$b
      if (j != L) {
        a[[j]] <- sigma(z[[j]])
      } else {
        a[[j]] <- casl_util_softmax(z[[j]])
      }
    }
    list(z=z, a=a)
  }

casl_nnmulti_backward_prop <-function(x, y, weights, f_obj, sigma_p)
  {
    z <- f_obj$z; a <- f_obj$a
    L <- length(weights)
    grad_z <- vector("list", L)
    grad_w <- vector("list", L)
    for (j in rev(seq_len(L)))
    {
      if (j == L)
      {
        grad_z[[j]] <- a[[j]] - y
      } else {
        grad_z[[j]] <- (t(weights[[j + 1L]]$w) %*%
                          grad_z[[j + 1L]]) * sigma_p(z[[j]])
      }
      a_j1 <- if(j == 1) x else a[[j - 1L]]
      grad_w[[j]] <- grad_z[[j]] %*% t(a_j1)
    }
    list(grad_z=grad_z, grad_w=grad_w)
}


casl_nnmulti_sgd <-function(X, y, sizes, epochs, eta, mu, l2, weights=NULL) {
    if (is.null(weights))
    {
      weights <- casl_nn_make_weights_mu(sizes)
    }
    for (epoch in seq_len(epochs))
    {
      for (i in seq_len(nrow(X)))
      {
        f_obj <- casl_nnmulti_forward_prop(X[i, ], weights,
                                           casl_util_ReLU)
        b_obj <- casl_nnmulti_backward_prop(X[i, ], y[i, ],
                                            weights, f_obj,
                                            casl_util_ReLU_p)
        for (j in seq_along(b_obj))
        {
          weights[[j]]$b <- weights[[j]]$b -
            eta * b_obj$grad_z[[j]]
          weights[[j]]$v <- mu * weights[[j]]$v -
            eta * b_obj$grad_w[[j]]
          weights[[j]]$w <- (1 - eta * l2) *
            weights[[j]]$w +
            weights[[j]]$v
        }
      }
    }
  weights
}
casl_nnmulti_sag_sgd <-function(X, y, sizes, epochs, eta, mu, l2, weights=NULL) {
  if (is.null(weights))
  {
    weights <- casl_nn_make_weights_mu(sizes)
  }
  L <- length(weights)
  gradW <-vector("list", nrow(X))
  gradZ <- vector("list", nrow(X))
  for (i in seq_len(nrow(X)))
  {
    f_obj <- casl_nnmulti_forward_prop(X[i, ], weights,
                                       casl_util_ReLU)
    b_obj <- casl_nnmulti_backward_prop(X[i, ], y[i, ],
                                        weights, f_obj,
                                        casl_util_ReLU_p)
    gradW[[i]]<-b_obj$grad_w
    gradZ[[i]]<-b_obj$grad_z
    for (j in seq_along(b_obj))
    {
      weights[[j]]$b <- weights[[j]]$b -
        eta * b_obj$grad_z[[j]]
      weights[[j]]$v <- mu * weights[[j]]$v -
        eta * b_obj$grad_w[[j]]
    }
  }
  w_k_bar_all<-vector("list",L)
  for (l in seq_len(L)){
    w_k_bar <- gradW[[1]][[l]]
    for (i in 2:nrow(X)){
      
      w_k_bar = w_k_bar+gradW[[i]][[l]]
      
    }
    w_k_bar_all[[l]]<-w_k_bar
  }
  perm_list<-sample(seq_len(nrow(X)),epochs,replace = FALSE)
  for (epoch in seq_len(epochs))
  {
    k = perm_list[epoch]
    f_obj <- casl_nnmulti_forward_prop(X[k, ], weights,
                                       casl_util_ReLU)
    b_obj <- casl_nnmulti_backward_prop(X[k, ], y[k, ],
                                        weights, f_obj,
                                        casl_util_ReLU_p)
    for (j in seq_along(b_obj)){
      grad_fk_we<-b_obj$grad_w[[j]]
      g_e <- (grad_fk_we - gradW[[k]][[j]]+w_k_bar_all[[j]])/nrow(X)
      gradW[[k]][[j]]<-grad_fk_we
      weights[[j]]$w<-weights[[j]]$w-eta*g_e
    }
  }
  weights
}

casl_nnmulti_sag_sgd <-function(X, y, sizes, epochs, eta, mu=0, l2=0, weights=NULL) {
  if (is.null(weights))
  {
    weights <- casl_nn_make_weights_mu(sizes)
  }
  for (epoch in seq_len(epochs))
  {
    k= sample(seq_len(nrow(X)),1)
    for (i in seq_len(nrow(X))){
      
      if (i==k){
        f_obj <- casl_nnmulti_forward_prop(X[i, ], weights,
                                           casl_util_ReLU)
        b_obj <- casl_nnmulti_backward_prop(X[i, ], y[i, ],
                                            weights, f_obj,
                                            casl_util_ReLU_p)
      }
      
      else{
        f_obj <- casl_nnmulti_forward_prop(X[1, ], weights,
                                           casl_util_ReLU)
        b_obj <- casl_nnmulti_backward_prop(X[1, ], y[1, ],
                                            weights, f_obj,
                                            casl_util_ReLU_p)
      }
      
      for (j in seq_along(b_obj))
      {
        
        weights[[j]]$b <- weights[[j]]$b -
          eta/nrow(X) * b_obj$grad_z[[j]]
        weights[[j]]$v <- mu * weights[[j]]$v -
          eta/nrow(X) * b_obj$grad_w[[j]]
        weights[[j]]$w <- (1 - eta/nrow(X) * l2) *
          weights[[j]]$w +
          weights[[j]]$v
      }
      
    }
    
  }
  weights
}

casl_nnmulti_saga_sgd <-function(X, y, sizes, epochs, eta, mu=0, l2=0, weights=NULL) {
  if (is.null(weights))
  {
    weights <- casl_nn_make_weights_mu(sizes)
  }
  L <- length(weights)
  z_vals <- vector("list", nrow(X))
  a_vals <- vector("list", nrow(X))
  gradW <-vector("list", nrow(X))
  gradZ <- vector("list", nrow(X))
  for (i in seq_len(nrow(X))){
    f_obj <- casl_nnmulti_forward_prop(X[i, ], weights,
                                       casl_util_ReLU)
    b_obj <- casl_nnmulti_backward_prop(X[i, ], y[i, ],
                                        weights, f_obj,
                                        casl_util_ReLU_p)
    z_vals[[i]]<-f_obj$z
    a_vals[[i]]<-f_obj$a
    gradW[[i]]<-b_obj$grad_w
    gradZ[[i]]<-b_obj$grad_z
  }
  perm_list<-sample(seq_len(nrow(X)),epochs,replace = FALSE)
  for (epoch in seq_len(epochs))
  {
      k = perm_list[epoch]
      f_obj <- casl_nnmulti_forward_prop(X[k, ], weights,
                                         casl_util_ReLU)
      b_obj <- casl_nnmulti_backward_prop(X[k, ], y[k, ],
                                          weights, f_obj,
                                          casl_util_ReLU_p)
      z_vals[[k]]<-f_obj$z
      a_vals[[k]]<-f_obj$a
      gradW[[k]]<-b_obj$grad_w
      gradZ[[k]]<-b_obj$grad_z
    
      # initialization
      w_k_bar_all<-vector("list",L)
      z_k_bar_all<-vector("list",L)
      for (l in seq_len(L)){
        w_k_bar <- gradW[[1]][[l]]
        z_k_bar<-gradZ[[1]][[l]]
        for (i in 2:nrow(X)){
          
          w_k_bar = w_k_bar+gradW[[i]][[l]]
          z_k_bar = z_k_bar+gradZ[[i]][[l]]
          
        }
        w_k_bar_all[[l]]<-w_k_bar/nrow(X)
        z_k_bar_all[[l]]<- z_k_bar/nrow(X)
      }
      for (i in seq_len(nrow(X))){
        for (j in seq_along(b_obj))
        {
          weights[[j]]$b <- weights[[j]]$b -
            eta * (gradZ[[i]][[j]]-b_obj$grad_z[[j]]+z_k_bar_all[[j]] )
          weights[[j]]$v <- mu * weights[[j]]$v -
            eta * (gradW[[i]][[j]]-b_obj$grad_w[[j]]+w_k_bar_all[[j]])
          weights[[j]]$w <- (1 - eta * l2) *
            weights[[j]]$w +
            weights[[j]]$v
        }
      }
      
  }
  weights
}
casl_nnmulti_saga_sgd <-function(X, y, sizes, epochs, eta, mu=0, l2=0, weights=NULL) {
  if (is.null(weights))
  {
    weights <- casl_nn_make_weights_mu(sizes)
  }
  L <- length(weights)
  gradW <-vector("list", nrow(X))
  gradZ <- vector("list", nrow(X))
  for (i in seq_len(nrow(X)))
  {
    f_obj <- casl_nnmulti_forward_prop(X[i, ], weights,
                                       casl_util_ReLU)
    b_obj <- casl_nnmulti_backward_prop(X[i, ], y[i, ],
                                        weights, f_obj,
                                        casl_util_ReLU_p)
    gradW[[i]]<-b_obj$grad_w
    gradZ[[i]]<-b_obj$grad_z
    for (j in seq_along(b_obj))
    {
      weights[[j]]$b <- weights[[j]]$b -
        eta * b_obj$grad_z[[j]]
      weights[[j]]$v <- mu * weights[[j]]$v -
        eta * b_obj$grad_w[[j]]
    }
  }
  w_k_bar_all<-vector("list",L)
  for (l in seq_len(L)){
    w_k_bar <- gradW[[1]][[l]]
    for (i in 2:nrow(X)){
      
      w_k_bar = w_k_bar+gradW[[i]][[l]]
      
    }
    w_k_bar_all[[l]]<-w_k_bar/nrow(X)
  }
  perm_list<-sample(seq_len(nrow(X)),epochs,replace = FALSE)
  for (epoch in seq_len(epochs))
  {
    k = perm_list[epoch]
    f_obj <- casl_nnmulti_forward_prop(X[k, ], weights,
                                       casl_util_ReLU)
    b_obj <- casl_nnmulti_backward_prop(X[k, ], y[k, ],
                                        weights, f_obj,
                                        casl_util_ReLU_p)
    for (j in seq_along(b_obj)){
      grad_fk_we<-b_obj$grad_w[[j]]
      g_e <- grad_fk_we - gradW[[k]][[j]]+w_k_bar_all[[j]]
      gradW[[k]][[j]]<-grad_fk_we
      weights[[j]]$w<-weights[[j]]$w-eta*g_e
    }
  }
  weights
}

casl_nn_predict <-function(weights, X_test)
  {p <- length(weights[[length(weights)]]$b)
  y_hat <- matrix(0, ncol = p, nrow = nrow(X_test))
  for (i in seq_len(nrow(X_test)))
  {
    a <- casl_nn_forward_prop(X_test[i,], weights,
                              casl_util_ReLU)$a
    y_hat[i, ] <- a[[length(a)]]
  }
  y_hat
}

casl_nnmulti_predict <-function(weights, X_test)
  {
    p <- length(weights[[length(weights)]]$b)
    y_hat <- matrix(0, ncol=p, nrow=nrow(X_test))
    for (i in seq_len(nrow(X_test)))
    {
      a <- casl_nnmulti_forward_prop(X_test[i, ], weights,
                                     casl_util_ReLU)$a
      y_hat[i,] <- a[[length(a)]]
    }
    y_hat
}

# test
X <- matrix(runif(2000, min=-1, max=1), ncol=5)
y <- X[,1,drop = FALSE]^2 + rnorm(400, sd = 0.1)
weights <- casl_nn_sgd(X, y, sizes=c(5, 20, 1),epochs=25, eta=0.01)
y_pred <- casl_nn_predict(weights, X)

# test multinn
X <- matrix(runif(1000, min=-1, max=1), ncol=1)
y1 <- X[,1 , drop=FALSE]^2 + rnorm(1000, sd=0.1)
# y1 <- rowSums(X^2)/nrow(X^2)+rnorm(1000, sd=0.1)
y_true <- cbind(as.numeric(y1 > 0.5))
y <- cbind(as.numeric(y1 > 0.5), as.numeric(y1 <= 0.5))

# case0: pure sgd
case0_auc<-c()
case0_loss<-c()
for (e in 1:25){
  weights0 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                               epochs=e, eta=0.01,mu=0,l2=0)
  y_pred0 <- casl_nnmulti_predict(weights0, X)
  y_predd0<-data.matrix(as.numeric(y_pred0[,1]>0.5))
  nn0<- data.frame(result=y_true,predict=y_pred0[,1])
  nn0_roc <- roc(response = nn0$result,
                 predictor = nn0$predict,
                 levels = c(0,1))
  case0_auc[e]<- round(auc(nn0_roc), digits = 2)
  case0_loss[e]<- mean(casl_util_cross_entropy(y,y_pred0))
}
 plot(1:25,case0_loss)


# weights0 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
#                              epochs=25L, eta=0.01)
# y_pred0 <- casl_nnmulti_predict(weights0, X)
# y_predd0<-data.matrix(as.numeric(y_pred0[,1]>0.5))

# case1: sgd with momentum mu=0.09
case1_auc<-c()
case1_loss<-c()
for (e in 1:25){
  weights1 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                               epochs=e, eta=0.01,mu=0.09,l2=0)
  y_pred1 <- casl_nnmulti_predict(weights1, X)
  y_predd1<-data.matrix(as.numeric(y_pred1[,1]>0.5))
  nn1<- data.frame(result=y_true,predict=y_pred1[,1])
  nn1_roc <- roc(response = nn1$result,
                 predictor = nn1$predict,
                 levels = c(0,1))
  case1_auc[e]<- round(auc(nn1_roc), digits = 2)
  case1_loss[e]<- mean(casl_util_cross_entropy(y,y_pred1))
}


# weights1 <- casl_nnmulti_sgd(X, y, sizes=c(5, 25, 2),
#                              epochs=1, eta=0.01)
# y_pred1 <- casl_nnmulti_predict(weights1, X)
# y_predd<-data.matrix(as.numeric(y_pred1[,1]>0.5))

# case2: sgd with l2=0.001
case2_auc<-c()
case2_loss<-c()
for (e in 1:25){
  weights2 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                               epochs=e, eta=0.01,mu=0,l2=0.01)
  y_pred2 <- casl_nnmulti_predict(weights2, X)
  y_predd2<-data.matrix(as.numeric(y_pred2[,1]>0.5))
  nn2<- data.frame(result=y_true,predict=y_pred2[,1])
  nn2_roc <- roc(response = nn2$result,
                 predictor = nn2$predict,
                 levels = c(0,1))
  case2_auc[e]<- round(auc(nn2_roc), digits = 2)
  case2_loss[e]<- mean(casl_util_cross_entropy(y,y_pred2))
}
plot(1:25,case0_loss)
lines(1:25,case2_loss)
plot(1:25,case2_loss)



# weights2 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
#                              epochs=5L, eta=0.01)
# y_pred2 <- casl_nnmulti_predict(weights2, X)
# y_predd2<-data.matrix(as.numeric(y_pred2[,1]>0.5))

# case3: sgd with mu=0.09, l2=0.001
case3_auc<-c()
case3_loss<-c()
for (e in 1:25){
  weights3 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                               epochs=e, eta=0.01,mu=0.09,l2=0.001)
  y_pred3 <- casl_nnmulti_predict(weights3, X)
  y_predd3<-data.matrix(as.numeric(y_pred3[,1]>0.5))
  nn3<- data.frame(result=y_true,predict=y_pred3[,1])
  nn3_roc <- roc(response = nn3$result,
                 predictor = nn3$predict,
                 levels = c(0,1))
  case3_auc[e]<- round(auc(nn3_roc), digits = 2)
  case3_loss[e]<- mean(casl_util_mse(nn3$predict,nn3$result))
}
plot(1:25,case0_loss)
lines(1:25,case1_loss)
lines(1:25,case3_loss)
plot(1:25,case3_loss)

plot(1:25,case1_loss,type="l",col="steelblue3",lwd=5.0,xlab="epochs",ylab="loss")
lines(1:25,case1_loss,col="sienna1",lwd=3.0,lty=2)
lines(1:25,case2_loss,col="lightpink2",lwd=5.0)
legend("topright",legend=c("SGD","SGD with momentum","SGD with regularization"),col=c("steelblue3","sienna1","lightpink2"),lty=c(1,2,1),lwd=3,cex=1.5)

plot(1:25,case0_loss,type="l",col="steelblue3",lwd=5.0,xlab="epochs",ylab="loss")
lines(1:25,case2_loss,col="lightpink2",lwd=5.0)
legend("topright",legend=c("SGD","SGD with regularization"),col=c("steelblue3","lightpink2"),lty=c(1,1),lwd=3,cex=1.5)

plot(1:25,case0_auc,type="l",col="steelblue3",lwd=3.5,xlab="epochs",ylab="auc")
lines(1:25,case1_auc,col="sienna1",lwd=3.0)
lines(1:25,case2_auc,col="lightpink2",lwd=3.0)
legend("bottomright",legend=c("SGD","SGD with momentum","SGD with regularization"),col=c("steelblue3","sienna1","lightpink2"),lty=c(1,1,1),lwd=3,cex=1.5)

lines(1:25,case3_loss,col="limegreen",lwd=5.0)
# case 4 sag
case4_auc<-c()
case4_loss<-c()
for (e in 1:25){
  weights4 <- casl_nnmulti_sag_sgd(X, y, sizes=c(1,50,50,50,2),
                               epochs=e, eta=0.005,mu=0.09)
  y_pred4 <- casl_nnmulti_predict(weights4, X)
  y_predd4<-data.matrix(as.numeric(y_pred4[,1]>0.5))
  nn4<- data.frame(result=y_true,predict=y_pred4[,1])
  nn4_roc <- roc(response = nn4$result,
                 predictor = nn4$predict,
                 levels = c(0,1))
  case4_auc[e]<- round(auc(nn4_roc), digits = 2)
  case4_loss[e]<- mean(casl_util_cross_entropy(y,y_pred4))
}
# loss
plot(1:25,case4_loss,type="l",col="orange",lwd=5.0,xlab="epochs",ylab="loss")
lines(1:25,case5_loss,col="green",lwd=5.0)
legend("topright",c("SAG","SAGA"),col=c("orange","green"),lty=c(1,1),lwd=3,cex=1.5)

# auc
plot(1:25,case4_auc,type="l",col="orange",lwd=3.5,xlab="epochs",ylab="auc")
lines(1:25,case5_auc,col="green",lwd=3.0)
legend("bottomright",legend=c("SAG","SAGA"),col=c("orange","green"),lty=c(1,1),lwd=3,cex=1.5)


# case 5 saga
case5_auc<-c()
case5_loss<-c()
for (e in 1:50){
  weights5 <- casl_nnmulti_saga_sgd(X, y, sizes=c(1,50,50,50,50,50,2),
                                   epochs=e, eta=0.005,mu=0.09)
  y_pred5 <- casl_nnmulti_predict(weights5, X)
  y_predd5<-data.matrix(as.numeric(y_pred5[,1]>0.5))
  nn5<- data.frame(result=y_true,predict=y_pred5[,1])
  nn5_roc <- roc(response = nn5$result,
                 predictor = nn5$predict,
                 levels = c(0,1))
  case5_auc[e]<- round(auc(nn5_roc), digits = 2)
  case5_loss[e]<- mean(casl_util_cross_entropy(y,y_pred5))
}
plot(1:25,case0_loss)
lines(1:25,case4_loss)
plot(1:50,case5_loss)
plot(1:25,case4_loss)

# momentum
weights0m <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                             epochs=25L, eta=0.01)
y_pred0m <- casl_nnmulti_predict(weights0m, X)
y_predd0m <-data.matrix(as.numeric(y_pred0m[,1]>0.5))

weights1m <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                             epochs=50L, eta=0.01)
y_pred1m <- casl_nnmulti_predict(weights1m, X)
y_preddm <-data.matrix(as.numeric(y_pred1m[,1]>0.5))

weights2m <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                             epochs=5L, eta=0.01)
y_pred2m <- casl_nnmulti_predict(weights2m, X)
y_predd2m <-data.matrix(as.numeric(y_pred2m[,1]>0.5))

# simple nn
weights3m <- casl_nn_sgd(X, y, sizes=c(1, 25, 2),epochs=25, eta=0.01)
y_pred3m <- casl_nn_predict(weights3m, X)
y_predd3m<-data.matrix(as.numeric(y_pred3m[,1]>0.5))

# sag
weights4<-casl_nnmulti_sag_sgd(X,y,c(1,25,25,25,2),epochs=5,eta=0.01)
y_pred4 <- casl_nn_predict(weights4, X)
y_predd4<-data.matrix(as.numeric(y_pred4[,1]>0.5))

# saga
weights5<-casl_nnmulti_saga_sgd(X,y,c(1,25,25,25,2),epochs=5,eta=0.01)
y_pred5 <- casl_nn_predict(weights5, X)
y_predd5<-data.matrix(as.numeric(y_pred5[,1]>0.5))

plot(X,y_pred0[,1])
table(y_predd,y_true)
table(y_predd0,y_true)
table(y_predd2,y_true)

table(y_preddm,y_true)
table(y_predd0m,y_true)
table(y_predd2m,y_true)


table(y_predd3,y_true)

# ROC curve
nn0<- data.frame(result=y_true,predict=y_pred0[,1])
nn1<- data.frame(result=y_true,predict=y_pred1[,1])
nn2<- data.frame(result=y_true,predict=y_pred2[,1])
nnm0<- data.frame(result=y_true,predict=y_pred0m[,1])
nnm1<- data.frame(result=y_true,predict=y_pred1m[,1])
nnm2<- data.frame(result=y_true,predict=y_pred2m[,1])

nn0_roc <- roc(response = nn0$result,
                predictor = nn0$predict,
                levels = c(0,1))
ggroc(nn1_roc, legacy.axes = TRUE) +
  labs(x = 'False-positive rate', y = 'True-positive rate', title = 'Simulated ROC curve') +
  annotate('text', x = .5, y = .5, label = paste0('AUC: ',round(auc(nn0_roc), digits = 2)))

nn1_roc <- roc(response = nn1$result,
               predictor = nn1$predict,
               levels = c(0,1))
ggroc(nn1_roc, legacy.axes = TRUE) +
  labs(x = 'False-positive rate', y = 'True-positive rate', title = 'Simulated ROC curve') +
  annotate('text', x = .5, y = .5, label = paste0('AUC: ',round(auc(nn1_roc), digits = 2)))

nn2_roc <- roc(response = nn2$result,
               predictor = nn2$predict,
               levels = c(0,1))
ggroc(nn2_roc, legacy.axes = TRUE) +
  labs(x = 'False-positive rate', y = 'True-positive rate', title = 'Simulated ROC curve') +
  annotate('text', x = .5, y = .5, label = paste0('AUC: ',round(auc(nn2_roc), digits = 2)))

nnm0_roc <- roc(response = nnm0$result,
               predictor = nnm0$predict,
               levels = c(0,1))
ggroc(nnm1_roc, legacy.axes = TRUE) +
  labs(x = 'False-positive rate', y = 'True-positive rate', title = 'Simulated ROC curve') +
  annotate('text', x = .5, y = .5, label = paste0('AUC: ',round(auc(nnm0_roc), digits = 2)))

nnm1_roc <- roc(response = nnm1$result,
               predictor = nnm1$predict,
               levels = c(0,1))
ggroc(nnm1_roc, legacy.axes = TRUE) +
  labs(x = 'False-positive rate', y = 'True-positive rate', title = 'Simulated ROC curve') +
  annotate('text', x = .5, y = .5, label = paste0('AUC: ',round(auc(nnm1_roc), digits = 2)))

nnm2_roc <- roc(response = nnm2$result,
               predictor = nnm2$predict,
               levels = c(0,1))
ggroc(nnm2_roc, legacy.axes = TRUE) +
  labs(x = 'False-positive rate', y = 'True-positive rate', title = 'Simulated ROC curve') +
  annotate('text', x = .5, y = .5, label = paste0('AUC: ',round(auc(nnm2_roc), digits = 2)))


data_2021<-read.csv("data_2021.csv")
spring_indicators<- c(58,72,77,121,201,205,210,256,412,428,494,500,501)
data <- filter(data_2021,phenophase_id ==spring_indicators)
data1 <- subset(data, select = c(4,5,6,17,18))
data1 = data1[!(data1$phenophase==-1),]






# test own data
training_idx1 <- sample(which(data1$phenophase_status==1),250,replace=FALSE)
training_idx2 <- sample(which(data1$phenophase_status==0),250,replace=FALSE)
training_idx<-append(training_idx1,training_idx2)
# validation_idx <- setdiff(1:nrow(data1),training_idx)
# td<-cbind(scale(data1[,1:4]),data1$phenophase_status)
training_data<-data1[training_idx,]
validation_data<-data1[validation_idx,]
#training_data<-td[training_idx,]
#validation_data<-td[validation_idx,]
X<-data.matrix(training_data[,4])
y<-data.matrix(training_data[,5])
y2<-cbind(as.numeric(y > 0.5), as.numeric(y <= 0.5))

weights1<-casl_nnmulti_sgd(X,y2,sizes=c(1,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.09,l2=0.01)



#weights<-casl_nnmulti_sgd(X,y2,sizes=c(4,10,25,50,75,100,75,50,25,20,10,5,3,2),epochs=50,eta=0.01,0.09,0.01)
y_pred1 <- casl_nnmulti_predict(weights1, X)
y_predd1<-data.matrix(as.numeric(y_pred[,1]>0.3))
nn1<- data.frame(result=y,predict=y_predd1[,1])
#y_pred2 <- casl_nnmulti_predict(weights, data.matrix(training_data[,1:4]))
nn1_roc <- roc(response = nn1$result,
  predictor = nn1$predict,
  levels = c(0,1))
auc(nn1_roc)
mean(casl_util_cross_entropy(y2,y_pred1))

case1_loss<-c()
for (i in 1:20){
  weights1<-casl_nnmulti_sgd(X,y2,sizes=c(1,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.09,l2=0.01)
  y_pred1 <- casl_nnmulti_predict(weights1, X)
  case1_loss[i]<-casl_util_cross_entropy(y2,y_pred1)
}
median(case1_loss)

weights2<-casl_nnmulti_saga_sgd(X,y2,sizes=c(1,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0,l2=0)
y_pred2 <- casl_nnmulti_predict(weights2, X)
y_predd2<-data.matrix(as.numeric(y_pred2[,1]>0.04))
nn2<- data.frame(result=y,predict=y_predd2[,1])
#y_pred2 <- casl_nnmulti_predict(weights, data.matrix(training_data[,1:4]))
nn2_roc <- roc(response = nn2$result,
               predictor = nn2$predict,
               levels = c(0,1))
auc(nn2_roc)
mean(casl_util_cross_entropy(y2,y_pred2))

case2_loss<-c()
for (i in 1:20){
  weights2<-casl_nnmulti_saga_sgd(X,y2,sizes=c(1,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.09,l2=0.01)
  y_pred2 <- casl_nnmulti_predict(weights2, X)
  case2_loss[i]<-casl_util_cross_entropy(y2,y_pred2)
}
median(case2_loss)


weights3<-casl_nnmulti_sag_sgd(X,y2,sizes=c(1,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.07,l2=0.01)
y_pred3 <- casl_nnmulti_predict(weights3, X)
y_predd3<-data.matrix(as.numeric(y_pred3[,1]>0.022))
nn3<- data.frame(result=y,predict=y_predd3[,1])
#y_pred2 <- casl_nnmulti_predict(weights, data.matrix(training_data[,1:4]))
nn3_roc <- roc(response = nn3$result,
               predictor = nn3$predict,
               levels = c(0,1))
auc(nn3_roc)
mean(casl_util_cross_entropy(y2,y_pred3))

case3_loss<-c()
for (i in 1:20){
  weights3<-casl_nnmulti_sag_sgd(X,y2,sizes=c(1,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.09,l2=0.01)
  y_pred3 <- casl_nnmulti_predict(weights3, X)
  case3_loss[i]<-casl_util_cross_entropy(y2,y_pred3)
}
median(case3_loss)

# more features
X = data.matrix(training_data[,1:4])
startTime<-Sys.time()
weights1<-casl_nnmulti_sgd(X,y2,sizes=c(4,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0,l2=0)
endTime<-Sys.time()
y_pred1 <- casl_nnmulti_predict(weights1, X)
y_predd1 <-data.matrix(as.numeric(y_pred1[,1]>0.6189))
nn1<- data.frame(result=y,predict=y_predd1[,1])
nn1_roc <- roc(response = nn1$result,
               predictor = nn1$predict,
               levels = c(0,1))
auc(nn1_roc)
mean(casl_util_cross_entropy(y2,y_pred1))

case1_loss<-c()
for (i in 1:20){
  weights1<-casl_nnmulti_sgd(X,y2,sizes=c(4,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.09,l2=0.01)
  y_pred1 <- casl_nnmulti_predict(weights1, X)
  case1_loss[i]<-casl_util_cross_entropy(y2,y_pred1)
}
median(case1_loss)

startTime<-Sys.time()
weights2<-casl_nnmulti_saga_sgd(X,y2,sizes=c(4,25,25,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0,l2=0)
endTime<-Sys.time()
y_pred2 <- casl_nnmulti_predict(weights2, X)
y_predd2<-data.matrix(as.numeric(y_pred2[,1]>0.178))
nn2<- data.frame(result=y,predict=y_predd2[,1])
nn2_roc <- roc(response = nn2$result,
               predictor = nn2$predict,
               levels = c(0,1))
auc(nn2_roc)
mean(casl_util_cross_entropy(y2,y_pred2))

case2_loss<-c()
for (i in 1:20){
  weights2<-casl_nnmulti_saga_sgd(X,y2,sizes=c(4,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.09,l2=0.01)
  y_pred2 <- casl_nnmulti_predict(weights2, X)
  case2_loss[i]<-casl_util_cross_entropy(y2,y_pred2)
}
median(case2_loss)

weights3<-casl_nnmulti_sag_sgd(X,y2,sizes=c(4,25,25,25,25,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0,l2=0)
y_pred3 <- casl_nnmulti_predict(weights3, X)
y_predd3<-data.matrix(as.numeric(y_pred3[,2]>0.253))
nn3<- data.frame(result=y,predict=y_predd3[,1])
nn3_roc <- roc(response = nn3$result,
               predictor = nn3$predict,
               levels = c(0,1))
auc(nn3_roc)
mean(casl_util_cross_entropy(y2,y_pred3))


case3_loss<-c()
for (i in 1:20){
  weights3<-casl_nnmulti_sag_sgd(X,y2,sizes=c(4,25,25,25,25,25,25,25,25,25,25,2),epochs=50,eta=0.01,mu=0.09,l2=0.01)
  y_pred3 <- casl_nnmulti_predict(weights3, X)
  case3_loss[i]<-casl_util_cross_entropy(y2,y_pred3)
}
median(case3_loss)



sort()# case0: pure sgd
case0_auc<-c()
case0_loss<-c()
for (e in 1:25){
  weights0 <- casl_nnmulti_sgd(X, y2, sizes=c(1,25,25,25,25,25,25,25,25,25,25,25,25,2),
                               epochs=e, eta=0.01,mu=0.09,l2=0.01)
  y_pred0 <- casl_nnmulti_predict(weights0, X)
  y_predd0<-data.matrix(as.numeric(y_pred0[,1]>0.1))
  nn0<- data.frame(result=y_true,predict=y_pred0[,1])
  nn0_roc <- roc(response = nn0$result,
                 predictor = nn0$predict,
                 levels = c(0,1))
  case0_auc[e]<- round(auc(nn0_roc), digits = 2)
  case0_loss[e]<- mean(casl_util_mse(nn0$predict,nn0$result))
}
plot(1:25,case0_loss)
barplot(c(auc(nn1_roc),auc(nn2_roc),auc(nn3_roc)))

# weights0 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
#                              epochs=25L, eta=0.01)
# y_pred0 <- casl_nnmulti_predict(weights0, X)
# y_predd0<-data.matrix(as.numeric(y_pred0[,1]>0.5))

# case1: sgd with momentum mu=0.09
case1_auc<-c()
epoch = seq(50,100,10)
case1_loss<-c()
for (e in 1:25){
  weights1 <- casl_nnmulti_sag_sgd(X, y, sizes=c(1,25,25,25,2),
                               epochs=e, eta=0.01,mu=0,l2=0)
  y_pred1 <- casl_nnmulti_predict(weights1, X)
  y_predd1<-data.matrix(as.numeric(y_pred1[,1]>0.5))
  nn1<- data.frame(result=y_true,predict=y_pred1[,1])
  #nn1_roc <- roc(response = nn1$result,
                 #predictor = nn1$predict,
                 #levels = c(0,1))
  #case1_auc[e]<- round(auc(nn1_roc), digits = 2)
  case1_loss[e]<- mean(casl_util_mse(nn1$predict,nn1$result))
}


# weights1 <- casl_nnmulti_sgd(X, y, sizes=c(5, 25, 2),
#                              epochs=1, eta=0.01)
# y_pred1 <- casl_nnmulti_predict(weights1, X)
# y_predd<-data.matrix(as.numeric(y_pred1[,1]>0.5))

# case2: sgd with l2=0.001
case2_auc<-c()
case2_loss<-c()
for (e in 1:25){
  weights2 <- casl_nnmulti_sgd(X, y, sizes=c(1,25,25,25,25,25,25,25,25,25,25,25,25, 2),
                               epochs=e, eta=0.01,mu=0,l2=0.01)
  y_pred2 <- casl_nnmulti_predict(weights2, X)
  y_predd2<-data.matrix(as.numeric(y_pred2[,1]>0.5))
  nn2<- data.frame(result=y_true,predict=y_pred2[,1])
  #nn2_roc <- roc(response = nn2$result,
                 #predictor = nn2$predict,
                 #levels = c(0,1))
  #case2_auc[e]<- round(auc(nn2_roc), digits = 2)
  case2_loss[e]<- mean(casl_util_mse(nn2$predict,nn2$result))
}
plot(1:25,case0_loss)
lines(1:25,case2_loss)
plot(1:25,case2_loss)



# weights2 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
#                              epochs=5L, eta=0.01)
# y_pred2 <- casl_nnmulti_predict(weights2, X)
# y_predd2<-data.matrix(as.numeric(y_pred2[,1]>0.5))

# case3: sgd with mu=0.09, l2=0.001
case3_auc<-c()
case3_loss<-c()
for (e in 1:25){
  weights3 <- casl_nnmulti_sgd(X, y, sizes=c(1, 25, 2),
                               epochs=e, eta=0.01,mu=0.09,l2=0.001)
  y_pred3 <- casl_nnmulti_predict(weights3, X)
  y_predd3<-data.matrix(as.numeric(y_pred3[,1]>0.5))
  nn3<- data.frame(result=y_true,predict=y_pred3[,1])
  nn3_roc <- roc(response = nn3$result,
                 predictor = nn3$predict,
                 levels = c(0,1))
  case3_auc[e]<- round(auc(nn3_roc), digits = 2)
  case3_loss[e]<- mean(casl_util_mse(nn3$predict,nn3$result))
}
plot(1:25,case0_loss)
lines(1:25,case1_loss)
lines(1:25,case3_loss)
plot(1:25,case3_loss)

plot(1:25,case0_loss,type="l",col="steelblue3",lwd=5.0,xlab="epochs",ylab="loss")
lines(1:25,case1_loss,col="sienna1",lwd=5.0)
legend("topright",legend=c("SGD","SGD with momentum"),col=c("steelblue3","sienna1"),lty=c(1,1),lwd=3,cex=1.5)

plot(1:25,case0_loss,type="l",col="steelblue3",lwd=5.0,xlab="epochs",ylab="loss")
lines(1:25,case2_loss,col="lightpink2",lwd=5.0)
legend("topright",legend=c("SGD","SGD with regularization"),col=c("steelblue3","lightpink2"),lty=c(1,1),lwd=3,cex=1.5)

plot(1:25,case0_auc,type="l",col="steelblue3",lwd=3.5,xlab="epochs",ylab="auc")
lines(1:25,case1_auc,col="sienna1",lwd=3.0)
lines(1:25,case2_auc,col="lightpink2",lwd=3.0)
legend("bottomright",legend=c("SGD","SGD with momentum","SGD with regularization"),col=c("steelblue3","sienna1","lightpink2"),lty=c(1,1,1),lwd=3,cex=1.5)

lines(1:25,case3_loss,col="limegreen",lwd=5.0)
# case 4 sag
case4_auc<-c()
case4_loss<-c()
for (e in 1:100){
  weights4 <- casl_nnmulti_sag_sgd(X, y, sizes=c(1,25,2),
                                   epochs=e, eta=0.001,mu=0.09)
  y_pred4 <- casl_nnmulti_predict(weights4, X)
  y_predd4<-data.matrix(as.numeric(y_pred4[,1]>0.5))
  nn4<- data.frame(result=y_true,predict=y_pred4[,1])
  nn4_roc <- roc(response = nn4$result,
                 predictor = nn4$predict,
                 levels = c(0,1))
  case4_auc[e]<- round(auc(nn4_roc), digits = 2)
  case4_loss[e]<- mean(casl_util_mse(nn4$predict,nn4$result))
}
# loss
plot(1:100,case4_loss,type="l",col="orange",lwd=5.0,xlab="epochs",ylab="loss")
lines(1:100,case5_loss,col="green",lwd=5.0)
legend("topright",c("SAG","SAGA"),col=c("orange","green"),lty=c(1,1),lwd=3,cex=1.5)

# auc
plot(1:25,case4_auc,type="l",col="orange",lwd=3.5,xlab="epochs",ylab="auc")
lines(1:25,case5_auc,col="green",lwd=3.0)
legend("topleft",legend=c("SAG","SAGA"),col=c("orange","green"),lty=c(1,1),lwd=3,cex=1.5)


# case 5 saga
case5_auc<-c()
case5_loss<-c()
for (e in 1:100){
  weights5 <- casl_nnmulti_saga_sgd(X, y, sizes=c(1,25,2),
                                    epochs=e, eta=0.001,mu=0.09)
  y_pred5 <- casl_nnmulti_predict(weights5, X)
  y_predd5<-data.matrix(as.numeric(y_pred5[,1]>0.5))
  nn5<- data.frame(result=y_true,predict=y_pred5[,1])
  nn5_roc <- roc(response = nn5$result,
                 predictor = nn5$predict,
                 levels = c(0,1))
  case5_auc[e]<- round(auc(nn5_roc), digits = 2)
  case5_loss[e]<- mean(casl_util_mse(nn5$predict,nn5$result))
}

plot(X,y)
lines(X,y_pred,col="#fd6467")
