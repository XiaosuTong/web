

ker <- function(X) {
  X %*% t(X)
}



takeStep <- function(i1, i2, X, Y, C, E2, alpha, b, kernel){
  if (i1 == i2) {
		return(list(flag = 0, alpha = alpha, b = b))
	}
	eps <- 0.001
	alph1 <- alpha[i1] ##Lagrange multiplier for i1
	alph2 <- alpha[i2]
	y1 <- Y[i1]
	y2 <- Y[i2]
	u1 <- sum(kernel[i1, ]*Y*alpha) + b
	E1 <- u1 - y1 ##SVM output on point[i1] – y1 (check in error cache)
	s <- y1*y2
	if (s == 1) { ##Compute L, H via equations (13) and (14)
		L <- max(0, alph1 + alph2 - C) 
		H <- min(C, alph2 + alph1)
	}else {
		L <- max(0, alph2 - alph1)
		H <- min(C, C + alph2 - alph1)
	} 
	if (L == H) {
		return(list(flag = 0, alpha = alpha, b = b))
	}
	k11 <- kernel[i1, i1]
	k12 <- kernel[i1, i2]
	k22 <- kernel[i2, i2]
	eta <- k11 + k22 - 2*k12
	if (eta > 0) {
		a2 <- alph2 + y2*(E1 - E2)/eta
		if (a2 < L) {
			a2 <- L
		}else if (a2 > H) {
			a2 <- H
		}
	}else {
		f1 <- y1*(E1 - b) - alph1*k11 - s*alph2*k12
		f2 <- y2*(E2 - b) - s*alph1*k12 - alph2*k22
		L1 <- alph1 + s*(alph2 - L)
		H1 <- alph1 + s*(alph2 - H)
		Lobj <- L1*f1 + L*f2 + 1/2*L1^2*k11 + 1/2*L^2*k22 + s*L*L1*k12 ##objective function at a2=L
		Hobj <- H1*f1 + H*f2 + 1/2*H1^2*k11 + 1/2*H^2*k22 + s*H*H1*k12 ##objective function at a2=H
		if (Lobj < Hobj - eps) {
			a2 <- L
		}else if(Lobj > Hobj + eps) {
			a2 <- H
		}else {
			a2 <- alph2
		}
	}
	if (abs(a2 - alph2) < eps*(a2 + alph2 + eps)) {
		return(list(flag = 0, alpha = alpha, b = b))
	}
	a1 <- alph1 + s*(alph2 - a2)
	##Update threshold to reflect change in Lagrange multipliers
	b1 <- b - (E1 + y1*(a1 - alph1)*k11 + y2*(a2 - alph2)*k12)  
	b2 <- b - (E2 + y1*(a1 - alph1)*k12 + y2*(a2 - alph2)*k22)  
	if(a1 < C & a1 > 0){
		b <- b1
	}else if(a2 < C & a2 > 0){
		b <- b2
	}else {
		b <- (b1 + b2)/2
	}
	##Update weight vector to reflect change in a1 & a2, if SVM is linear
	##Update error cache using new Lagrange multipliers
	alpha[i1] <- a1 ##Store a1 in the alpha array
	alpha[i2] <- a2 ##Store a2 in the alpha array
	return(list(flag = 1, alpha = alpha, b = b))
}



examineExample <- function(i2, X, Y, alpha, b, C, kernel){
  y2 <- Y[i2]
	tol <- 0.001
	alph2 <- alpha[i2]
	u2 <- sum(kernel[i2, ]*Y*alpha) + b
	E2 <- u2 - y2 ##SVM output on point[i2] – y2 (check in error cache)
	r2 <- E2*y2 ##r2 = y2*(u2 - y2) = y2*u2 - 1 = 0 is KKT condition, it will be used to check KTT violation
	if ((r2 < -tol && alph2 < C) | (r2 > tol && alph2 > 0)) {
		index <- which(alpha != C & alpha != 0)
		if (sum(alpha != C & alpha != 0) > 1) { 
			E1 <- kernel[index, ] %*% (Y*alpha) + rep(b, length(index))
			diff <- which.max(abs(E1 - rep(E2, length(E1))))
			i1 <- index[diff] ##result of second choice heuristic (section 2.2)
			result <- takeStep(i1, i2, X, Y, C, E2, alpha, b, kernel)
			if(result$flag) {
				return(list(flag = 1, alpha = result$alpha, b = result$b))
			}
		}
		##loop over all non-zero and non-C alpha, starting at a random point
		for(i in sample(index, length(index))) {
			i1 <- i
			result <- takeStep(i1, i2, X, Y, C, E2, alpha, b, kernel)
			if(result$flag) {
				return(list(flag = 1, alpha = result$alpha, b = result$b))
			}
		}
		##loop over all possible i1, starting at a random point
		for(j in sample(1:length(Y), length(Y))) {
			i1 <- j
			result <- takeStep(i1, i2, X, Y, C, E2, alpha, b, kernel)
			if(result$flag) {
				return(list(flag = 1, alpha = result$alpha, b = result$b))
			}
		}
	}
	return(list(flag = 0, alpha = alpha, b = b))
}



SMO <- function(mean1, mean2, sd1, sd2, seed1, seed2) {
  library(MASS)
	set.seed(seed1)
	x1 <- mvrnorm(1000, mean1, sd1)
	set.seed(seed2)
	x2 <- mvrnorm(1000, mean2, sd2)
	X <- rbind(x1,x2)
	Y<- rep(c(1, -1), each = 1000)
	kernel <- ker(X)

	alpha <- rep(1,length(Y))
	b <- 0
	numChanged <- 0
	examineAll <- 1
  C <- 1

	while (numChanged > 0 | examineAll) {
		numChanged <- 0
		if (examineAll) {
			for(I in 1:length(Y)) { ##loop I over all training examples
				result <- examineExample(I, X, Y, alpha, b, C, kernel)
				numChanged <- numChanged + result$flag
				alpha <- result$alpha
				b <- result$b
			}
		}else {
			for(I in which(alpha != C & alpha != 0)) { ##loop I over examples where alpha is not 0 & not C
				result <- examineExample(I, X, Y, alpha, b, C, kernel)
				numChanged <- numChanged + result$flag
				alpha <- result$alpha
				b <- result$b
			}
		}
		if (examineAll == 1) {
			examineAll <- 0
		}else if (numChanged == 0) {
			examineAll <- 1
		}
	}
	w <- matrix(Y*alpha, nrow = 1) %*% X
	list(w,b)
}

SMO(mean1, mean2, sd1, sd2, seed1, seed2)

