clear
clc

%Data generation. 5 different data sets are generated with different number
%of features and weights.
n = 1000;
X = zeros(n,5,5);

NoFeatures(1) = 4;
X(:,1:NoFeatures(1),1) = rand(n, NoFeatures(1));
Y(:,1) = 1 * X(:, 1,1) - 3 * X(:, 2,1) + 1.5 * X(:, 3,1)+ 4 * X(:, 4,1)+0.3*randn(n,1);
actualW(1,:) = [1,-3,1.5,4,0];

NoFeatures(2) = 2;
X(:,1:NoFeatures(2),2) = rand(n, NoFeatures(2));
Y(:,2) = 10 * X(:, 1, 2) - 4 * X(:, 2,2)+0.5*randn(n,1);
actualW(2,:) = [10,-4,0,0,0];

NoFeatures(3) = 5;
X(:,1:NoFeatures(3),3) = rand(n, NoFeatures(3));
Y(:,3) = 10 * X(:, 1,3) + 8 * X(:, 2,3) - 4 * X(:, 3,3) + 3 * X(:, 4,3) + 1 * X(:, 5,3)+randn(n,1);
actualW(3,:) = [10,8,-4,3,1];

NoFeatures(4) = 3;
X(:,1:NoFeatures(4),4) = rand(n, NoFeatures(4));
Y(:,4) = 3.5 * X(:, 1,4) + 8 * X(:, 2,4) - 1.5 * X(:, 3,4)+randn(n,1);
actualW(4,:) = [3.5,8,-1.5,0,0];

NoFeatures(5) = 4;
X(:,1:NoFeatures(5),5) = rand(n, NoFeatures(5));
Y(:,5) = 2 * X(:, 1,5) + 4 * X(:, 2,5) + 6 * X(:, 3,5)+ 8 * X(:, 4,5)+ randn(n,1);
actualW(5,:) = [2,4,6,8,0];

for d = 1:5
    fprintf('\nData set %d...\n', d);
    fprintf('Actual weights\n');
    disp(actualW(d,1: NoFeatures(d)));
    
    fprintf('Running Gradient Descent for Square error...\n')
    Winit = zeros(NoFeatures(d),1);
    WfinErr = gradient_descent_Err(Winit, X(:,1:NoFeatures(d),d), Y(:,d));
    fprintf('Predicted weights:\n');
    disp(WfinErr');
    r2Err(d) = rigError(WfinErr, X(:,1:NoFeatures(d),d), Y(:,d));
    fprintf('R2 for Square error:%f\n',r2Err(d));
   
    fprintf('\nRunning Gradient Descent for Square distances...\n')
    Winit = ones(1, NoFeatures(d)+1);
    WfinDist = gradient_descent_Dist(Winit, X(:,1:NoFeatures(d),d), Y(:,d));
    fprintf('Predicted weights:\n');
    disp(WfinDist(2:end));
    r2Dist(d) = rigError(WfinDist(2:end)', X(:,1:NoFeatures(d),d), Y(:,d));
    fprintf('R2 for Square distance:%f\n',r2Dist(d));

    fprintf('\nRunning closed form solution...\n')
    Wcf = inv(X(:,1:NoFeatures(d),d)' * X(:,1:NoFeatures(d),d)) * X(:,1:NoFeatures(d),d)' * Y(:,d);
    fprintf('Predicted weights:\n');
    disp(Wcf')
    r2cf(d) = rigError(Wcf, X(:,1:NoFeatures(d),d), Y(:,d));
    fprintf('R2 for Square distance:%f\n',r2cf(d));
end

function W = gradient_descent_Dist(W, X, Y)
    eta = .001;
    tol = 0.0000001;
    obj = sqDist(W, X, Y);
    maxIter = 10000;
    iter = 0;
    while iter < maxIter
        Wnew = W - eta * gradDist(W, X, Y);
        if abs((sqDist(Wnew, X, Y) - obj)/sqDist(Wnew, X, Y)) <  tol
            break;
        end
        W = Wnew;
        iter = iter + 1;
        obj = sqDist(Wnew, X, Y);
    end
    return  
end 

function obj = sqDist(W, X, Y)
addTerm = 0;
for i = 1:size(X,1)
    mulTerm = 0;
    for j = 2:size(W,2)
        mulTerm = mulTerm + X(i,j-1)*W(j);
    end
    addTerm = addTerm + (W(1)+ mulTerm - Y(i)).^2;
end
obj = addTerm/sum(W(2:end).^2);
end

function g = gradDist(W, X, Y)
addTerm = 0;
for i = 1:size(X,1)
    mulTerm = 0;
    for j = 2:size(W,2)
        mulTerm = mulTerm + X(i,j-1)*W(j);
    end
    addTerm = addTerm + W(1) + mulTerm - Y(i);
end
g(1,1) = 2*addTerm/sum(W(2:end).^2);

term1 = 0;
term2 = 0;
for k = 2:size(W,2)
    for i = 1:size(X,1)
        mulTerm = 0;
        for j = 2:size(W,2)
            mulTerm = mulTerm + X(i,j-1)*W(j);
        end
        term1 = term1 + (mulTerm - Y(i))*X(i,k-1)*sum(W(2:end).^2);
        term2 = term2 + ((mulTerm - Y(i)).^2)* W(k);
    end
    g(k) = 2*(term1 - term2)/(sum(W(2:end).^2).^2);
end
end

function W = gradient_descent_Err(W, X, Y)
    eta = .1;
    tol = 0.0000001;
    obj = sqErr(W, X, Y);
    maxIter = 10000;
    iter = 0;
    while iter < maxIter
        Wnew = W - eta * gradErr(W, X, Y);
        if abs((sqErr(Wnew, X, Y) - obj)/sqErr(Wnew, X, Y)) <  tol
            break;
        end
        W = Wnew;
        iter = iter + 1;
        obj = sqErr(Wnew, X, Y);
    end
    return  
end 


function err = sqErr(W, X, Y)
err = (1/2*size(X,1)) * sum((X*W - Y).^2);
    return
end

function g = gradErr(W, X, Y)
g = (1/size(X,1)) * (X'*(X*W-Y));
return
end

function r = rigError(W, X, Y)
fx = X*W;
m = mean(Y);
r = 1-sum((fx - Y).^2)/sum((m-Y).^2);
end