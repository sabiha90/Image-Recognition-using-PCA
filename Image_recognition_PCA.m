%Read the files from the path%
path = dir('/Users/sabihabarlaskar/Documents/MATLAB/FaceRecognition_Data/ALL/small_*.TIF');

%Initialize the matrix with ones%
tau = ones(1024,35);
for i =1:35
    filename = strcat('/Users/sabihabarlaskar/Documents/MATLAB/FaceRecognition_Data/ALL/',path(i).name);
    %Read the images %
    I = double(imread(filename));
  
    %Vectorize the images into a column vector of size N^2 X 1 and store it
    %in a matrix to form an image space
    V_I = I(:);
    tau(:,i) = V_I;
    
end

%calculating the average face vector
psi = mean(tau,2);
figure('NumberTitle','off','Name','Mean image');
imshow(uint8(reshape(psi,32,32)));
size(psi);

%Subtracting the average face vector from each face vector Tau to get a set of
%vectors phi. The purpose of subtracting the mean image from each image vector is to 
%be left with only the unique features 
%from each face and removing common features
phi = tau - repmat(psi,1,35);
%A is a matrix that contains the phi of all the images
A = phi;
size(A);
C = A * A';
[eigvec,eigval] = eig(C);
eigval = diag(eigval);
[sortedeigval, eig_indices] = sort(abs(eigval),'descend');
Sorted_eig = eigvec(:,eig_indices);
size(Sorted_eig);
k = 19;
figure('NumberTitle','off','Name','Eigen faces')
top_k_eig_vec = Sorted_eig(:,1:k);
for i = 1:19
    colormap('gray');
    subplot(5,4,i);
    imagesc(reshape(top_k_eig_vec(:, i), 32, 32)); 
end

yj = top_k_eig_vec' * phi;
path1 = dir('/Users/sabihabarlaskar/Documents/MATLAB/FaceRecognition_Data/FA/small_*.TIF');


tau_train = ones(1024,12);
for i =1:12
    filename1 = strcat('/Users/sabihabarlaskar/Documents/MATLAB/FaceRecognition_Data/FA/',path1(i).name);
    %Read the images %
    I_train = double(imread(filename1));
    %Vectorize the images into a column vector of size N^2 X 1 and store it
    %in a matrix to form an image space
    V_I_train = I_train(:);
    tau_train(:,i) = V_I_train;
end

phi_train = tau_train - repmat(psi,1,12);
yj_train = top_k_eig_vec' * phi_train;


I_test = double(imread('/Users/sabihabarlaskar/Documents/MATLAB/FaceRecognition_Data/FB/small_00100FB0.tif'));
I_t = imread('/Users/sabihabarlaskar/Documents/MATLAB/FaceRecognition_Data/FB/small_00100FB0.tif');

%Initialize the matrix with ones%
tau_test = I_test(:);
size(tau_test);
phi_test = tau_test - psi;
yz = top_k_eig_vec' * phi_test;
yz_u = top_k_eig_vec * yz;
figure; 
subplot(2,2,1)
imagesc(I_t); 
title('Test Image');
subplot(2,2,2);
imagesc(reshape(yz_u,32,32)); 
colormap('gray');
title('Test image projected');


%Calculate distance to face space
%face_dist = sqrt(sum(((phi_test - yz_u)/1024).^2)')/sqrt(k);

%fprintf('Face dist: %f\n', face_dist);
%Calculate normalized Euclidean distance'
%dist = sqrt(sum(((omega_all - yz)/1024).^2)')/sqrt(k);
dist = sqrt(sum((yj - yz).^2)');
[sorted_dist,order] = sort(dist);
theta = 900;
figure('NumberTitle','off','Name','ALL images');
for i=1:10

    %subplot(3,3,i)
    %imagesc(reshape(tau(:,order(i)),32,32)); 
    %colormap('gray');
    %title(sprintf('dist: %f', sorted_dist(i)));

    subplot(3,3,1)
    imagesc(I_t); 
    colormap('gray');
    title('Test Image projected');
    
    if sorted_dist(i) < theta
        subplot(3,3,i+1)
        imagesc(reshape(tau(:,order(i)),32,32)); 
        colormap('gray');
        title(sprintf('Image %d,distance %f',i, sorted_dist(i)));
    end
end


dist_train = sqrt(sum((yj_train - yz).^2)');
[sorted_dist_train,order] = sort(dist_train);
theta = 900;
figure('NumberTitle','off','Name','FA images');
for i=1:10
    subplot(3,3,1)
    imagesc(I_t); 
    colormap('gray');
    title('Test Image');
    
    if sorted_dist(i) < theta
        subplot(3,3,i+1)
        imagesc(reshape(tau_train(:,order(i)),32,32)); 
        colormap('gray');
        title(sprintf('Image %d,distance %f',i, sorted_dist_train(i)));
    end
end

