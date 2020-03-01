

#include "tool.h"
#include "readdata.h"
using namespace std;

int main(int argc, const char * argv[]) {
    cout << argv[1];
    vector <vector <float>> output = readfile(argv[1]);
    vector<float> X_train = output[0];
    vector<float> y_train = output[1];


    int xsize = static_cast<int>(X_train.size());
    int ysize = static_cast<int>(y_train.size());

    // Some hyperparameters for the NN
    int BATCH_SIZE = 256;
    float lr = .01/BATCH_SIZE;

    // Random initialization of the weights
    vector <float> W1 = random_vector(784*128);
    vector <float> W2 = random_vector(128*64);
    vector <float> W3 = random_vector(64*10);

    cout << "Training the model ...\n";
    for (unsigned i = 0; i < 10000; ++i) {

        // Building batches of input variables (X) and labels (y)
        int randindx = rand() % (42000-BATCH_SIZE);
        vector<float> b_X;
        vector<float> b_y;
        for (unsigned j = randindx*784; j < (randindx+BATCH_SIZE)*784; ++j){
            b_X.push_back(X_train[j]);
        }
        for (unsigned k = randindx*10; k < (randindx+BATCH_SIZE)*10; ++k){
            b_y.push_back(y_train[k]);
        }

        // Feed forward
        vector<float> a1 = relu(dot( b_X, W1, BATCH_SIZE, 784, 128 ));
        vector<float> a2 = relu(dot( a1, W2, BATCH_SIZE, 128, 64 ));
        vector<float> yhat = softmax(dot( a2, W3, BATCH_SIZE, 64, 10 ), 10);

        // Back propagation
        vector<float> dyhat = (yhat - b_y);
        // dW3 = a2.T * dyhat
        vector<float> dW3 = dot(transpose( &a2[0], BATCH_SIZE, 64 ), dyhat, 64, BATCH_SIZE, 10);
        // dz2 = dyhat * W3.T * relu'(a2)
        vector<float> dz2 = dot(dyhat, transpose( &W3[0], 64, 10 ), BATCH_SIZE, 10, 64) * reluPrime(a2);
        // dW2 = a1.T * dz2
        vector<float> dW2 = dot(transpose( &a1[0], BATCH_SIZE, 128 ), dz2, 128, BATCH_SIZE, 64);
        // dz1 = dz2 * W2.T * relu'(a1)
        vector<float> dz1 = dot(dz2, transpose( &W2[0], 128, 64 ), BATCH_SIZE, 64, 128) * reluPrime(a1);
        // dW1 = X.T * dz1
        vector<float> dW1 = dot(transpose( &b_X[0], BATCH_SIZE, 784 ), dz1, 784, BATCH_SIZE, 128);

        // Updating the parameters
        W3 = W3 - lr * dW3;
        W2 = W2 - lr * dW2;
        W1 = W1 - lr * dW1;


        if ((i+1) % 100 == 0){

            vector<float> loss_m = yhat - b_y;
            float loss = 0.0;
            for (unsigned k = 0; k < BATCH_SIZE*10; ++k){
                loss += loss_m[k]*loss_m[k];
            }
            cout << "                                            Loss " << loss/BATCH_SIZE <<"\n";
        };
    };
    return 0;
}
