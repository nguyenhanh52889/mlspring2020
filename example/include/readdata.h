#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

using namespace std;

vector <vector <float>> readfile(const char* path_file)
{
  vector <vector <float>> output;
  string line;
  vector<string> line_v;

  cout << "Loading data ...\n";
  vector<float> X_train;
  vector<float> y_train;
  ifstream myfile (path_file);
  if (myfile.is_open())
  {
      while ( getline (myfile,line) )
      {
          line_v = split(line, '\t');
          int digit = strtof((line_v[0]).c_str(),0);
          for (unsigned i = 0; i < 10; ++i) {
              if (i == digit)
              {
                  y_train.push_back(1.);
              }
              else y_train.push_back(0.);
          }

          int size = static_cast<int>(line_v.size());
          for (unsigned i = 1; i < size; ++i) {
              X_train.push_back(strtof((line_v[i]).c_str(),0));
          }
      }
      X_train = X_train/255.0;
      myfile.close();
  }
  else cout << "Unable to open file" << '\n';
  output.push_back(X_train);
  output.push_back(y_train);
  return output;
}
