/*****************************************************/
/*                                                   */
/* build with $ gcc nn_run.c -o nn_run               */
/*                                                   */
/* run with $ ./nn_run testing_images testing_labels */
/*                                                   */
/*****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "nn.dat"

#define NUM_LAYERS 6

float activation(int p, float z){

  float ret;
  
  if (p == 0){ // sigmoid
        ret=1/(1+exp(-z));
  } else {
    if (p == 1){ // SoftPlus
      ret=log(1+exp(z));
    } else {
      if (p == 2){ // identity
        ret=z;
      } else {
	if (p == 3){ //squaring
          ret=z*z;
	} else {
          if (p == 7){ // squaring scaled for [-8, 8]
            ret = (z/8.0)*(z/8.0);
          } else {
            if (p == 8){ // squaring scaled for [-300, 300]
              ret = (z/300.0)*(z/300.0);
            }
          }
	}
      }
    }
  }
  return ret;
}

void run_testing_net(int p[NUM_LAYERS + 1],
	             float a0[28][28],
	             float w1[5][5][5],
	             float w3[10][5][5],
	             float b5[100], float w5[100][10][5][5][5],
	             float b6[10], float w6[10][100],
	             float a6[10]){

  float z1[5][13][13];
  float a1[5][13][13];
  float a2[5][13][13];
  float z3[10][5][5][5];
  float a3[10][5][5][5];
  float a4[10][5][5][5];
  float z5[100];
  float a5[100];
  float z6[10];
  
  int j, k, m, n, r, c, a, b;

  // apply layer 1
  for (m=0; m<5; m++){
    for (c=0; c<13; c++){
      for (r=0; r<13; r++){
        z1[m][r][c] = 0;
	for (b=0; b<5; b++){
	  for (a=0; a<5; a++){
	    if ((2*r + a < 28) && (2*c + b < 28)){
              z1[m][r][c]+= w1[m][a][b]*a0[2*r + a][2*c + b];
	    }
	  }
	}
        a1[m][r][c]=activation(p[1], z1[m][r][c]);
      }
    }
  }

  // apply layer 2
  for (m=0; m<5; m++){
    for (c=0; c<13; c++){
      for (r=0; r<13; r++){
	a2[m][r][c]=0;
        for (b=0; b<3; b++){
	  for (a=0; a<3; a++){
	    if ((r + a < 13) && (c + b < 13)){
              a2[m][r][c]+=a1[m][r + a][c + b]/(float)9;
	    }
	  }
	}
      }
    }
  }

  // apply layer 3
  for (n=0; n<10; n++){
    for (m=0; m<5; m++){
      for (c=0; c<5; c++){
        for (r=0; r<5; r++){
          z3[n][m][r][c]=0;
	  for (b=0; b<5; b++){
	    for (a=0; a<5; a++){
	      z3[n][m][r][c]+= w3[n][a][b]*a2[m][2*r + a][2*c + b];
	    }
	  }
	  a3[n][m][r][c]=activation(p[3], z3[n][m][r][c]);
	}
      }
    }
  }

  // apply layer 4
  for (n=0; n<10; n++){
    for (m=0; m<5; m++){
      for (c=0; c<5; c++){
        for (r=0; r<5; r++){
	  a4[n][m][r][c]=0;
          for (b=0; b<3; b++){
	    for (a=0; a<3; a++){
	      if ((r + a < 5) && (c + b < 5)){
                a4[n][m][r][c]+=a3[n][m][r + a][c + b]/(float)9;
	      }
	    }
	  }
	}
      }
    }
  }
  
  // apply layer 5
  for (j=0; j<100; j++){
    z5[j]=b5[j];
    for (n=0; n<10; n++){
      for (m=0; m<5; m++){
        for (c=0; c<5; c++){
	  for (r=0; r<5; r++){
            z5[j]+=w5[j][n][m][r][c]*a4[n][m][r][c];
	  }
	}
      }
    }
    a5[j]=activation(p[5], z5[j]);
  }

  // apply layer 6
  for (j=0; j<10; j++){
    z6[j]=b6[j];
    for (k=0; k<100; k++){
      z6[j]+=w6[j][k]*a5[k];
    }
    a6[j]=activation(p[6], z6[j]);
  }
}

int main(int argc, char* argv[]){

  // flush output buffer instantly for debugging purposes
  setbuf(stdout, NULL);

  // set activation function per layer;
  // 0 => sigmoid;
  // 1 => SoftPlus (approx. to ReLU);
  // 2 => identity;
  // 3 => squaring
  // 7 => squaring scaled to [-8, 8]
  // 8 => squaring scaled to [-300, 300]
  int p[NUM_LAYERS + 1] = {0, 7, 2, 2, 2, 8, 0};

  // activated neurons by layer
  float a0[28][28];
  float a1[5][13][13];
  float a2[5][13][13];
  float a3[10][5][5][5];
  float a4[10][5][5][5];
  float a5[100];
  float a6[10];

  // pre-activated neurons by layer
  float z1[5][13][13];
  float z3[10][5][5][5];
  float z5[100];
  float z6[10];

  // weights and biases by layer
  float w1[5][5][5];
  float w3[10][5][5];
  float b5[100];
  float w5[100][10][5][5][5];
  float b6[10];
  float w6[10][100];

  // testing data
  FILE *fptr_testing_images = fopen(argv[1],"rb");
  FILE *fptr_testing_labels = fopen(argv[2],"rb");
  
  unsigned char testing_image[28*28];
  unsigned char testing_label;
  
  int i, j, jj, t;
  
  init_net(w1,
	   w3,
	   b5, w5,
	   b6, w6);

  int m, a, b, n, k;
  
  printf("\nBeginning testing...\n");
  for (i=1; i<=10000; i++){
    
    // load sample testing image and its label
    fread(&testing_image, 784, 1, fptr_testing_images);
    fread(&testing_label, 1, 1, fptr_testing_labels);

    // load sample image into layer 0
    for (j=0; j<28; j++){
      for (jj=0; jj<28; jj++){
        a0[j][jj]=(float)testing_image[jj + j*28]/(float)255;
      }
    }
    
    // run net on sample image
    run_testing_net(p,
	            a0,
	            w1,
	            w3,
	            b5, w5,
		    b6, w6,
	            a6);

    // print label and output layer
    printf("%d    ", testing_label);
    for (j=0; j<10; j++){
      printf("%4.3f    ", a6[j]);
    }
    printf("\n");
  }

  // close infiles
  fclose(fptr_testing_images);
  fclose(fptr_testing_labels);
}
