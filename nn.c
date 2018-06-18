/*********************************************************************************/
/*                                                                               */
/* build with $ gcc nn.c -o nn                                                   */
/*                                                                               */
/* run with $ ./nn training_images training_labels testing_images testing_labels */
/*                                                                               */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

//#include "nn.dat"

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
	if (p == 3){ // signed squaring
	  if (z >= 0){
            ret=z*z;
	  } else {
            ret=-z*z;
	  }
	} else {
          if (p == 4){ // cubing
            ret=z*z*z;
	  } else {
	    if (p == 5){ // cubic approx to sigmoid on [-8,8]
	      ret = 0.5 -1.20096*(-z/8.0) + 0.81562*(-z/8.0)*(-z/8.0)*(-z/8.0);
	    } else {
              if (p == 6){ // cubic approx to sigmoid on [-50, 50]
                ret = 0.5 -1.20096*(-z/50.0) + 0.81562*(-z/50.0)*(-z/50.0)*(-z/50.0);
	      } else {
                if (p == 7){ // squaring scaled to [-8, 8] 
                  ret = (z/8.0)*(z/8.0);
		} else {
                  if (p == 8){ // squaring scaled to [-300, 300]
                    ret = (z/300.0)*(z/300.0);
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  return ret;
}

float activation_der(int p, float a, float z){

  float ret;
  
  if (p == 0){ // sigmoid
    ret=a*(1-a);
  } else {
    if (p == 1){ // SoftPlus
      ret=1-exp(-a);
    } else {
      if (p == 2){ // identity
        ret=1;
      } else {
        if (p == 3){ // signed squaring
	  if (z >= 0){
            ret=2*z;
	  } else {
	    ret=-2*z;
	  }
	} else {
          if (p == 4){ // cubing
            ret=3*z*z;
	  } else {
            if (p == 5){ // cubic approx to sigmoid on [-8, 8] 
              ret=-1.20096*(-1/8.0) + 3.0*0.81562*(-z/8.0)*(-z/8.0)*(-1/8.0);
            } else {
              if (p == 6){ // cubic approx to sigmoid on [-50, 50]
                ret=-1.20096*(-1/50.0) + 3.0*0.81562*(-z/50.0)*(-z/50.0)*(-1/50.0);
              } else {
                if (p == 7){ // squaring scaled to [-8, 8]
                  ret = 2*(z/8.0)*(1/8.0);
		} else {
                  if (p == 8){ // squaring scaled to [-300, 300]
                    ret = 2*(z/300.0)*(1/300.0);
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  return ret;
}

void init_net(float w1[5][5][5],
	      float w3[10][5][5],
	      float b5[100], float w5[100][10][5][5][5],
	      float b6[10], float w6[10][100]){

  int j, k, m, n, r, c, a, b;
  
  // init layer 1
  for (m=0; m<5; m++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
        w1[m][a][b]=(2*((float)rand()/(float)RAND_MAX)-1)/(float)1.0;
      }
    }
  }

  // no init needed for layer 2 (mean pool),
  // weights are always 0 or 1/9
  // biases are always 0
  
  // init layer 3
  for (n=0; n<10; n++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
        w3[n][a][b]=(2*((float)rand()/(float)RAND_MAX)-1)/(float)1.0;
      }
    }
  }

  // no init needed for layer 4 (mean pool),
  // weights are always 0 or 1/9
  // biases are always 0
  
  // init layer 5
  for (j=0; j<100; j++){
    b5[j]=2*((float)rand()/(float)RAND_MAX)-1;
    for (n=0; n<10; n++){
      for (m=0; m<5; m++){
        for (c=0; c<5; c++){
	  for (r=0; r<5; r++){
            w5[j][n][m][r][c]=(2*((float)rand()/(float)RAND_MAX)-1)/(float)1.0;
	  }
	}
      }
    }
  }

  // init layer 6
  for (j=0; j<10; j++){
    b6[j]=2*((float)rand()/(float)RAND_MAX)-1;
    for (k=0; k<100; k++){
      w6[j][k]=(2*((float)rand()/(float)RAND_MAX)-1)/(float)1.0;
    }
  }
}

void run_training_net(int p[NUM_LAYERS + 1],
	              float a0[28][28],
	              float w1[5][5][5], float z1[5][13][13], float a1[5][13][13],
	              float a2[5][13][13],
	              float w3[10][5][5], float z3[10][5][5][5], float a3[10][5][5][5],
		      float a4[10][5][5][5],
	              float b5[100], float w5[100][10][5][5][5], float z5[100], float a5[100],
		      float b6[10], float w6[10][100], float z6[10], float a6[10]){

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
	//printf("%f\n", z1[m][r][c]);
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
	  //printf("%f\n", z3[n][m][r][c]);
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
    //printf("%f\n", z5[j]);
    a5[j]=activation(p[5], z5[j]);
  }

  // apply layer 6
  for (j=0; j<10; j++){
    z6[j]=b6[j];
    for (k=0; k<100; k++){
      z6[j]+=w6[j][k]*a5[k];
    }
    //printf("z6[%d] = %f\n", j, z6[j]);
    a6[j]=activation(p[6], z6[j]);
  }
}

void update_grad(int p[NUM_LAYERS + 1],
		 float y[10],
		 float a6[10],
		 float z6[10],
		 float b6_grad[10],
		 float w6[10][100],
		 float w6_grad[10][100],
		 float a5[100],
		 float z5[100],
		 float b5_grad[100],
		 float w5[100][10][5][5][5],
		 float w5_grad[100][10][5][5][5],
		 float a4[10][5][5][5],
		 float a3[10][5][5][5],
		 float z3[10][5][5][5],
		 float w3[10][5][5],
		 float w3_grad[10][5][5],
		 float a2[5][13][13],
		 float a1[5][13][13],
		 float z1[5][13][13],
		 float w1_grad[5][5][5],
		 float a0[28][28]){

  float dC_by_da6[10];
  float da6_by_dz6[10];
  float dz6_by_db6[10];
  float dC_by_db6[10];
  float dz6_by_dw6[10][100];
  float dC_by_dw6[10][100];
  
  float dz6_by_da5[10][100];
  
  float dC_by_da5[100];
  float da5_by_dz5[100];
  float dz5_by_db5[100];
  float dC_by_db5[100];
  float dz5_by_dw5[100][10][5][5][5];
  float dC_by_dw5[100][10][5][5][5];
  
  float dz5_by_da4[100][10][5][5][5];

  float dC_by_da4[10][5][5][5];
  
  float dC_by_da3[10][5][5][5];
  float da3_by_dz3[10][5][5][5];
  float dz3_by_dw3[10][5][5][5][5][5];
  float dC_by_dw3[10][5][5];
      
  float dz3_by_da2[10][5][5][5][13][13];

  float dC_by_da2[5][13][13];
      
  float dC_by_da1[5][13][13];
  float da1_by_dz1[5][13][13];
  float dz1_by_dw1[5][13][13][5][5];
  float dC_by_dw1[5][5][5];

  int j, k, n, m, r, c, a, b, rr, cc;

  // update layer 6
  for (j=0; j<10; j++){
    dC_by_da6[j]=a6[j] - y[j];
    da6_by_dz6[j]=activation_der(p[6], a6[j], z6[j]);
    dz6_by_db6[j]=1;
    dC_by_db6[j]=dC_by_da6[j]*da6_by_dz6[j]*dz6_by_db6[j];
    b6_grad[j]+=dC_by_db6[j];
    for (k=0; k<100; k++){
      dz6_by_dw6[j][k]=a5[k];
      dC_by_dw6[j][k]=dC_by_da6[j]*da6_by_dz6[j]*dz6_by_dw6[j][k];
      w6_grad[j][k]+=dC_by_dw6[j][k];
      dz6_by_da5[j][k]=w6[j][k];
    }
  }
  
  // update layer 5
  for (j=0; j<100; j++){
    dC_by_da5[j]=0;
    for (k=0; k<10; k++){
      dC_by_da5[j]+=dC_by_da6[k]*da6_by_dz6[k]*dz6_by_da5[k][j];
    }
    da5_by_dz5[j]=activation_der(p[5], a5[j], z5[j]);
    dz5_by_db5[j]=1;
    dC_by_db5[j]=dC_by_da5[j]*da5_by_dz5[j]*dz5_by_db5[j];
    b5_grad[j]+=dC_by_db5[j];
    for (n=0; n<10; n++){
      for (m=0; m<5; m++){
        for (c=0; c<5; c++){
	  for (r=0; r<5; r++){
            dz5_by_dw5[j][n][m][r][c]=a4[n][m][r][c];
	    dC_by_dw5[j][n][m][r][c]=dC_by_da5[j]*da5_by_dz5[j]*dz5_by_dw5[j][n][m][r][c];
            w5_grad[j][n][m][r][c]+=dC_by_dw5[j][n][m][r][c];
	    dz5_by_da4[j][n][m][r][c]=w5[j][n][m][r][c];
	  }
	}
      }
    }
  }

  // update layer 4
  for (n=0; n<10; n++){
    for (m=0; m<5; m++){
      for (c=0; c<5; c++){
        for (r=0; r<5; r++){
	  dC_by_da4[n][m][r][c]=0;
	  for (k=0; k<100; k++){
            dC_by_da4[n][m][r][c]+=dC_by_da5[k]*da5_by_dz5[k]*dz5_by_da4[k][n][m][r][c];
	  }
	}
      }
    }
  }

  // update layer 3
  for (n=0; n<10; n++){
    for (m=0; m<5; m++){
      for (c=0; c<5; c++){
        for (r=0; r<5; r++){
          dC_by_da3[n][m][r][c]=0;
	  for (b=0; b<3; b++){
	    for (a=0; a<3; a++){
	      if ((r - a >= 0) && (c - b >= 0)){
                dC_by_da3[n][m][r][c]+=dC_by_da4[n][m][r - a][c - b]/(float)9; // other partial in product is unity
	      }
	    }
	  }
	  da3_by_dz3[n][m][r][c]=activation_der(p[3], a3[n][m][r][c], z3[n][m][r][c]);
	}
      }
    }
  }
  for (n=0; n<10; n++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
        dC_by_dw3[n][a][b]=0;
        for (m=0; m<5; m++){
	  for (c=0; c<5; c++){
	    for (r=0; r<5; r++){
	      dz3_by_dw3[n][m][r][c][a][b]=a2[m][2*r + a][2*c + b];
              dC_by_dw3[n][a][b]+=dC_by_da3[n][m][r][c]*da3_by_dz3[n][m][r][c]*dz3_by_dw3[n][m][r][c][a][b];
	      dz3_by_da2[n][m][r][c][2*r + a][2*c + b]=w3[n][a][b];
	    }
	  }
	}
	w3_grad[n][a][b]+=dC_by_dw3[n][a][b];
      }
    }
  }

  // update layer 2
  for (m=0; m<5; m++){
    for (c=0; c<13; c++){
      for (r=0; r<13; r++){
	dC_by_da2[m][r][c]=0;
	for (cc=0; cc<5; cc++){
	  for (rr=0; rr<5; rr++){
	    for (b=0; b<5; b++){
	      for (a=0; a<5; a++){
	        if ((2*rr + a == r) && (2*cc + b == c)){
		  for (n=0; n<10; n++){
                    dC_by_da2[m][r][c]+=dC_by_da3[n][m][rr][cc]*da3_by_dz3[n][m][rr][cc]*dz3_by_da2[n][m][rr][cc][r][c];
		  }
	        }
	      }
	    }
	  }
	}
      }
    }
  }
  
  // update layer 1
  for (m=0; m<5; m++){
    for (c=0; c<13; c++){
      for (r=0; r<13; r++){
	dC_by_da1[m][r][c]=0;
	for (b=0; b<3; b++){
	  for (a=0; a<3; a++){
	    if ((r - a >= 0) && (c - b >= 0)){
              dC_by_da1[m][r][c]+=dC_by_da2[m][r - a][c - b]/(float)9; // other partial in product is unity
	    }
	  }
	}
        da1_by_dz1[m][r][c]=activation_der(p[1], a1[m][r][c], z1[m][r][c]);
      }
    }
  }
  for (m=0; m<5; m++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
	dC_by_dw1[m][a][b]=0;
	for (c=0; c<13; c++){
	  for (r=0; r<13; r++){
	    if ((2*r + a < 28) && (2*c + b < 28)){
	      dz1_by_dw1[m][r][c][a][b]=a0[2*r + a][2*c + b];
              dC_by_dw1[m][a][b]+=dC_by_da1[m][r][c]*da1_by_dz1[m][r][c]*dz1_by_dw1[m][r][c][a][b];
	    }
	  }
	}
	w1_grad[m][a][b]+=dC_by_dw1[m][a][b];
      }
    }
  }
}

void step_grad(float learning_rate,
	       int batch_size,
	       float w1[5][5][5], float w1_grad[5][5][5],
	       float w3[10][5][5], float w3_grad[10][5][5],
	       float b5[100], float b5_grad[100],
	       float w5[100][10][5][5][5], float w5_grad[100][10][5][5][5],
	       float b6[10], float b6_grad[10],
	       float w6[10][100], float w6_grad[10][100]){

  int j, k, n, m, r, c, a, b;

  // step layer 6
  for (j=0; j<10; j++){
    b6[j]-=learning_rate*b6_grad[j]/(float)batch_size;
    for (k=0; k<100; k++){
      w6[j][k]-=learning_rate*w6_grad[j][k]/(float)batch_size;
    }
  }
  
  // step layer 5
  for (j=0; j<100; j++){
    b5[j]-=learning_rate*b5_grad[j]/(float)batch_size;
    for (n=0; n<10; n++){
      for (m=0; m<5; m++){
        for (c=0; c<5; c++){
	  for (r=0; r<5; r++){
	    w5[j][n][m][r][c]-=learning_rate*w5_grad[j][n][m][r][c]/(float)batch_size;
	  }
	}
      }
    }
  }

  // step layer 3
  for (n=0; n<10; n++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
        w3[n][a][b]-=learning_rate*w3_grad[n][a][b]/(float)batch_size;
      }
    }
  }

  // step layer 1
  for (m=0; m<5; m++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
	w1[m][a][b]-=learning_rate*w1_grad[m][a][b]/(float)batch_size;
      }
    }
  }
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

  // set learning rate;
  // larger => faster learning;
  // smaller => more accurate learning
  float learning_rate = 0.01;

  // set batch size for stochastic gradient descent training
  // smaller => greater stochasticity
  int batch_size = 1;
  
  // set activation function per layer;
  // 0 => sigmoid;
  // 1 => SoftPlus (approx. to ReLU);
  // 2 => identity;
  // 3 => squaring
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

  // gradients of weights and biases by layer
  float w1_grad[5][5][5];
  float w3_grad[10][5][5];
  float b5_grad[100];
  float w5_grad[100][10][5][5][5];
  float b6_grad[10];
  float w6_grad[10][100];

  // training and testing data
  FILE *fptr_training_images;
  FILE *fptr_training_labels;
  FILE *fptr_testing_images = fopen(argv[3],"rb");
  FILE *fptr_testing_labels = fopen(argv[4],"rb");
 
  unsigned char training_image[28*28];
  unsigned char training_label;
  
  unsigned char testing_image[28*28];
  unsigned char testing_label;
  
  int i, j, jj, t;
  
  printf("Beginning training...\n");
  
  // initialize weights and biases to
  // uniform random between -1 and 1
  int seed=time(0);
  printf("%d\n", seed);
  srand(seed);
  init_net(w1,
	   w3,
	   b5, w5,
	   b6, w6);
  
  // desired output 
  float y[10];

  // prediction error
  float cost;

  // init number of trials
  int trial=0;

  int iter;
  for (iter=0; iter<10; iter++){

    // open infiles
    fptr_training_images = fopen(argv[1],"rb");
    fptr_training_labels = fopen(argv[2],"rb");
    
    // use stochastic gradient descent with batch size = batch_size
    for (t=0; t<60000; t+=batch_size){
      trial++;

      // zero out gradient
      memset(w1_grad, 0, sizeof w1_grad);
      memset(w3_grad, 0, sizeof w3_grad);
      memset(b5_grad, 0, sizeof b5_grad);
      memset(w5_grad, 0, sizeof w5_grad);
      memset(b6_grad, 0, sizeof b6_grad);
      memset(w6_grad, 0, sizeof w6_grad);

      // initialize prediction error
      cost=0;

      // run over batch
      for (i=t; i<t+batch_size; i++){

        // load next sample training image and its label
        fread(&training_image, 784, 1, fptr_training_images);
        fread(&training_label, 1, 1, fptr_training_labels);
      
        // load sample image into layer 0
        for (j=0; j<28; j++){
          for (jj=0; jj<28; jj++){
            a0[j][jj]=(float)training_image[jj + j*28]/(float)255;
	  }
        }

        // run net on sample image
        run_training_net(p,
       	                 a0,
	                 w1,
	                 z1, a1,
	                 a2,
	                 w3,
	                 z3, a3,
		         a4,
	                 b5, w5,
	                 z5, a5,
		         b6, w6,
		         z6, a6);
      
        // init and set desired output
        memset(y, 0, sizeof y);
        y[training_label]=1;

        // update cost
        for (j=0; j<10; j++){
          cost+=(a6[j]-y[j])*(a6[j]-y[j]);
        }

        update_grad(p,
       		    y,
		    a6,
		    z6,
		    b6_grad,
		    w6,
		    w6_grad,
		    a5,
		    z5,
		    b5_grad,
		    w5,
		    w5_grad,
		    a4,
		    a3,
		    z3,
		    w3,
		    w3_grad,
		    a2,
		    a1,
		    z1,
		    w1_grad,
		    a0);

      }

      cost/=2;
      cost/=(float)batch_size;
      printf("\ttrial: %10d        cost: %f\n", trial, cost);
    
      step_grad(learning_rate,
  	        batch_size,
	        w1, w1_grad,
	        w3, w3_grad,
	        b5, b5_grad,
	        w5, w5_grad,
	        b6, b6_grad,
	        w6, w6_grad);
    }

    // close infiles
    fclose(fptr_training_images);
    fclose(fptr_training_labels);
  }

  // print learned weights and biases
  int m, a, b, n, k;
  
  for (m=0; m<5; m++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
        printf("  w1[%d][%d][%d] = %20.15f;\n", m, a, b, w1[m][a][b]);
      }
    }
  }

  for (n=0; n<10; n++){
    for (b=0; b<5; b++){
      for (a=0; a<5; a++){
        printf("  w3[%d][%d][%d] = %20.15f;\n", n, a, b, w3[n][a][b]);
      }
    }
  }

  for (k=0; k<100; k++){
    printf("  b5[%d] = %20.15f;\n", k, b5[k]);
  }

  for (k=0; k<100; k++){
    for (n=0; n<10; n++){
      for (m=0; m<5; m++){
        for (b=0; b<5; b++){
          for (a=0; a<5; a++){
            printf("  w5[%d][%d][%d][%d][%d] = %20.15f;\n", k, n, m, a, b, w5[k][n][m][a][b]);
	  }
	}
      }
    }
  }

  for (k=0; k<10; k++){
    printf("  b6[%d] = %20.15f;\n", k, b6[k]);
  }

  for (k=0; k<10; k++){
    for (i=0; i<100; i++){
      printf("  w6[%d][%d] = %20.15f;\n", k, i, w6[k][i]);
    }
  }
  
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
  fclose(fptr_training_images);
  fclose(fptr_training_labels);
  fclose(fptr_testing_images);
  fclose(fptr_testing_labels);
}
